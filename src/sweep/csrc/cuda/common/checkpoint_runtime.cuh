#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "cudautils.h"

enum class CheckpointProfileKind {
    ForwardSave,
    BackwardLoad,
    StateCopy,
    StateZero,
};

class CheckpointRuntime {
public:
    CheckpointRuntime(
        const std::vector<torch::Tensor>& checkpoints,
        int expected_tensors,
        bool enabled,
        bool recursive,
        int checkpoint_interval,
        const torch::Tensor& checkpoint_steps,
        bool checkpoint_on_cpu,
        const char* role,
        const char* label = "checkpoint"
    )
        : checkpoints_(checkpoints),
          expected_tensors_(expected_tensors),
          enabled_(enabled),
          recursive_(recursive),
          checkpoint_interval_(checkpoint_interval),
          checkpoint_steps_(checkpoint_steps),
          checkpoint_on_cpu_(checkpoint_on_cpu),
          role_(role == nullptr ? "unknown" : role),
          label_(label == nullptr ? "checkpoint" : label)
    {
        profile_enabled_ = std::getenv("SWEEP_CKPT_PROFILE") != nullptr;
        if (!enabled_)
            return;

        TORCH_CHECK(expected_tensors_ >= 0, "CheckpointRuntime expected_tensors must be non-negative");
        TORCH_CHECK(
            static_cast<int>(checkpoints_.size()) == expected_tensors_,
            label_, " checkpointing expects ", expected_tensors_, " checkpoint tensors"
        );
        TORCH_CHECK(checkpoint_interval_ >= 1, "checkpoint_interval must be >= 1");
        for (const auto& checkpoint : checkpoints_) {
            TORCH_CHECK(checkpoint.defined(), label_, " checkpoint tensor must be defined");
            TORCH_CHECK(checkpoint.is_contiguous(), label_, " checkpoint tensor must be contiguous");
            if (checkpoint_on_cpu_) {
                TORCH_CHECK(!checkpoint.is_cuda() && checkpoint.device().is_cpu(), label_, " CPU checkpoint storage expects CPU tensors");
            } else {
                TORCH_CHECK(checkpoint.is_cuda(), label_, " GPU checkpoint storage expects CUDA tensors");
            }
        }
        if (recursive_) {
            TORCH_CHECK(checkpoint_steps_.defined(), "Recursive checkpointing expects checkpoint_steps");
            TORCH_CHECK(checkpoint_steps_.dim() == 1, "checkpoint_steps must be 1-D");
        }
    }

    ~CheckpointRuntime()
    {
        print_profile_if_needed();
    }

    inline void save_forward(int it, int nt, const std::vector<torch::Tensor>& tensors)
    {
        const int checkpoint_idx = forward_checkpoint_index(it, nt);
        if (checkpoint_idx < 0)
            return;
        save(checkpoint_idx, tensors);
    }

    inline void save(int checkpoint_idx, const std::vector<torch::Tensor>& tensors)
    {
        if (!enabled_)
            return;
        check_tensor_count(tensors, expected_tensors_, "save");
        profile_group(
            CheckpointProfileKind::ForwardSave,
            static_cast<int>(tensors.size()),
            tensor_list_bytes(tensors),
            [&]() {
                for (int i = 0; i < expected_tensors_; ++i)
                    copy_to_checkpoint(i, checkpoint_idx, tensors[i]);
            }
        );
    }

    inline void load(int checkpoint_idx, const std::vector<torch::Tensor>& dst_tensors)
    {
        static const std::vector<torch::Tensor> no_zero_after;
        load(checkpoint_idx, dst_tensors, no_zero_after);
    }

    inline void load(
        int checkpoint_idx,
        const std::vector<torch::Tensor>& dst_tensors,
        const std::vector<torch::Tensor>& zero_after
    )
    {
        if (!enabled_)
            return;
        check_tensor_count(dst_tensors, expected_tensors_, "load");
        profile_group(
            CheckpointProfileKind::BackwardLoad,
            static_cast<int>(dst_tensors.size()),
            tensor_list_bytes(dst_tensors),
            [&]() {
                for (int i = 0; i < expected_tensors_; ++i)
                    copy_from_checkpoint(dst_tensors[i], i, checkpoint_idx);
            }
        );

        if (!zero_after.empty())
            zero_state(zero_after);
    }

    inline void copy_state(
        const std::vector<torch::Tensor>& dst_tensors,
        const std::vector<torch::Tensor>& src_tensors
    )
    {
        TORCH_CHECK(
            dst_tensors.size() == src_tensors.size(),
            "CheckpointRuntime state copy expects matching tensor counts"
        );
        profile_group(
            CheckpointProfileKind::StateCopy,
            static_cast<int>(src_tensors.size()),
            tensor_list_bytes(src_tensors),
            [&]() {
                for (size_t i = 0; i < src_tensors.size(); ++i)
                    copy_tensor_device_to_device_async(dst_tensors[i], src_tensors[i]);
            }
        );
    }

    inline void zero_state(const std::vector<torch::Tensor>& tensors)
    {
        profile_group(
            CheckpointProfileKind::StateZero,
            static_cast<int>(tensors.size()),
            tensor_list_bytes(tensors),
            [&]() {
                for (const auto& tensor : tensors)
                    zero_tensor_device_async(tensor);
            }
        );
    }

private:
    struct ProfileSpan {
        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
        CheckpointProfileKind kind;
        int tensors = 0;
        size_t bytes = 0;
    };

    struct ProfileTotals {
        double forward_save_s = 0.0;
        double backward_load_s = 0.0;
        double state_copy_s = 0.0;
        double state_zero_s = 0.0;
        int forward_save_calls = 0;
        int backward_load_calls = 0;
        int state_copy_calls = 0;
        int state_zero_calls = 0;
        size_t forward_save_bytes = 0;
        size_t backward_load_bytes = 0;
        size_t state_copy_bytes = 0;
        size_t state_zero_bytes = 0;
        int forward_save_tensors = 0;
        int backward_load_tensors = 0;
        int state_copy_tensors = 0;
        int state_zero_tensors = 0;
    };

    template <typename Fn>
    inline void profile_group(CheckpointProfileKind kind, int tensors, size_t bytes, Fn&& fn)
    {
        if (!profile_enabled_) {
            fn();
            return;
        }

        ProfileSpan span;
        span.kind = kind;
        span.tensors = tensors;
        span.bytes = bytes;
        C10_CUDA_CHECK(cudaEventCreate(&span.start));
        C10_CUDA_CHECK(cudaEventCreate(&span.end));
        C10_CUDA_CHECK(cudaEventRecord(span.start, at::cuda::getCurrentCUDAStream()));
        fn();
        C10_CUDA_CHECK(cudaEventRecord(span.end, at::cuda::getCurrentCUDAStream()));
        spans_.push_back(span);
    }

    inline int forward_checkpoint_index(int it, int nt)
    {
        if (!enabled_)
            return -1;

        if (recursive_) {
            const int num_steps = static_cast<int>(checkpoint_steps_.numel());
            const int* steps = checkpoint_steps_.data_ptr<int>();
            if (next_checkpoint_idx_ < num_steps && steps[next_checkpoint_idx_] == it + 1)
                return next_checkpoint_idx_++;
            return -1;
        }

        if (((it + 1) % checkpoint_interval_ == 0) && (it + 1 < nt))
            return (it + 1) / checkpoint_interval_;
        return -1;
    }

    inline void copy_to_checkpoint(int idx, int checkpoint_idx, const torch::Tensor& src)
    {
        auto dst = checkpoints_[idx].select(0, checkpoint_idx);
        copy_tensor_cuda_async(dst, src);
    }

    inline void copy_from_checkpoint(const torch::Tensor& dst, int idx, int checkpoint_idx)
    {
        auto src = checkpoints_[idx].select(0, checkpoint_idx);
        copy_tensor_cuda_async(dst, src);
    }

    inline static void check_tensor_count(
        const std::vector<torch::Tensor>& tensors,
        int expected,
        const char* op
    )
    {
        TORCH_CHECK(
            static_cast<int>(tensors.size()) == expected,
            "CheckpointRuntime ", op, " expects ", expected, " tensors, got ", tensors.size()
        );
    }

    inline static size_t tensor_bytes(const torch::Tensor& tensor)
    {
        return tensor.defined() ? static_cast<size_t>(tensor.nbytes()) : 0;
    }

    inline static size_t tensor_list_bytes(const std::vector<torch::Tensor>& tensors)
    {
        size_t bytes = 0;
        for (const auto& tensor : tensors)
            bytes += tensor_bytes(tensor);
        return bytes;
    }

    inline static void accumulate(ProfileTotals& totals, const ProfileSpan& span, double seconds)
    {
        switch (span.kind) {
        case CheckpointProfileKind::ForwardSave:
            totals.forward_save_s += seconds;
            ++totals.forward_save_calls;
            totals.forward_save_bytes += span.bytes;
            totals.forward_save_tensors += span.tensors;
            break;
        case CheckpointProfileKind::BackwardLoad:
            totals.backward_load_s += seconds;
            ++totals.backward_load_calls;
            totals.backward_load_bytes += span.bytes;
            totals.backward_load_tensors += span.tensors;
            break;
        case CheckpointProfileKind::StateCopy:
            totals.state_copy_s += seconds;
            ++totals.state_copy_calls;
            totals.state_copy_bytes += span.bytes;
            totals.state_copy_tensors += span.tensors;
            break;
        case CheckpointProfileKind::StateZero:
            totals.state_zero_s += seconds;
            ++totals.state_zero_calls;
            totals.state_zero_bytes += span.bytes;
            totals.state_zero_tensors += span.tensors;
            break;
        }
    }

    inline void print_profile_if_needed()
    {
        if (!profile_enabled_ || profile_printed_)
            return;
        profile_printed_ = true;
        if (spans_.empty())
            return;

        ProfileTotals totals;
        for (auto& span : spans_) {
            C10_CUDA_CHECK(cudaEventSynchronize(span.end));
            float milliseconds = 0.0f;
            C10_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, span.start, span.end));
            accumulate(totals, span, 0.001 * static_cast<double>(milliseconds));
            cudaEventDestroy(span.start);
            cudaEventDestroy(span.end);
            span.start = nullptr;
            span.end = nullptr;
        }

        const double total_s = totals.forward_save_s + totals.backward_load_s + totals.state_copy_s + totals.state_zero_s;
        const size_t total_bytes = totals.forward_save_bytes + totals.backward_load_bytes + totals.state_copy_bytes + totals.state_zero_bytes;
        std::fprintf(
            stderr,
            "SWEEP_CKPT_PROFILE "
            "label=%s role=%s storage=%s expected_tensors=%d "
            "forward_save_time=%.6f forward_save_calls=%d forward_save_tensors=%d forward_save_bytes=%zu "
            "backward_load_time=%.6f backward_load_calls=%d backward_load_tensors=%d backward_load_bytes=%zu "
            "state_copy_time=%.6f state_copy_calls=%d state_copy_tensors=%d state_copy_bytes=%zu "
            "state_zero_time=%.6f state_zero_calls=%d state_zero_tensors=%d state_zero_bytes=%zu "
            "total_time=%.6f total_bytes=%zu\n",
            label_.c_str(),
            role_.c_str(),
            checkpoint_on_cpu_ ? "cpu" : "gpu",
            expected_tensors_,
            totals.forward_save_s, totals.forward_save_calls, totals.forward_save_tensors, totals.forward_save_bytes,
            totals.backward_load_s, totals.backward_load_calls, totals.backward_load_tensors, totals.backward_load_bytes,
            totals.state_copy_s, totals.state_copy_calls, totals.state_copy_tensors, totals.state_copy_bytes,
            totals.state_zero_s, totals.state_zero_calls, totals.state_zero_tensors, totals.state_zero_bytes,
            total_s, total_bytes
        );
        std::fflush(stderr);
    }

    const std::vector<torch::Tensor>& checkpoints_;
    int expected_tensors_ = 0;
    bool enabled_ = false;
    bool recursive_ = false;
    int checkpoint_interval_ = 1;
    const torch::Tensor& checkpoint_steps_;
    bool checkpoint_on_cpu_ = false;
    std::string role_;
    std::string label_;
    int next_checkpoint_idx_ = 0;
    bool profile_enabled_ = false;
    bool profile_printed_ = false;
    std::vector<ProfileSpan> spans_;
};
