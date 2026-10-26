// The elastic Born backward is elastic2d's.  The scattered field's adjoint
// obeys the background elastic equation, and the gradient of the scattered
// record with respect to (dvp, dvs, drho) is that adjoint imaged against the
// background field with elastic2d's imaging condition (the Born sources are
// the derivative of the elastic step in (vp, vs, rho)).  The forward stored
// the background in elastic2d's layout (u_allt, boundary strips, last state,
// checkpoints), so each mode runs elastic2d's driver on
//   models    = {vp, vs, rho}                (the background),
//   grads_out = {grad_dvp, grad_dvs, grad_drho},
//   receivers = the scattered field ids shifted onto the elastic ones; a
//               background receiver maps to -1, whose residual is injected
//               nowhere (the background record does not depend on the
//               perturbation).
// The background and wavelet gradients are refused in Python
// (ElasticBorn.c_refuse_grads).
#include <vector>

#include "elastic_born2d.h"
#include "../elastic2d/elastic2d.h"

namespace elastic_born2d {

namespace {

constexpr int N_MODELS = 6;
constexpr int N_FIELDS = 15;

template <class Run>
BackwardOutputCore as_elastic(const BackwardInputCore& in, Run run)
{
    SWEEP_CHECK(in.models.size() == N_MODELS && in.grads_out.size() == N_MODELS,
                "elastic_born2d/backward expects the six models {vp, vs, rho, dvp, dvs, drho} "
                "and their six grads_out, got ", in.models.size(), " and ",
                in.grads_out.size());
    std::vector<Buf> models(in.models.begin(), in.models.begin() + 3);
    std::vector<Buf> grads(in.grads_out.begin() + 3, in.grads_out.end());
    std::vector<int> receivers;
    receivers.reserve(static_cast<size_t>(in.receiver_field_indices.size()));
    for (int r : in.receiver_field_indices)
        receivers.push_back(r >= N_FIELDS && r < 2 * N_FIELDS ? r - N_FIELDS : -1);

    BackwardInputCore q = in;
    q.models = BufList{models.data(), static_cast<int64_t>(models.size())};
    q.grads_out = BufList{grads.data(), static_cast<int64_t>(grads.size())};
    q.receiver_field_indices = IntSpan{receivers.data(), static_cast<int64_t>(receivers.size())};
    BackwardOutputCore out = run(q);
    out.grads.assign(in.grads_out.begin(), in.grads_out.end());
    return out;
}

} // namespace

BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    return as_elastic(in, elastic2d::backward_core);
}

BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    return as_elastic(in, elastic2d::backward_bs_core);
}

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in)
{
    return as_elastic(in, elastic2d::backward_ckpt_core);
}

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in)
{
    return as_elastic(in, elastic2d::backward_recursive_ckpt_core);
}

}
