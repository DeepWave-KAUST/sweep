// cuda/common/capi.cu -- the C boundary (core/capi.h): a switch over the entry
// enum onto the equations' core entries, runner handles, the session, the
// stream, the visco FFT query.  Every function catches everything and returns
// it as a message; the core never throws across this file.
// GENERATED table (the entry list) by the 4b apply script from the equation
// headers; keep it in step with them.
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include "../../core/capi.h"
#include "../../core/check.h"
#include "../../core/device.h"
#include "../../core/input_core.h"
#include "../../core/outputs.h"
#include "../../core/runner.h"
#include "../../shared/boundary_session.h"
#include "../equations/acoustic2d/acoustic2d.h"
#include "../equations/acoustic3d/acoustic3d.h"
#include "../equations/acoustic_lsrtm2d/acoustic_lsrtm2d.h"
#include "../equations/acoustic_lsrtm3d/acoustic_lsrtm3d.h"
#include "../equations/acoustic_vrz2d/acoustic_vrz2d.h"
#include "../equations/acoustic_vrz3d/acoustic_vrz3d.h"
#include "../equations/acoustic_vti_1st_2d/acoustic_vti_1st_2d.h"
#include "../equations/acoustic_vti_1st_3d/acoustic_vti_1st_3d.h"
#include "../equations/das2d/das2d.h"
#include "../equations/das3d/das3d.h"
#include "../equations/das_mu2d/das_mu2d.h"
#include "../equations/das_mu3d/das_mu3d.h"
#include "../equations/elastic2d/elastic2d.h"
#include "../equations/elastic3d/elastic3d.h"
#include "../equations/elastic_tti_2nd2d/elastic_tti_2nd2d.h"
#include "../equations/elastic_tti_sg2d/elastic_tti_sg2d.h"
#include "../equations/elastic_tti_sg3d/elastic_tti_sg3d.h"
#include "../equations/elastic_vr2d/elastic_vr2d.h"
#include "../equations/visco_acoustic2d/visco_acoustic2d.h"

namespace {

const char* const ENTRY_NAMES[SWEEP_ENTRY_COUNT] = {
    "acoustic2d_forward",
    "acoustic2d_backward",
    "acoustic2d_backward_bs",
    "acoustic2d_backward_ckpt",
    "acoustic2d_backward_recursive_ckpt",
    "acoustic3d_forward",
    "acoustic3d_backward",
    "acoustic3d_backward_bs",
    "acoustic3d_backward_ckpt",
    "acoustic3d_backward_recursive_ckpt",
    "acoustic_lsrtm2d_forward",
    "acoustic_lsrtm2d_backward",
    "acoustic_lsrtm2d_backward_bs",
    "acoustic_lsrtm2d_backward_ckpt",
    "acoustic_lsrtm2d_backward_recursive_ckpt",
    "acoustic_lsrtm3d_forward",
    "acoustic_lsrtm3d_backward",
    "acoustic_lsrtm3d_backward_bs",
    "acoustic_lsrtm3d_backward_ckpt",
    "acoustic_lsrtm3d_backward_recursive_ckpt",
    "acoustic_vrz2d_forward",
    "acoustic_vrz2d_backward",
    "acoustic_vrz2d_backward_bs",
    "acoustic_vrz2d_backward_ckpt",
    "acoustic_vrz2d_backward_recursive_ckpt",
    "acoustic_vrz3d_forward",
    "acoustic_vrz3d_backward",
    "acoustic_vrz3d_backward_bs",
    "acoustic_vrz3d_backward_ckpt",
    "acoustic_vrz3d_backward_recursive_ckpt",
    "acoustic_vti_1st_2d_forward",
    "acoustic_vti_1st_2d_backward",
    "acoustic_vti_1st_2d_backward_bs",
    "acoustic_vti_1st_2d_backward_ckpt",
    "acoustic_vti_1st_2d_backward_recursive_ckpt",
    "acoustic_vti_1st_3d_forward",
    "acoustic_vti_1st_3d_backward",
    "acoustic_vti_1st_3d_backward_bs",
    "acoustic_vti_1st_3d_backward_ckpt",
    "acoustic_vti_1st_3d_backward_recursive_ckpt",
    "das2d_forward",
    "das2d_backward",
    "das2d_backward_bs",
    "das2d_backward_ckpt",
    "das2d_backward_recursive_ckpt",
    "das3d_forward",
    "das3d_backward",
    "das3d_backward_bs",
    "das3d_backward_ckpt",
    "das3d_backward_recursive_ckpt",
    "das_mu2d_forward",
    "das_mu2d_backward",
    "das_mu2d_backward_bs",
    "das_mu2d_backward_ckpt",
    "das_mu2d_backward_recursive_ckpt",
    "das_mu3d_forward",
    "das_mu3d_backward",
    "das_mu3d_backward_bs",
    "das_mu3d_backward_ckpt",
    "das_mu3d_backward_recursive_ckpt",
    "elastic2d_forward",
    "elastic2d_backward",
    "elastic2d_backward_bs",
    "elastic2d_backward_ckpt",
    "elastic2d_backward_recursive_ckpt",
    "elastic2d_apm_forward",
    "elastic3d_forward",
    "elastic3d_backward_bs",
    "elastic3d_backward_ckpt",
    "elastic3d_backward_recursive_ckpt",
    "elastic3d_backward",
    "elastic3d_apm_forward",
    "elastic_tti_2nd2d_forward",
    "elastic_tti_2nd2d_backward",
    "elastic_tti_2nd2d_backward_bs",
    "elastic_tti_2nd2d_backward_ckpt",
    "elastic_tti_sg2d_forward",
    "elastic_tti_sg2d_backward",
    "elastic_tti_sg2d_backward_bs",
    "elastic_tti_sg2d_backward_ckpt",
    "elastic_tti_sg3d_forward",
    "elastic_tti_sg3d_backward",
    "elastic_tti_sg3d_backward_bs",
    "elastic_tti_sg3d_backward_ckpt",
    "elastic_vr2d_forward",
    "elastic_vr2d_backward",
    "elastic_vr2d_backward_bs",
    "elastic_vr2d_backward_ckpt",
    "elastic_vr2d_backward_recursive_ckpt",
    "visco_acoustic2d_forward",
    "visco_acoustic2d_backward",
    "visco_acoustic2d_backward_bs",
    "visco_acoustic2d_backward_ckpt",
    "visco_acoustic2d_backward_recursive_ckpt"
};
const int ENTRY_KINDS[SWEEP_ENTRY_COUNT] = { 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1 };

void fill_err(char* err, int cap, const char* what)
{
    if (err == nullptr || cap <= 0) return;
    std::strncpy(err, what, static_cast<size_t>(cap - 1));
    err[cap - 1] = '\0';
}

template <class F>
int guarded(F f, char* err, int cap)
{
    try { f(); return 0; }
    catch (const std::exception& e) { fill_err(err, cap, e.what()); return 1; }
    catch (...) { fill_err(err, cap, "sweep core: unknown exception"); return 1; }
}

void dispatch(int entry, const void* in, void* out)
{
    switch (entry) {
        case 0: *static_cast<ForwardOutputCore*>(out) = acoustic2d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 1: *static_cast<BackwardOutputCore*>(out) = acoustic2d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 2: *static_cast<BackwardOutputCore*>(out) = acoustic2d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 3: *static_cast<BackwardOutputCore*>(out) = acoustic2d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 4: *static_cast<BackwardOutputCore*>(out) = acoustic2d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 5: *static_cast<ForwardOutputCore*>(out) = acoustic3d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 6: *static_cast<BackwardOutputCore*>(out) = acoustic3d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 7: *static_cast<BackwardOutputCore*>(out) = acoustic3d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 8: *static_cast<BackwardOutputCore*>(out) = acoustic3d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 9: *static_cast<BackwardOutputCore*>(out) = acoustic3d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 10: *static_cast<ForwardOutputCore*>(out) = acoustic_lsrtm2d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 11: *static_cast<BackwardOutputCore*>(out) = acoustic_lsrtm2d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 12: *static_cast<BackwardOutputCore*>(out) = acoustic_lsrtm2d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 13: *static_cast<BackwardOutputCore*>(out) = acoustic_lsrtm2d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 14: *static_cast<BackwardOutputCore*>(out) = acoustic_lsrtm2d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 15: *static_cast<ForwardOutputCore*>(out) = acoustic_lsrtm3d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 16: *static_cast<BackwardOutputCore*>(out) = acoustic_lsrtm3d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 17: *static_cast<BackwardOutputCore*>(out) = acoustic_lsrtm3d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 18: *static_cast<BackwardOutputCore*>(out) = acoustic_lsrtm3d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 19: *static_cast<BackwardOutputCore*>(out) = acoustic_lsrtm3d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 20: *static_cast<ForwardOutputCore*>(out) = acoustic_vrz2d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 21: *static_cast<BackwardOutputCore*>(out) = acoustic_vrz2d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 22: *static_cast<BackwardOutputCore*>(out) = acoustic_vrz2d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 23: *static_cast<BackwardOutputCore*>(out) = acoustic_vrz2d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 24: *static_cast<BackwardOutputCore*>(out) = acoustic_vrz2d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 25: *static_cast<ForwardOutputCore*>(out) = acoustic_vrz3d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 26: *static_cast<BackwardOutputCore*>(out) = acoustic_vrz3d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 27: *static_cast<BackwardOutputCore*>(out) = acoustic_vrz3d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 28: *static_cast<BackwardOutputCore*>(out) = acoustic_vrz3d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 29: *static_cast<BackwardOutputCore*>(out) = acoustic_vrz3d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 30: *static_cast<ForwardOutputCore*>(out) = acoustic_vti_1st_2d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 31: *static_cast<BackwardOutputCore*>(out) = acoustic_vti_1st_2d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 32: *static_cast<BackwardOutputCore*>(out) = acoustic_vti_1st_2d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 33: *static_cast<BackwardOutputCore*>(out) = acoustic_vti_1st_2d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 34: *static_cast<BackwardOutputCore*>(out) = acoustic_vti_1st_2d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 35: *static_cast<ForwardOutputCore*>(out) = acoustic_vti_1st_3d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 36: *static_cast<BackwardOutputCore*>(out) = acoustic_vti_1st_3d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 37: *static_cast<BackwardOutputCore*>(out) = acoustic_vti_1st_3d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 38: *static_cast<BackwardOutputCore*>(out) = acoustic_vti_1st_3d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 39: *static_cast<BackwardOutputCore*>(out) = acoustic_vti_1st_3d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 40: *static_cast<ForwardOutputCore*>(out) = das2d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 41: *static_cast<BackwardOutputCore*>(out) = das2d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 42: *static_cast<BackwardOutputCore*>(out) = das2d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 43: *static_cast<BackwardOutputCore*>(out) = das2d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 44: *static_cast<BackwardOutputCore*>(out) = das2d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 45: *static_cast<ForwardOutputCore*>(out) = das3d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 46: *static_cast<BackwardOutputCore*>(out) = das3d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 47: *static_cast<BackwardOutputCore*>(out) = das3d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 48: *static_cast<BackwardOutputCore*>(out) = das3d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 49: *static_cast<BackwardOutputCore*>(out) = das3d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 50: *static_cast<ForwardOutputCore*>(out) = das_mu2d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 51: *static_cast<BackwardOutputCore*>(out) = das_mu2d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 52: *static_cast<BackwardOutputCore*>(out) = das_mu2d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 53: *static_cast<BackwardOutputCore*>(out) = das_mu2d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 54: *static_cast<BackwardOutputCore*>(out) = das_mu2d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 55: *static_cast<ForwardOutputCore*>(out) = das_mu3d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 56: *static_cast<BackwardOutputCore*>(out) = das_mu3d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 57: *static_cast<BackwardOutputCore*>(out) = das_mu3d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 58: *static_cast<BackwardOutputCore*>(out) = das_mu3d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 59: *static_cast<BackwardOutputCore*>(out) = das_mu3d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 60: *static_cast<ForwardOutputCore*>(out) = elastic2d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 61: *static_cast<BackwardOutputCore*>(out) = elastic2d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 62: *static_cast<BackwardOutputCore*>(out) = elastic2d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 63: *static_cast<BackwardOutputCore*>(out) = elastic2d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 64: *static_cast<BackwardOutputCore*>(out) = elastic2d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 65: *static_cast<ForwardOutputCore*>(out) = elastic2d::apm_forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 66: *static_cast<ForwardOutputCore*>(out) = elastic3d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 67: *static_cast<BackwardOutputCore*>(out) = elastic3d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 68: *static_cast<BackwardOutputCore*>(out) = elastic3d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 69: *static_cast<BackwardOutputCore*>(out) = elastic3d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 70: *static_cast<BackwardOutputCore*>(out) = elastic3d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 71: *static_cast<ForwardOutputCore*>(out) = elastic3d::apm_forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 72: *static_cast<ForwardOutputCore*>(out) = elastic_tti_2nd2d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 73: *static_cast<BackwardOutputCore*>(out) = elastic_tti_2nd2d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 74: *static_cast<BackwardOutputCore*>(out) = elastic_tti_2nd2d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 75: *static_cast<BackwardOutputCore*>(out) = elastic_tti_2nd2d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 76: *static_cast<ForwardOutputCore*>(out) = elastic_tti_sg2d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 77: *static_cast<BackwardOutputCore*>(out) = elastic_tti_sg2d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 78: *static_cast<BackwardOutputCore*>(out) = elastic_tti_sg2d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 79: *static_cast<BackwardOutputCore*>(out) = elastic_tti_sg2d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 80: *static_cast<ForwardOutputCore*>(out) = elastic_tti_sg3d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 81: *static_cast<BackwardOutputCore*>(out) = elastic_tti_sg3d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 82: *static_cast<BackwardOutputCore*>(out) = elastic_tti_sg3d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 83: *static_cast<BackwardOutputCore*>(out) = elastic_tti_sg3d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 84: *static_cast<ForwardOutputCore*>(out) = elastic_vr2d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 85: *static_cast<BackwardOutputCore*>(out) = elastic_vr2d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 86: *static_cast<BackwardOutputCore*>(out) = elastic_vr2d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 87: *static_cast<BackwardOutputCore*>(out) = elastic_vr2d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 88: *static_cast<BackwardOutputCore*>(out) = elastic_vr2d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 89: *static_cast<ForwardOutputCore*>(out) = visco_acoustic2d::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
        case 90: *static_cast<BackwardOutputCore*>(out) = visco_acoustic2d::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 91: *static_cast<BackwardOutputCore*>(out) = visco_acoustic2d::backward_bs_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 92: *static_cast<BackwardOutputCore*>(out) = visco_acoustic2d::backward_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        case 93: *static_cast<BackwardOutputCore*>(out) = visco_acoustic2d::backward_recursive_ckpt_core(*static_cast<const BackwardInputCore*>(in)); return;
        default: SWEEP_CHECK(false, "sweep_call: no entry ", entry);
    }
}

ForwardRunnerCorePtr make_forward_runner(int entry, const ForwardInputCore& in)
{
    switch (entry) {
        case 0: return acoustic2d::forward_runner_core(in);
        case 5: return acoustic3d::forward_runner_core(in);
        case 20: return acoustic_vrz2d::forward_runner_core(in);
        case 50: return das_mu2d::forward_runner_core(in);
        case 55: return das_mu3d::forward_runner_core(in);
        case 60: return elastic2d::forward_runner_core(in);
        case 66: return elastic3d::forward_runner_core(in);
        case 76: return elastic_tti_sg2d::forward_runner_core(in);
        case 80: return elastic_tti_sg3d::forward_runner_core(in);
        case 84: return elastic_vr2d::forward_runner_core(in);
        default: SWEEP_CHECK(false, "no stepped forward runner for entry ", entry, " (", sweep_entry_name(entry) ? sweep_entry_name(entry) : "?", ")"); return nullptr;
    }
}

BackwardRunnerCorePtr make_backward_runner(int entry, const BackwardInputCore& in)
{
    switch (entry) {
        case 2: return acoustic2d::backward_bs_runner_core(in);
        case 7: return acoustic3d::backward_bs_runner_core(in);
        case 22: return acoustic_vrz2d::backward_bs_runner_core(in);
        case 52: return das_mu2d::backward_bs_runner_core(in);
        case 57: return das_mu3d::backward_bs_runner_core(in);
        case 62: return elastic2d::backward_bs_runner_core(in);
        case 67: return elastic3d::backward_bs_runner_core(in);
        case 78: return elastic_tti_sg2d::backward_bs_runner_core(in);
        case 82: return elastic_tti_sg3d::backward_bs_runner_core(in);
        case 86: return elastic_vr2d::backward_bs_runner_core(in);
        default: SWEEP_CHECK(false, "no stepped backward runner for entry ", entry, " (", sweep_entry_name(entry) ? sweep_entry_name(entry) : "?", ")"); return nullptr;
    }
}

}  // namespace

extern "C" {

int sweep_core_abi_version(void) { return SWEEP_CORE_ABI_VERSION; }
int sweep_entry_count(void) { return SWEEP_ENTRY_COUNT; }
const char* sweep_entry_name(int entry) { return (entry >= 0 && entry < SWEEP_ENTRY_COUNT) ? ENTRY_NAMES[entry] : nullptr; }
int sweep_entry_kind(int entry) { return (entry >= 0 && entry < SWEEP_ENTRY_COUNT) ? ENTRY_KINDS[entry] : -1; }

int sweep_call(int entry, const void* in, void* out, char* err, int err_cap)
{
    return guarded([&] { dispatch(entry, in, out); }, err, err_cap);
}

int sweep_forward_runner_create(int entry, const void* in, void** handle, char* err, int err_cap)
{
    return guarded([&] {
        *handle = new ForwardRunnerCorePtr(make_forward_runner(entry, *static_cast<const ForwardInputCore*>(in)));
    }, err, err_cap);
}
int sweep_forward_runner_run(void* handle, int it_begin, int it_end, int step_phase, void* out, char* err, int err_cap)
{
    return guarded([&] {
        *static_cast<ForwardOutputCore*>(out) = (*static_cast<ForwardRunnerCorePtr*>(handle))->run(it_begin, it_end, step_phase);
    }, err, err_cap);
}
int sweep_forward_runner_device_index(void* handle) { return (*static_cast<ForwardRunnerCorePtr*>(handle))->device_index(); }
void sweep_forward_runner_destroy(void* handle) { delete static_cast<ForwardRunnerCorePtr*>(handle); }

int sweep_backward_runner_create(int entry, const void* in, void** handle, char* err, int err_cap)
{
    return guarded([&] {
        *handle = new BackwardRunnerCorePtr(make_backward_runner(entry, *static_cast<const BackwardInputCore*>(in)));
    }, err, err_cap);
}
int sweep_backward_runner_run(void* handle, int bw_it_begin, int bw_it_end, int step_phase, void* out, char* err, int err_cap)
{
    return guarded([&] {
        *static_cast<BackwardOutputCore*>(out) = (*static_cast<BackwardRunnerCorePtr*>(handle))->run(bw_it_begin, bw_it_end, step_phase);
    }, err, err_cap);
}
int sweep_backward_runner_device_index(void* handle) { return (*static_cast<BackwardRunnerCorePtr*>(handle))->device_index(); }
void sweep_backward_runner_destroy(void* handle) { delete static_cast<BackwardRunnerCorePtr*>(handle); }

void* sweep_session_create(void) { return new BoundarySession(); }
int sweep_session_finish(void* session, char* err, int err_cap)
{
    return guarded([&] { static_cast<BoundarySession*>(session)->finish(); }, err, err_cap);
}
int sweep_session_used(void* session) { return static_cast<BoundarySession*>(session)->used() ? 1 : 0; }
void sweep_session_destroy(void* session) { delete static_cast<BoundarySession*>(session); }

void sweep_set_stream(void* stream) { sweep::current_stream_slot() = static_cast<cudaStream_t>(stream); }
void* sweep_get_stream(void) { return static_cast<void*>(sweep::current_stream()); }

size_t sweep_visco_fft_workspace_bytes(int64_t B, int64_t nz, int64_t nx) { return visco_acoustic2d::fft_workspace_bytes(B, nz, nx); }

}  // extern "C"
