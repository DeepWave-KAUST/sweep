#pragma once
// core/outputs.h -- the torch-free twins of ForwardOutput / BackwardOutput /
// RTMOutput (shared/wavetypes.h).  Every buffer a driver hands back is one the
// propagator bound on the input struct (record_out, u_allt_out, last_two,
// grads_out, illum_out, adcig_out), so these carry descriptors and the torch
// side maps each one back to its tensor by pointer identity
// (cuda/common/adapt_inputs.h, tensor_of).
#include <vector>
#include "buf.h"

struct ForwardOutputCore {
    Buf wavefield;   // u_allt_out when save_all_wavefields
    Buf last_two;    // the bound last_two when boundary saving is on
    Buf record;      // record_out
};

struct RTMOutputCore {
    Buf source_illumination;
    Buf receiver_illumination;
    Buf adcig;
};

struct BackwardOutputCore {
    std::vector<Buf> checkpoints;   // never filled by the skeletons; kept for the layout
    std::vector<Buf> grads;
    Buf source_illumination;
    Buf receiver_illumination;
    Buf adcig;
};
