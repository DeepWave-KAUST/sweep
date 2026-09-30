#include <cuda_runtime.h>

#include "visco_elastic2d.h"
#include "driver_traits.cuh"

namespace visco_elastic2d {

// The traits are a template on the mechanism count (the wavefield and
// checkpoint counts depend on it); n_sls_of refuses anything outside 1..4.
#define VISCO_ELASTIC2D_BY_SLS(core, in)                                     \
    switch (n_sls_of((in).models)) {                                        \
        case 1:  return eqdrv::core<DriverL<1>>(in);                        \
        case 2:  return eqdrv::core<DriverL<2>>(in);                        \
        case 3:  return eqdrv::core<DriverL<3>>(in);                        \
        default: return eqdrv::core<DriverL<4>>(in);                        \
    }

BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    VISCO_ELASTIC2D_BY_SLS(sg_generic_backward_core, in)
}

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in)
{
    VISCO_ELASTIC2D_BY_SLS(sg_generic_backward_ckpt_core, in)
}

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in)
{
    VISCO_ELASTIC2D_BY_SLS(sg_generic_backward_recursive_ckpt_core, in)
}

#undef VISCO_ELASTIC2D_BY_SLS

BackwardOutputCore backward_bs_core(const BackwardInputCore&)
{
    SWEEP_CHECK(false,
                "ViscoElastic does not support boundary saving (the attenuation is "
                "dissipative: no stable reverse reconstruction); use "
                "memory=Full() or memory=Ckpt(...) (sweep.propagator.options)");
    return BackwardOutputCore{};
}

}
