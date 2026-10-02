#include <cuda_runtime.h>

#include "visco_elastic2d.h"
#include "driver_traits.cuh"

namespace visco_elastic2d {

ForwardOutputCore forward_core(const ForwardInputCore& in)
{
    switch (n_sls_of(in.models)) {
        case 1:  return eqdrv::sg_generic_forward_core<DriverL<1>>(in);
        case 2:  return eqdrv::sg_generic_forward_core<DriverL<2>>(in);
        case 3:  return eqdrv::sg_generic_forward_core<DriverL<3>>(in);
        default: return eqdrv::sg_generic_forward_core<DriverL<4>>(in);
    }
}

}
