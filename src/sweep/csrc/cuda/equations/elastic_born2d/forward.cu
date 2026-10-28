#include <cuda_runtime.h>

#include "elastic_born2d.h"
#include "driver_traits.cuh"

namespace elastic_born2d {

ForwardOutputCore forward_core(const ForwardInputCore& in)
{
    return eqdrv::sg_generic_forward_core<Driver>(in);
}

}
