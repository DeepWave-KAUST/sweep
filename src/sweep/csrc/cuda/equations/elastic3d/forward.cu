#include <cuda_runtime.h>


#include "elastic3d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../launch/config.h"
#include "../../operators/staggered.cuh"
#include "driver_traits.cuh"

namespace elastic3d {

ForwardOutputCore forward_core(const ForwardInputCore& in)
{
    return eqdrv::sg_generic_forward_core<Driver>(in);
}


ForwardRunnerCorePtr forward_runner_core(const ForwardInputCore& in)
{
    return std::make_shared<eqdrv::SgForwardRunner<Driver>>(in);
}



}
