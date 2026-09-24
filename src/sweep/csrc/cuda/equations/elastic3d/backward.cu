#include <torch/extension.h>

#include <algorithm>

#include "kernels.cuh"
#include "elastic3d.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "../../operators/staggered.cuh"
#include "driver_traits.cuh"

namespace elastic3d {

BackwardOutput backward(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward<Driver>(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_bs<Driver>(in);
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_ckpt<Driver>(in);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_recursive_ckpt<Driver>(in);
}

BackwardRunnerPtr backward_bs_runner(const BackwardInput& in)
{
    return std::make_shared<eqdrv::SgBackwardBsRunner<Driver>>(in);
}

}
