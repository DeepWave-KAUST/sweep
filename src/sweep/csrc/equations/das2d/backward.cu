#include <torch/extension.h>

#include "das2d.h"

namespace das2d {

static BackwardOutput das2d_backward_not_implemented()
{
    TORCH_CHECK(false, "DAS 2D CUDA backward is not implemented yet. Use eager mode for gradients.");
    return BackwardOutput{};
}

BackwardOutput backward(const BackwardInput& in)
{
    (void)in;
    return das2d_backward_not_implemented();
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    (void)in;
    return das2d_backward_not_implemented();
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    (void)in;
    return das2d_backward_not_implemented();
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    (void)in;
    return das2d_backward_not_implemented();
}

}
