#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
namespace at {
namespace native {
std::vector<torch::Tensor> grid_sample2d_cuda_grad2(
    const torch::Tensor &grad2_grad_input,
    const torch::Tensor &grad2_grad_grid,
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    bool padding_mode,
    bool align_corners);

std::vector<torch::Tensor> grid_sample3d_cuda_grad2(
    const torch::Tensor &grad2_grad_input,
    const torch::Tensor &grad2_grad_grid,
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    bool padding_mode,
    bool align_corners);
} // namespace native
} // namespace at

// Wrapper functions
std::vector<torch::Tensor> grid_sample2d_grad2(
    const torch::Tensor &grad2_grad_input,
    const torch::Tensor &grad2_grad_grid,
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    bool padding_mode,
    bool align_corners) {
    return at::native::grid_sample2d_cuda_grad2(
        grad2_grad_input, grad2_grad_grid, grad_output, input, grid, padding_mode, align_corners);
}

std::vector<torch::Tensor> grid_sample3d_grad2(
    const torch::Tensor &grad2_grad_input,
    const torch::Tensor &grad2_grad_grid,
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    bool padding_mode,
    bool align_corners) {
    return at::native::grid_sample3d_cuda_grad2(
        grad2_grad_input, grad2_grad_grid, grad_output, input, grid, padding_mode, align_corners);
}

// Register the operators using TORCH_LIBRARY
TORCH_LIBRARY(gridsample_grad2, m) {
    m.def("grad2_2d(Tensor grad2_grad_input, Tensor grad2_grad_grid, Tensor grad_output, Tensor input, Tensor grid, bool padding_mode, bool align_corners) -> Tensor[]",
          grid_sample2d_grad2);
    m.def("grad2_3d(Tensor grad2_grad_input, Tensor grad2_grad_grid, Tensor grad_output, Tensor input, Tensor grid, bool padding_mode, bool align_corners) -> Tensor[]",
          grid_sample3d_grad2);
}
