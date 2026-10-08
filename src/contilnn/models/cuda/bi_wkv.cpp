/******************************************************************************
 * Copyright (c) 2025 Shanghai AI Lab.
 ******************************************************************************/
// Editor: Jialei.He

#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void check_wkv_shapes(
    const torch::Tensor& w,
    const torch::Tensor& u,
    const torch::Tensor& k,
    const torch::Tensor& v) {
    TORCH_CHECK(k.dim() == 3, "k must have shape [B,T,C]");
    TORCH_CHECK(v.sizes() == k.sizes(), "v must match k");
    const auto batch = k.size(0);
    const auto tokens = k.size(1);
    const auto channels = k.size(2);
    TORCH_CHECK(w.dim() == 1 && w.numel() == channels, "w must have C values");
    TORCH_CHECK(u.dim() == 1 && u.numel() == channels, "u must have C values");
    TORCH_CHECK(tokens >= 64, "WKV requires T >= 64");
    TORCH_CHECK(channels >= 8, "WKV requires C >= 8");
    TORCH_CHECK((batch * channels) % 8 == 0, "WKV requires B*C divisible by 8");
    TORCH_CHECK(w.scalar_type() == k.scalar_type(), "w and k dtypes must match");
    TORCH_CHECK(u.scalar_type() == k.scalar_type(), "u and k dtypes must match");
    TORCH_CHECK(v.scalar_type() == k.scalar_type(), "v and k dtypes must match");
    TORCH_CHECK(w.device() == k.device(), "w and k devices must match");
    TORCH_CHECK(u.device() == k.device(), "u and k devices must match");
    TORCH_CHECK(v.device() == k.device(), "v and k devices must match");
}

torch::Tensor bi_wkv_cuda_forward(
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor k,
    torch::Tensor v);

std::vector<torch::Tensor> bi_wkv_cuda_backward(
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gy);

torch::Tensor bi_wkv_forward(
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor k,
    torch::Tensor v) {
    CHECK_INPUT(w);
    CHECK_INPUT(u);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    check_wkv_shapes(w, u, k, v);
    return bi_wkv_cuda_forward(w, u, k, v);
}

std::vector<torch::Tensor> bi_wkv_backward(
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gy) {
    CHECK_INPUT(w);
    CHECK_INPUT(u);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(gy);
    check_wkv_shapes(w, u, k, v);
    TORCH_CHECK(gy.sizes() == k.sizes(), "gy must match k");
    TORCH_CHECK(gy.scalar_type() == k.scalar_type(), "gy and k dtypes must match");
    TORCH_CHECK(gy.device() == k.device(), "gy and k devices must match");
    return bi_wkv_cuda_backward(w, u, k, v, gy);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bi_wkv_forward", &bi_wkv_forward, "Bi-WKV Forward(CUDA)");
    m.def("bi_wkv_backward", &bi_wkv_backward, "Bi-WKV Backward(CUDA)");
}

TORCH_LIBRARY(bi_wkv, m) {
    m.def("bi_wkv_forward", bi_wkv_forward);
    m.def("bi_wkv_backward", bi_wkv_backward);
}
