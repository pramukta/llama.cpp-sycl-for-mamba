#pragma once

#include "common.hpp"

// GPU-resident fused MoE GEMV: replaces the CPU sync + per-expert MMVQ loop
// in ggml_sycl_mul_mat_id for the ne12==1 case with Q4_0 or F32 weights.
void ggml_sycl_moe_gemv_q4_0(ggml_backend_sycl_context & ctx, const ggml_tensor * dst);
