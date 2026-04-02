#include "moe.hpp"
#include "presets.hpp"
#include "ggml-common.h"
#include "vecdotq.hpp"
#include "quantize.hpp"

// Fused MoE implementation with GPU-resident expert routing and Q8_1 quantized activations.
//
// Decode (ne12==1):
// 1. Quantize activations to Q8_1 (matches MMVQ optimization)
// 2. Direct-dispatch GEMV - grid [n_ids, ne01], one work-group per expert call
// 3. Use vec_dot_q4_0_q8_1 for optimal performance

struct moe_gemv_params {
    const char * ids;
    const void * src0;      // Q4_0 weights
    const void * src1_q8;   // Q8_1 quantized activations
    float      * dst;
    int64_t ne00, ne01, nb01, nb02;
    int64_t ne11, nb11, nb12;
    int64_t nb1, nb2;
    int64_t n_ids;
    int64_t ids_nb0, ids_nb1;
};

static __dpct_inline__ int32_t read_expert_id(const moe_gemv_params & p,
                                               int64_t iid1, int64_t id) {
    return *(const int32_t *)(p.ids + iid1 * p.ids_nb1 + id * p.ids_nb0);
}

// Q4_0 x Q8_1 decode kernel - matches MMVQ performance.
static void k_moe_gemv_q4_0_q8_1(
        const moe_gemv_params p,
        const sycl::nd_item<2> item) {

    const int64_t call_idx = item.get_group(0);
    const int64_t out_row  = item.get_group(1);
    const int     lane     = item.get_local_id(1);

    // Direct dispatch: each work-group handles one expert call.
    const int64_t id   = call_idx % p.n_ids;
    const int64_t iid1 = call_idx / p.n_ids;

    const int32_t expert_id = read_expert_id(p, iid1, id);

    const block_q4_0 * wrow = (const block_q4_0 *)
        ((const char *)p.src0 + expert_id * p.nb02 + out_row * p.nb01);
    const int64_t n_blocks = p.ne00 / QK4_0;

    const int64_t  i11 = id % p.ne11;
    const block_q8_1 *  act_q8 = (const block_q8_1 *)
        ((const char *)p.src1_q8 + i11 * p.nb11 + iid1 * p.nb12);

    // Use vec_dot_q4_0_q8_1 like MMVQ does.
    float sum = 0.0f;
    for (int64_t ib = lane; ib < n_blocks; ib += WARP_SIZE) {
        sum += vec_dot_q4_0_q8_1(&wrow[ib], &act_q8[ib], 0);
    }

    sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<float>{});
    if (lane == 0) {
        ((float *)((char *)p.dst + id * p.nb1 + iid1 * p.nb2))[out_row] = sum;
    }
}

void ggml_sycl_moe_gemv_q4_0(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int64_t ne01  = src0->ne[1];
    const int64_t ne12  = src1->ne[2];
    const int64_t n_ids = ids->ne[0];

    // Only handle decode (ne12==1) for now.
    GGML_ASSERT(ne12 == 1 && "Fused MoE kernel currently decode-only");
    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const queue_ptr stream = ctx.stream();

    // Step 1: Quantize activations to Q8_1 (like MMVQ does).
    // Allocate temp buffer for Q8_1 quantized activations.
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t src1_q8_size = ne11 * ne12 * ggml_row_size(GGML_TYPE_Q8_1, ne10);

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(), src1_q8_size);
    void * src1_q8 = src1_q8_alloc.get();

    // Quantize src1 (F32) to Q8_1.
    quantize_row_q8_1_sycl<quantize_q8_1>(
        (const float *)src1->data,
        src1_q8,
        ne10,           // kx: row width
        ne11 * ne12,    // ky: number of rows
        ne10,           // kx_padded
        stream
    );

    // Step 2: Dispatch fused MoE GEMV kernel with Q8_1 activations.
    const moe_gemv_params p {
        (const char *)ids->data,
        src0->data,
        src1_q8,
        (float *)dst->data,
        src0->ne[0], ne01, src0->nb[1], src0->nb[2],
        src1->ne[1], ggml_row_size(GGML_TYPE_Q8_1, ne10),
        ne12 * ggml_row_size(GGML_TYPE_Q8_1, ne10),
        dst->nb[1], dst->nb[2],
        n_ids,
        ids->nb[0], ids->nb[1],
    };

    // Grid: [n_ids, ne01] - one work-group per expert call (decode only).
    const sycl::range<2> global(n_ids, ne01 * WARP_SIZE);
    const sycl::range<2> local(1, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                k_moe_gemv_q4_0_q8_1(p, item);
            });
    });
}
