#include "moe.hpp"
#include "presets.hpp"
#include "ggml-common.h"
#include "vecdotq.hpp"
#include "quantize.hpp"

// Grid: [n_ids*ne12, ne01], work-group: [1, WARP_SIZE].
// Each work-group maps to one (id-slot, out_row) pair. It reads the expert ID
// for its slot directly from the IDs tensor, then computes the dot product for
// that (expert, output-row) pair and writes to the unique dst row.
// No scanning — dispatches exactly top_k*n_tokens*ne01 warps.

struct moe_gemv_params {
    const char * ids;      // raw byte pointer, indexed via nb0/nb1 strides
    float      * dst;
    int64_t ne00, ne01, nb01, nb02;
    int64_t ne11, nb11, nb12;
    int64_t nb1, nb2;
    int64_t n_ids;          // ids->ne[0] = top_k
    int64_t ne12;           // src1->ne[2], number of sequences
    int64_t ids_nb0, ids_nb1;
};

static __dpct_inline__ int32_t read_expert_id(const moe_gemv_params & p,
                                               int64_t iid1, int64_t id) {
    return *(const int32_t *)(p.ids + iid1 * p.ids_nb1 + id * p.ids_nb0);
}

// Q4_0 x Q8_1 variant — MMVQ inner loop with multi-row output per work-group.
// Each work-group processes ROWS_PER_WG consecutive output rows, loading each
// expert weight block once and reusing it across all rows. Reduces warp count
// and improves weight reuse.
template <int ROWS_PER_WG>
static void k_moe_gemv_q4_0_q8_1(
        const void  * __restrict__ src0,
        const void  * __restrict__ src1_q8,
        const moe_gemv_params p,
        const sycl::nd_item<2> item) {

    const int64_t slot      = item.get_group(0);  // 0..n_ids*ne12
    const int64_t base_row  = item.get_group(1) * ROWS_PER_WG;
    const int     lane      = item.get_local_id(1);

    const int64_t iid1      = slot / p.n_ids;
    const int64_t id        = slot % p.n_ids;
    const int32_t expert_id = read_expert_id(p, iid1, id);

    const block_q4_0 * x_expert = (const block_q4_0 *)
        ((const char *)src0 + expert_id * p.nb02);
    const block_q8_1 * y = (const block_q8_1 *)
        ((const char *)src1_q8 + (id % p.ne11) * p.nb11 + iid1 * p.nb12);

    constexpr int qi_per_vdr      = QI4_0 / VDR_Q4_0_Q8_1_MMVQ;          // 2
    constexpr int blocks_per_warp = (VDR_Q4_0_Q8_1_MMVQ * WARP_SIZE + QI4_0 - 1) / QI4_0;  // 16
    const int iqs = VDR_Q4_0_Q8_1_MMVQ * (lane % qi_per_vdr);

    const int blocks_per_row = p.ne00 / QK4_0;
    float tmp[ROWS_PER_WG];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WG; ++r) {
        tmp[r] = 0.0f;
    }

    // Stream through expert weight matrix once; accumulate all ROWS_PER_WG outputs.
    for (int i = lane / qi_per_vdr; i < blocks_per_row; i += blocks_per_warp) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WG; ++r) {
            const block_q4_0 * x_row = (const block_q4_0 *)
                ((const char *)x_expert + (base_row + r) * p.nb01);
            tmp[r] += vec_dot_q4_0_q8_1(&x_row[i], &y[i], iqs);
        }
    }

    // Reduce and write each output row.
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WG; ++r) {
        tmp[r] = sycl::reduce_over_group(item.get_sub_group(), tmp[r], sycl::plus<float>{});
        if (lane == 0) {
            ((float *)((char *)p.dst + id * p.nb1 + iid1 * p.nb2))[base_row + r] = tmp[r];
        }
    }
}

// F32 weight variant with multi-row output per work-group.
template <int ROWS_PER_WG>
static void k_moe_gemv_f32(
        const float * __restrict__ src0,
        const float * __restrict__ src1,
        const moe_gemv_params p,
        const sycl::nd_item<2> item) {

    const int64_t slot      = item.get_group(0);  // 0..n_ids*ne12
    const int64_t base_row  = item.get_group(1) * ROWS_PER_WG;
    const int     lane      = item.get_local_id(1);

    const int64_t iid1      = slot / p.n_ids;
    const int64_t id        = slot % p.n_ids;
    const int32_t expert_id = read_expert_id(p, iid1, id);

    const float * expert_base = (const float *)
        ((const char *)src0 + expert_id * p.nb02);

    const int64_t  i11 = id % p.ne11;
    const float *  act = (const float *)
        ((const char *)src1 + i11 * p.nb11 + iid1 * p.nb12);

    float partial[ROWS_PER_WG];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WG; ++r) {
        partial[r] = 0.0f;
    }

    for (int64_t k = lane; k < p.ne00; k += WARP_SIZE) {
        const float act_val = act[k];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WG; ++r) {
            const float * wrow = (const float *)
                ((const char *)expert_base + (base_row + r) * p.nb01);
            partial[r] += wrow[k] * act_val;
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WG; ++r) {
        partial[r] = sycl::reduce_over_group(item.get_sub_group(), partial[r],
                                             sycl::plus<float>{});
        if (lane == 0) {
            ((float *)((char *)p.dst + id * p.nb1 + iid1 * p.nb2))[base_row + r] = partial[r];
        }
    }
}

static void launch_moe_gemv(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    const queue_ptr stream = ctx.stream();

    // Pre-quantize F32 activations to Q8_1 once — all GEMV work-groups share
    // this buffer via L2 cache, matching MMVQ's quantize-then-GEMV structure.
    const int64_t src1_q8_size = ne11 * ne12 * ggml_row_size(GGML_TYPE_Q8_1, ne10);

    // Use persistent buffer if pre-allocated before graph recording (graph-compatible path).
    // Fall back to pool alloc for non-graph execution.
    ggml_sycl_pool_alloc<char> src1_q8_pool_alloc;
    void * src1_q8;
    if (ctx.moe_q8_buffer != nullptr && ctx.moe_q8_buffer_size >= (size_t)src1_q8_size) {
        src1_q8 = ctx.moe_q8_buffer;
    } else {
        src1_q8_pool_alloc.alloc(ctx.pool(), src1_q8_size);
        src1_q8 = src1_q8_pool_alloc.get();
    }

    quantize_row_q8_1_sycl<quantize_q8_1>(
        (const float *)src1->data,
        src1_q8,
        ne10,
        ne11 * ne12,
        ne10,
        stream
    );

    const int64_t nb11_q8 = ggml_row_size(GGML_TYPE_Q8_1, ne10);
    const int64_t nb12_q8 = ne11 * nb11_q8;

    const moe_gemv_params p {
        (const char *)ids->data,
        (float *)dst->data,
        src0->ne[0], ne01, src0->nb[1], src0->nb[2],
        ne11, nb11_q8, nb12_q8,
        dst->nb[1], dst->nb[2],
        ids->ne[0], ne12,
        ids->nb[0], ids->nb[1],
    };

    // Multi-row-per-work-group: reduces warp count and improves weight reuse.
    // ROWS_PER_WG output rows computed per work-group; grid shrinks by that factor.
    constexpr int ROWS_PER_WG = 16;
    GGML_ASSERT(ne01 % ROWS_PER_WG == 0);
    const sycl::range<2> global(p.n_ids * ne12, (ne01 / ROWS_PER_WG) * WARP_SIZE);
    const sycl::range<2> local(1, WARP_SIZE);

    if (src0->type == GGML_TYPE_Q4_0) {
        const void * src0_data = src0->data;
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(global, local),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    k_moe_gemv_q4_0_q8_1<ROWS_PER_WG>(src0_data, src1_q8, p, item);
                });
        });
    } else {
        const float * src0_data = (const float *)src0->data;
        const float * src1_data = (const float *)src1->data;
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(global, local),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    k_moe_gemv_f32<ROWS_PER_WG>(src0_data, src1_data, p, item);
                });
        });
    }
}

void ggml_sycl_moe_gemv_q4_0(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    launch_moe_gemv(ctx, dst);
}
