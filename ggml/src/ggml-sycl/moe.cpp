#include "moe.hpp"
#include "presets.hpp"
#include "ggml-common.h"

// Grid: [n_experts, ne01], work-group: [1, WARP_SIZE].
// Each work-group (expert_id, out_row) scans all (iid1, id) slots in the IDs
// tensor on-GPU to find the ones assigned to it. For each match it computes
// one complete dot product and writes directly to the unique dst row.
//
// Grid shape is derived entirely from src0 metadata (n_experts = ne02, ne01)
// and never changes — graph-compatible for any ne12, ne12 value.

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

// Q4_0 weight variant.
static void k_moe_gemv_q4_0(
        const void  * __restrict__ src0,
        const float * __restrict__ src1,
        const moe_gemv_params p,
        const sycl::nd_item<2> item) {

    const int32_t expert_id = (int32_t)item.get_group(0);
    const int64_t out_row   = item.get_group(1);
    const int     lane      = item.get_local_id(1);

    const block_q4_0 * wrow = (const block_q4_0 *)
        ((const char *)src0 + expert_id * p.nb02 + out_row * p.nb01);
    const int64_t n_blocks = p.ne00 / QK4_0;

    for (int64_t iid1 = 0; iid1 < p.ne12; iid1++) {
        for (int64_t id = 0; id < p.n_ids; id++) {
            if (read_expert_id(p, iid1, id) != expert_id) continue;

            const int64_t  i11 = id % p.ne11;
            const float *  act = (const float *)
                ((const char *)src1 + i11 * p.nb11 + iid1 * p.nb12);

            float partial = 0.0f;
            for (int64_t ib = lane; ib < n_blocks; ib += WARP_SIZE) {
                const sycl::half scale = wrow[ib].d;
                const uint8_t * qs = wrow[ib].qs;
                for (int j = 0; j < QK4_0 / 2; j++) {
                    const uint8_t    q    = qs[j];
                    const sycl::half w_lo = sycl::half((int)( q       & 0xF) - 8) * scale;
                    const sycl::half w_hi = sycl::half((int)((q >> 4) & 0xF) - 8) * scale;
                    partial += (float)w_lo * act[ib * QK4_0 + j];
                    partial += (float)w_hi * act[ib * QK4_0 + j + QK4_0 / 2];
                }
            }

            partial = sycl::reduce_over_group(item.get_sub_group(), partial,
                                              sycl::plus<float>{});
            if (lane == 0) {
                ((float *)((char *)p.dst + id * p.nb1 + iid1 * p.nb2))[out_row] = partial;
            }
        }
    }
}

// F32 weight variant.
static void k_moe_gemv_f32(
        const float * __restrict__ src0,
        const float * __restrict__ src1,
        const moe_gemv_params p,
        const sycl::nd_item<2> item) {

    const int32_t expert_id = (int32_t)item.get_group(0);
    const int64_t out_row   = item.get_group(1);
    const int     lane      = item.get_local_id(1);

    const float * wrow = (const float *)
        ((const char *)src0 + expert_id * p.nb02 + out_row * p.nb01);

    for (int64_t iid1 = 0; iid1 < p.ne12; iid1++) {
        for (int64_t id = 0; id < p.n_ids; id++) {
            if (read_expert_id(p, iid1, id) != expert_id) continue;

            const int64_t  i11 = id % p.ne11;
            const float *  act = (const float *)
                ((const char *)src1 + i11 * p.nb11 + iid1 * p.nb12);

            float partial = 0.0f;
            for (int64_t k = lane; k < p.ne00; k += WARP_SIZE) {
                partial += wrow[k] * act[k];
            }

            partial = sycl::reduce_over_group(item.get_sub_group(), partial,
                                              sycl::plus<float>{});
            if (lane == 0) {
                ((float *)((char *)p.dst + id * p.nb1 + iid1 * p.nb2))[out_row] = partial;
            }
        }
    }
}

static void launch_moe_gemv(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int64_t n_experts = src0->ne[2];
    const int64_t ne01      = src0->ne[1];
    const int64_t ne12      = src1->ne[2];

    const moe_gemv_params p {
        (const char *)ids->data,
        (float *)dst->data,
        src0->ne[0], ne01, src0->nb[1], src0->nb[2],
        src1->ne[1], src1->nb[1], src1->nb[2],
        dst->nb[1], dst->nb[2],
        ids->ne[0], ne12,
        ids->nb[0], ids->nb[1],
    };

    const queue_ptr stream = ctx.stream();
    // Static grid: always n_experts × ne01 work-groups regardless of ne12 or routing.
    const sycl::range<2> global(n_experts, ne01 * WARP_SIZE);
    const sycl::range<2> local(1, WARP_SIZE);

    if (src0->type == GGML_TYPE_Q4_0) {
        const void  * src0_data = src0->data;
        const float * src1_data = (const float *)src1->data;
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(global, local),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    k_moe_gemv_q4_0(src0_data, src1_data, p, item);
                });
        });
    } else {
        const float * src0_data = (const float *)src0->data;
        const float * src1_data = (const float *)src1->data;
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(global, local),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    k_moe_gemv_f32(src0_data, src1_data, p, item);
                });
        });
    }
}

void ggml_sycl_moe_gemv_q4_0(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    launch_moe_gemv(ctx, dst);
}
