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

// Q4_0 weight variant.
static void k_moe_gemv_q4_0(
        const void  * __restrict__ src0,
        const float * __restrict__ src1,
        const moe_gemv_params p,
        const sycl::nd_item<2> item) {

    const int64_t slot      = item.get_group(0);  // 0..n_ids*ne12
    const int64_t out_row   = item.get_group(1);
    const int     lane      = item.get_local_id(1);

    const int64_t iid1      = slot / p.n_ids;
    const int64_t id        = slot % p.n_ids;
    const int32_t expert_id = read_expert_id(p, iid1, id);

    const block_q4_0 * wrow = (const block_q4_0 *)
        ((const char *)src0 + expert_id * p.nb02 + out_row * p.nb01);
    const int64_t n_blocks = p.ne00 / QK4_0;

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

// Q4_0 x Q8_1 variant — inner loop cloned from mul_mat_vec_q (mmvq.cpp)
// with pre-computed row pointers instead of row*blocks_per_row offsets.
// Half the lanes use iqs=0, half use iqs=2, covering both halves of each
// Q4_0 block simultaneously across the warp (same as MMVQ).
static void k_moe_gemv_q4_0_q8_1(
        const void  * __restrict__ src0,
        const void  * __restrict__ src1_q8,
        const moe_gemv_params p,
        const sycl::nd_item<2> item) {

    const int64_t slot      = item.get_group(0);  // 0..n_ids*ne12
    const int64_t out_row   = item.get_group(1);
    const int     lane      = item.get_local_id(1);

    const int64_t iid1      = slot / p.n_ids;
    const int64_t id        = slot % p.n_ids;
    const int32_t expert_id = read_expert_id(p, iid1, id);

    // Pre-computed row pointers — replaces ibx = row*blocks_per_row + i.
    const block_q4_0 * x = (const block_q4_0 *)
        ((const char *)src0 + expert_id * p.nb02 + out_row * p.nb01);
    const block_q8_1 * y = (const block_q8_1 *)
        ((const char *)src1_q8 + (id % p.ne11) * p.nb11 + iid1 * p.nb12);

    // MMVQ constants for Q4_0 x Q8_1.
    constexpr int qi_per_vdr    = QI4_0 / VDR_Q4_0_Q8_1_MMVQ;          // 2
    constexpr int blocks_per_warp = (VDR_Q4_0_Q8_1_MMVQ * WARP_SIZE + QI4_0 - 1) / QI4_0;  // 16

    // Each lane's iqs is fixed: even lanes → 0, odd lanes → 2.
    // (The elem loop in mul_mat_vec_q runs once since qi/vdr=2 < WARP_SIZE.)
    const int iqs = VDR_Q4_0_Q8_1_MMVQ * (lane % qi_per_vdr);

    const int blocks_per_row = p.ne00 / QK4_0;
    float tmp = 0.0f;

    for (int i = lane / qi_per_vdr; i < blocks_per_row; i += blocks_per_warp) {
        tmp += vec_dot_q4_0_q8_1(&x[i], &y[i], iqs);  // iby = i*(QK4_0/QK8_1) = i
    }

    tmp = sycl::reduce_over_group(item.get_sub_group(), tmp, sycl::plus<float>{});
    if (lane == 0) {
        ((float *)((char *)p.dst + id * p.nb1 + iid1 * p.nb2))[out_row] = tmp;
    }
}

// F32 weight variant.
static void k_moe_gemv_f32(
        const float * __restrict__ src0,
        const float * __restrict__ src1,
        const moe_gemv_params p,
        const sycl::nd_item<2> item) {

    const int64_t slot      = item.get_group(0);  // 0..n_ids*ne12
    const int64_t out_row   = item.get_group(1);
    const int     lane      = item.get_local_id(1);

    const int64_t iid1      = slot / p.n_ids;
    const int64_t id        = slot % p.n_ids;
    const int32_t expert_id = read_expert_id(p, iid1, id);

    const float * wrow = (const float *)
        ((const char *)src0 + expert_id * p.nb02 + out_row * p.nb01);

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

static void launch_moe_gemv(ggml_backend_sycl_context & ctx, const ggml_tensor * dst, bool use_q8_1) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int64_t n_experts = src0->ne[2];
    const int64_t ne01      = src0->ne[1];
    const int64_t ne12      = src1->ne[2];
    const int64_t ne10      = src1->ne[0];
    const int64_t ne11      = src1->ne[1];

    const queue_ptr stream = ctx.stream();

    // Allocate Q8_1 buffer if needed
    const int64_t src1_q8_size = (use_q8_1 && src0->type == GGML_TYPE_Q4_0)
        ? ne11 * ne12 * ggml_row_size(GGML_TYPE_Q8_1, ne10)
        : 0;
    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(), src1_q8_size);

    void * src1_q8 = nullptr;
    int64_t nb11_q8 = src1->nb[1];
    int64_t nb12_q8 = src1->nb[2];

    if (use_q8_1 && src0->type == GGML_TYPE_Q4_0) {
        // Quantize src1 (F32) to Q8_1
        src1_q8 = src1_q8_alloc.get();

        quantize_row_q8_1_sycl<quantize_q8_1>(
            (const float *)src1->data,
            src1_q8,
            ne10,           // kx: row width
            ne11 * ne12,    // ky: total number of rows
            ne10,           // kx_padded
            stream
        );

        // Q8_1 strides
        nb11_q8 = ggml_row_size(GGML_TYPE_Q8_1, ne10);
        nb12_q8 = ne11 * nb11_q8;
    }

    const moe_gemv_params p {
        (const char *)ids->data,
        (float *)dst->data,
        src0->ne[0], ne01, src0->nb[1], src0->nb[2],
        ne11, nb11_q8, nb12_q8,
        dst->nb[1], dst->nb[2],
        ids->ne[0], ne12,
        ids->nb[0], ids->nb[1],
    };

    // Grid: [n_ids*ne12, ne01] — one work-group per (id-slot, output-row).
    // Each warp reads its expert ID directly; no scanning, no idle warps.
    const sycl::range<2> global(p.n_ids * ne12, ne01 * WARP_SIZE);
    const sycl::range<2> local(1, WARP_SIZE);

    if (use_q8_1 && src0->type == GGML_TYPE_Q4_0) {
        const void * src0_data = src0->data;
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(global, local),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    k_moe_gemv_q4_0_q8_1(src0_data, src1_q8, p, item);
                });
        });
    } else if (src0->type == GGML_TYPE_Q4_0) {
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
    // Use Q8_1 quantization for optimal performance (matches MMVQ)
    launch_moe_gemv(ctx, dst, /*use_q8_1=*/true);
}
