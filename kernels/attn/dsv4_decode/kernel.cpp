#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

constexpr int H = 64;
constexpr int D = 512;
constexpr int N = 128;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

template<int _D> struct decode_globals {
    using q_gl = gl<bf16, -1, -1, -1, -1>;   // [B, H, 1, D]
    using k_gl = gl<bf16, -1, -1, -1, -1>;   // [B, 1, N, D]
    using v_gl = gl<bf16, -1, -1, -1, -1>;   // [B, 1, N, D]
    using o_gl = gl<bf16, -1, -1, -1, -1>;   // [B, H, 1, D]
    using idx_gl = gl<int, -1, -1, -1, -1>;  // [B, 1, N, 1]

    q_gl q;
    k_gl k;
    v_gl v;
    idx_gl indices;
    o_gl o;
    float scale;

    dim3 grid() { return dim3(q.batch() * q.depth()); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<ducks::rv::all RV>
__device__ static inline void zero_vec(RV &x) {
    #pragma unroll
    for (int i = 0; i < RV::outer_dim; ++i) {
        #pragma unroll
        for (int j = 0; j < RV::inner_dim; ++j) {
            x[i][j] = base_types::constants<typename RV::dtype>::zero();
        }
    }
}

template<ducks::rv::all RV>
__device__ static inline void copy_vec(RV &dst, const RV &src) {
    #pragma unroll
    for (int i = 0; i < RV::outer_dim; ++i) {
        #pragma unroll
        for (int j = 0; j < RV::inner_dim; ++j) {
            dst[i][j] = src[i][j];
        }
    }
}

template<ducks::rv::all RV>
__device__ static inline void mul_vec(RV &dst, const RV &a, const RV &b) {
    #pragma unroll
    for (int i = 0; i < RV::outer_dim; ++i) {
        #pragma unroll
        for (int j = 0; j < RV::inner_dim; ++j) {
            dst[i][j] = a[i][j] * b[i][j];
        }
    }
}

template<ducks::rv::all RV>
__device__ static inline void update_out(RV &out, const RV &value, float alpha, float beta) {
    #pragma unroll
    for (int i = 0; i < RV::outer_dim; ++i) {
        #pragma unroll
        for (int j = 0; j < RV::inner_dim; ++j) {
            out[i][j] = out[i][j] * alpha + value[i][j] * beta;
        }
    }
}

template<int _D>
__global__ __launch_bounds__(NUM_THREADS, 4)
void dsv4_decode_dense_swa_kernel(const decode_globals<_D> g) {
    const int pid = blockIdx.x;
    const int batch = pid / H;
    const int head = pid - batch * H;

    rv<float, _D> q_reg, k_reg, v_reg, prod, out;
    zero_vec(out);

    load(q_reg, g.q, {batch, head, 0, 0});

    float m = -INFINITY;
    float l = 0.0f;

    #pragma unroll
    for (int t = 0; t < N; ++t) {
        const int token_idx = g.indices[{batch, 0, t, 0}];
        load(k_reg, g.k, {batch, 0, token_idx, 0});
        mul_vec(prod, q_reg, k_reg);
        float score;
        sum(score, prod);
        score *= g.scale;

        const float new_m = fmaxf(m, score);
        const float alpha = __expf(m - new_m);
        const float beta = __expf(score - new_m);

        load(v_reg, g.v, {batch, 0, token_idx, 0});
        update_out(out, v_reg, alpha, beta);

        l = l * alpha + beta;
        m = new_m;
    }

    const float inv_l = 1.0f / l;
    #pragma unroll
    for (int i = 0; i < decltype(out)::outer_dim; ++i) {
        #pragma unroll
        for (int j = 0; j < decltype(out)::inner_dim; ++j) {
            out[i][j] *= inv_l;
        }
    }

    store(g.o, out, {batch, head, 0, 0});
}

template<int _D>
void dispatch_decode(decode_globals<_D> g) {
    dsv4_decode_dense_swa_kernel<_D><<<g.grid(), g.block(), 0>>>(g);
}

PYBIND11_MODULE(dsv4_decode_kernel, m) {
    m.doc() = "DeepSeek V4 dense-SWA decode prototype";
    py::bind_function<dispatch_decode<D>>(m, "dispatch_decode",
        &decode_globals<D>::q,
        &decode_globals<D>::k,
        &decode_globals<D>::v,
        &decode_globals<D>::indices,
        &decode_globals<D>::o,
        &decode_globals<D>::scale
    );
}
