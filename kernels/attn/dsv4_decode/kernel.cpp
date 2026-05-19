#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

namespace kittens::base_types {
template<> struct packing<uint8_t> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = uint8_t;
    using packed_type = uint8_t;
    static __device__ inline constexpr uint8_t pack(const uint8_t &i) { return i; }
};
}

constexpr int H = 64;
constexpr int D = 512;
constexpr int NOPE_D = 448;
constexpr int ROPE_D = 64;
constexpr int N = 128;
constexpr int CACHE_TOKEN_BYTES = 576;

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

template<int _D> struct packed_decode_globals {
    using q_gl = gl<bf16, -1, -1, -1, -1>;       // [B, H, 1, D]
    using cache_gl = gl<uint8_t, -1, -1, -1, -1>; // [blocks, block, 584]
    using idx_gl = gl<int, -1, -1, -1, -1>;      // [B, 1, N, 1]
    using sink_gl = gl<float, -1, -1, -1, -1>;   // [H]
    using o_gl = gl<bf16, -1, -1, -1, -1>;       // [B, H, 1, D]

    q_gl q;
    cache_gl cache;
    idx_gl indices;
    sink_gl attn_sink;
    o_gl o;
    float scale;
    int has_attn_sink;

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

__device__ static inline float fp8_e4m3fnuz_to_float(uint8_t x) {
    if (x == 0) {
        return 0.0f;
    }
    if (x == 0x80) {
        return 0.0f;
    }
    const int sign = (x & 0x80) ? -1 : 1;
    const int exp = (x >> 3) & 0x0f;
    const int mant = x & 0x07;
    const int bias = 8;
    float val;
    if (exp == 0) {
        val = ldexpf(static_cast<float>(mant), 1 - bias - 3);
    } else {
        val = ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - bias);
    }
    return sign < 0 ? -val : val;
}

__device__ static inline float load_packed_cache_value(
    const uint8_t *cache,
    int block_size,
    int block_stride,
    int slot,
    int dim) {
    const int block_idx = slot / block_size;
    const int pos = slot - block_idx * block_size;
    const int block_base = block_idx * block_stride;
    const int token_base = block_base + pos * CACHE_TOKEN_BYTES;
    if (dim < NOPE_D) {
        const uint8_t encoded = cache[token_base + dim];
        const uint8_t encoded_scale =
            cache[block_base + block_size * CACHE_TOKEN_BYTES + pos * 8 + dim / 64];
        return fp8_e4m3fnuz_to_float(encoded) * exp2f(static_cast<float>(encoded_scale) - 127.0f);
    }

    const int rope_dim = dim - NOPE_D;
    const int byte_offset = token_base + NOPE_D + rope_dim * 2;
    const uint16_t raw = static_cast<uint16_t>(cache[byte_offset])
        | (static_cast<uint16_t>(cache[byte_offset + 1]) << 8);
    const bf16 value = std::bit_cast<bf16>(raw);
    return base_types::convertor<float, bf16>::convert(value);
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
__global__ __launch_bounds__(NUM_THREADS, 4)
void dsv4_decode_packed_swa_kernel(const packed_decode_globals<_D> g) {
    const int pid = blockIdx.x;
    const int batch = pid / H;
    const int head = pid - batch * H;
    const int block_size = g.cache.rows();
    const int block_stride = g.cache.template stride<1>();

    rv<float, _D> q_reg, k_reg, prod, out;
    zero_vec(out);
    load(q_reg, g.q, {batch, head, 0, 0});

    float m = -INFINITY;
    float l = 0.0f;

    #pragma unroll
    for (int t = 0; t < N; ++t) {
        const int slot = g.indices[{batch, 0, t, 0}];

        #pragma unroll
        for (int i = 0; i < decltype(k_reg)::outer_dim; ++i) {
            #pragma unroll
            for (int j = 0; j < decltype(k_reg)::inner_dim; ++j) {
                const int dim = i * 64 + laneid();
                const float kv = load_packed_cache_value(g.cache.raw_ptr, block_size, block_stride, slot, dim);
                k_reg[i][j] = kv;
            }
        }
        mul_vec(prod, q_reg, k_reg);
        float score;
        sum(score, prod);
        score *= g.scale;

        const float new_m = fmaxf(m, score);
        const float alpha = __expf(m - new_m);
        const float beta = __expf(score - new_m);
        update_out(out, k_reg, alpha, beta);

        l = l * alpha + beta;
        m = new_m;
    }

    if (g.has_attn_sink) {
        const float sink = g.attn_sink[{0, 0, 0, head}];
        const float new_m = fmaxf(m, sink);
        const float alpha = __expf(m - new_m);
        l = l * alpha + __expf(sink - new_m);
        m = new_m;
        #pragma unroll
        for (int i = 0; i < decltype(out)::outer_dim; ++i) {
            #pragma unroll
            for (int j = 0; j < decltype(out)::inner_dim; ++j) {
                out[i][j] *= alpha;
            }
        }
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

template<int _D>
void dispatch_packed_decode(packed_decode_globals<_D> g) {
    dsv4_decode_packed_swa_kernel<_D><<<g.grid(), g.block(), 0>>>(g);
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
    py::bind_function<dispatch_packed_decode<D>>(m, "dispatch_packed_decode",
        &packed_decode_globals<D>::q,
        &packed_decode_globals<D>::cache,
        &packed_decode_globals<D>::indices,
        &packed_decode_globals<D>::attn_sink,
        &packed_decode_globals<D>::o,
        &packed_decode_globals<D>::scale,
        &packed_decode_globals<D>::has_attn_sink
    );
}
