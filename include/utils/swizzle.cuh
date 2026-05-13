#ifndef UTILS_SWIZZLE_CUH
#define UTILS_SWIZZLE_CUH

// XOR-based bank-conflict avoidance for shared memory.
// Used in matmul tiled_v2, tiled_v3.
#define SWIZZLE_BANK(x) ((x) ^ ((x) >> 5))

// Complex XOR+shift bank-conflict avoidance for flash_v3 layout.
#define SWIZZLE_BANK_V3(x) ((((x >> 5) ^ (x >> 2)) << 2) + (x & 3))

// float4 pointer/reference helpers for vectorized global→smem loads.
#define FLOAT4_PTR(x) (reinterpret_cast<float4*>((x)))
#define FLOAT4_REF(x) (*reinterpret_cast<float4*>((x)))

#endif
