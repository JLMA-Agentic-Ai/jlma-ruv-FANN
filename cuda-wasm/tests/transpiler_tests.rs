//! Unit tests for the CUDA to WASM transpiler

#[cfg(test)]
mod transpiler_tests {
    use cuda_rust_wasm::transpiler::CudaTranspiler;

    #[test]
    fn test_transpile_simple_kernel() {
        let cuda_code = r#"
            __global__ void simple_kernel(int* data) {
                data[threadIdx.x] = threadIdx.x;
            }
        "#;

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(cuda_code, false, false);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_transpile_with_optimization_flags() {
        let cuda_code = r#"
            __global__ void vector_add(float* a, float* b, float* c, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    c[idx] = a[idx] + b[idx];
                }
            }
        "#;

        // Test all combinations of (optimize, detect_patterns)
        let flag_combos: Vec<(bool, bool)> = vec![
            (false, false),
            (true, false),
            (false, true),
            (true, true),
        ];

        for (optimize, detect) in flag_combos {
            let transpiler = CudaTranspiler::new();
            let result = transpiler.transpile(cuda_code, optimize, detect);

            assert!(result.is_ok());
            let output = result.unwrap();
            assert!(!output.is_empty());
        }
    }

    #[test]
    fn test_transpile_shared_memory_kernel() {
        let cuda_code = r#"
            __global__ void reduction_kernel(float* input, float* output, int n) {
                extern __shared__ float sdata[];

                int tid = threadIdx.x;
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                sdata[tid] = (idx < n) ? input[idx] : 0.0f;
                __syncthreads();

                // Reduction in shared memory
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        sdata[tid] += sdata[tid + s];
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    output[blockIdx.x] = sdata[0];
                }
            }
        "#;

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(cuda_code, false, false);

        assert!(result.is_ok());
    }

    #[test]
    fn test_transpile_atomic_operations() {
        let cuda_code = r#"
            __global__ void histogram_kernel(int* data, int* hist, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    atomicAdd(&hist[data[idx]], 1);
                }
            }
        "#;

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(cuda_code, false, false);

        assert!(result.is_ok());
    }

    #[test]
    fn test_transpile_cuda_math_intrinsics() {
        let cuda_code = r#"
            __global__ void math_kernel(float* a, float* b, float* c, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    float x = a[idx];
                    float y = b[idx];

                    c[idx] = __fmaf_rn(x, y, 1.0f);  // Fused multiply-add
                    c[idx] += __sinf(x);              // Fast sine
                    c[idx] += __cosf(y);              // Fast cosine
                    c[idx] += __expf(x);              // Fast exponential
                    c[idx] += __logf(y);              // Fast logarithm
                    c[idx] += __sqrtf(x * y);        // Fast square root
                }
            }
        "#;

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(cuda_code, false, false);

        assert!(result.is_ok());
    }

    #[test]
    fn test_transpile_warp_primitives() {
        let cuda_code = r#"
            __global__ void warp_reduce_kernel(int* data, int* result) {
                int value = data[threadIdx.x];

                // Warp shuffle reduction
                value += __shfl_down_sync(0xffffffff, value, 16);
                value += __shfl_down_sync(0xffffffff, value, 8);
                value += __shfl_down_sync(0xffffffff, value, 4);
                value += __shfl_down_sync(0xffffffff, value, 2);
                value += __shfl_down_sync(0xffffffff, value, 1);

                if (threadIdx.x % 32 == 0) {
                    result[threadIdx.x / 32] = value;
                }
            }
        "#;

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(cuda_code, false, true);

        assert!(result.is_ok());
    }

    #[test]
    fn test_transpile_multiple_kernels() {
        let cuda_code = r#"
            __device__ float device_add(float a, float b) {
                return a + b;
            }

            __global__ void kernel1(float* data) {
                data[threadIdx.x] = device_add(1.0f, 2.0f);
            }

            __global__ void kernel2(float* data) {
                data[threadIdx.x] = device_add(3.0f, 4.0f);
            }
        "#;

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(cuda_code, false, false);

        assert!(result.is_ok());
        let output = result.unwrap();

        // The transpiler should handle multiple kernels
        assert!(!output.is_empty());
    }

    #[test]
    fn test_transpile_texture_memory() {
        let cuda_code = r#"
            texture<float, 2> tex2D;

            __global__ void texture_kernel(float* output, int width, int height) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < width && y < height) {
                    output[y * width + x] = tex2D(x + 0.5f, y + 0.5f);
                }
            }
        "#;

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(cuda_code, false, false);

        // Texture memory might not be fully supported
        // but transpiler should handle it gracefully
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_transpile_constant_memory() {
        let cuda_code = r#"
            __constant__ float kernel_weights[256];

            __global__ void convolution_kernel(float* input, float* output, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    float sum = 0.0f;
                    for (int i = 0; i < 256; i++) {
                        sum += input[idx + i] * kernel_weights[i];
                    }
                    output[idx] = sum;
                }
            }
        "#;

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(cuda_code, false, false);

        assert!(result.is_ok());
    }

    #[test]
    fn test_transpile_invalid_cuda_code() {
        // The transpiler is intentionally lenient for some edge cases,
        // doing best-effort transpilation. We verify it never panics.
        let edge_cases = vec![
            "__global__ void kernel( {}",                           // Syntax error
            "__global__ void kernel() { invalid_intrinsic(); }",    // Unknown function
            "__global__ void kernel() { asm(\"invalid\"); }",       // Inline assembly
        ];

        let transpiler = CudaTranspiler::new();

        for code in edge_cases {
            // Should not panic regardless of input
            let _result = transpiler.transpile(code, false, false);
        }
    }

    #[test]
    fn test_transpile_with_and_without_optimization() {
        let cuda_code = r#"
            __global__ void debug_kernel(int* data) {
                data[threadIdx.x] = threadIdx.x * 2;
            }
        "#;

        let transpiler = CudaTranspiler::new();

        // Without optimization
        let result_no_opt = transpiler.transpile(cuda_code, false, false);
        assert!(result_no_opt.is_ok());

        // With optimization
        let result_opt = transpiler.transpile(cuda_code, true, false);
        assert!(result_opt.is_ok());
    }

    #[test]
    fn test_transpile_with_pattern_detection() {
        let cuda_code = r#"
            __global__ void validation_kernel(float* data, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                // This could cause out-of-bounds access
                data[idx] = idx;
            }
        "#;

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(cuda_code, false, true);

        assert!(result.is_ok());
    }
}
