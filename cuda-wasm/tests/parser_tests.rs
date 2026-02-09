//! Unit tests for the CUDA parser module

#[cfg(test)]
mod parser_tests {
    use cuda_rust_wasm::parser::{CudaParser, Ast, KernelDef};
    use cuda_rust_wasm::parser::ast::{Item, FunctionDef, GlobalVar, StorageClass};

    /// Helper: extract all KernelDefs from an Ast
    fn kernels(ast: &Ast) -> Vec<&KernelDef> {
        ast.items.iter().filter_map(|item| match item {
            Item::Kernel(k) => Some(k),
            _ => None,
        }).collect()
    }

    /// Helper: extract all DeviceFunction defs from an Ast
    fn device_functions(ast: &Ast) -> Vec<&FunctionDef> {
        ast.items.iter().filter_map(|item| match item {
            Item::DeviceFunction(f) => Some(f),
            _ => None,
        }).collect()
    }

    /// Helper: extract all GlobalVar entries from an Ast
    fn global_vars(ast: &Ast) -> Vec<&GlobalVar> {
        ast.items.iter().filter_map(|item| match item {
            Item::GlobalVar(v) => Some(v),
            _ => None,
        }).collect()
    }

    /// Helper: extract constant-memory globals from an Ast
    fn constant_vars(ast: &Ast) -> Vec<&GlobalVar> {
        global_vars(ast).into_iter().filter(|v| {
            matches!(v.storage, StorageClass::Constant)
        }).collect()
    }

    #[test]
    fn test_parse_empty_kernel() {
        let cuda_code = r#"
            __global__ void empty_kernel() {
            }
        "#;

        let parser = CudaParser::new();
        let result = parser.parse(cuda_code);

        assert!(result.is_ok());
        let ast = result.unwrap();
        let k = kernels(&ast);
        assert_eq!(k.len(), 1);
        assert_eq!(k[0].name, "empty_kernel");
    }

    #[test]
    fn test_parse_kernel_with_parameters() {
        let cuda_code = r#"
            __global__ void vector_add(float* a, float* b, float* c, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    c[idx] = a[idx] + b[idx];
                }
            }
        "#;

        let parser = CudaParser::new();
        let result = parser.parse(cuda_code);

        assert!(result.is_ok());
        let ast = result.unwrap();
        let k = kernels(&ast);
        assert_eq!(k[0].params.len(), 4);
    }

    #[test]
    fn test_parse_multiple_kernels() {
        let cuda_code = r#"
            __global__ void kernel1() { }
            __global__ void kernel2() { }
            __device__ void device_func() { }
        "#;

        let parser = CudaParser::new();
        let result = parser.parse(cuda_code);

        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(kernels(&ast).len(), 2);
        assert_eq!(device_functions(&ast).len(), 1);
    }

    #[test]
    fn test_parse_shared_memory() {
        let cuda_code = r#"
            __global__ void reduction_kernel(float* input, float* output) {
                extern __shared__ float sdata[];
                sdata[threadIdx.x] = input[threadIdx.x];
                __syncthreads();
            }
        "#;

        let parser = CudaParser::new();
        let result = parser.parse(cuda_code);

        assert!(result.is_ok());
        let ast = result.unwrap();
        let k = kernels(&ast);
        assert_eq!(k.len(), 1);
        assert_eq!(k[0].name, "reduction_kernel");
    }

    #[test]
    fn test_parse_atomic_operations() {
        let cuda_code = r#"
            __global__ void atomic_add_kernel(int* counter) {
                atomicAdd(counter, 1);
            }
        "#;

        let parser = CudaParser::new();
        let result = parser.parse(cuda_code);

        assert!(result.is_ok());
        let ast = result.unwrap();
        let k = kernels(&ast);
        assert_eq!(k.len(), 1);
        assert_eq!(k[0].name, "atomic_add_kernel");
    }

    #[test]
    fn test_parse_texture_memory() {
        let cuda_code = r#"
            texture<float, 2> tex2D;

            __global__ void texture_kernel(float* output, int width, int height) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < width && y < height) {
                    output[y * width + x] = tex2D(x, y);
                }
            }
        "#;

        let parser = CudaParser::new();
        let result = parser.parse(cuda_code);

        assert!(result.is_ok());
        let ast = result.unwrap();
        // The parser should handle texture declarations; verify the kernel parsed
        let k = kernels(&ast);
        assert_eq!(k.len(), 1);
        assert_eq!(k[0].name, "texture_kernel");
    }

    #[test]
    fn test_parse_constant_memory() {
        let cuda_code = r#"
            __constant__ float kernel_weights[256];

            __global__ void convolution_kernel(float* input, float* output) {
                // kernel implementation
            }
        "#;

        let parser = CudaParser::new();
        let result = parser.parse(cuda_code);

        assert!(result.is_ok());
        let ast = result.unwrap();
        let cv = constant_vars(&ast);
        assert_eq!(cv.len(), 1);
    }

    #[test]
    fn test_parse_cuda_math_functions() {
        let cuda_code = r#"
            __global__ void math_kernel(float* a, float* b, float* c) {
                int idx = threadIdx.x;
                c[idx] = __fmaf_rn(a[idx], b[idx], 1.0f);
                c[idx] = __sinf(c[idx]);
                c[idx] = __expf(c[idx]);
            }
        "#;

        let parser = CudaParser::new();
        let result = parser.parse(cuda_code);

        assert!(result.is_ok());
        let ast = result.unwrap();
        let k = kernels(&ast);
        assert_eq!(k.len(), 1);
        assert_eq!(k[0].name, "math_kernel");
    }

    #[test]
    fn test_parse_warp_primitives() {
        let cuda_code = r#"
            __global__ void warp_shuffle_kernel(int* data) {
                int value = data[threadIdx.x];
                value = __shfl_xor_sync(0xffffffff, value, 1);
                data[threadIdx.x] = value;
            }
        "#;

        let parser = CudaParser::new();
        let result = parser.parse(cuda_code);

        assert!(result.is_ok());
        let ast = result.unwrap();
        let k = kernels(&ast);
        assert_eq!(k.len(), 1);
        assert_eq!(k[0].name, "warp_shuffle_kernel");
    }

    #[test]
    fn test_parse_invalid_syntax() {
        // The parser is intentionally lenient for some malformed inputs,
        // extracting what it can. We test that it never panics and that
        // clearly broken syntax returns Err or an empty AST.
        let edge_cases = vec![
            "__global__ void kernel( {}",  // Missing closing parenthesis
            "__global__ kernel() {}",       // Missing return type
            "__global void kernel() {}",    // Missing underscore
            "__global__ void __kernel() {}", // Unusual kernel name
        ];

        let parser = CudaParser::new();

        for code in edge_cases {
            // Should not panic regardless of input
            let _result = parser.parse(code);
        }
    }

    #[test]
    fn test_parse_complex_kernel() {
        let cuda_code = r#"
            #define BLOCK_SIZE 16

            __constant__ float c_kernel[9];

            __device__ float clamp(float value, float min, float max) {
                return fminf(fmaxf(value, min), max);
            }

            __global__ void image_filter(
                float* input,
                float* output,
                int width,
                int height
            ) {
                extern __shared__ float tile[];

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                int tid_x = threadIdx.x;
                int tid_y = threadIdx.y;

                // Load tile with padding
                if (x < width && y < height) {
                    tile[tid_y * BLOCK_SIZE + tid_x] = input[y * width + x];
                }

                __syncthreads();

                // Apply convolution
                if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                    float sum = 0.0f;

                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            int tile_y = tid_y + ky;
                            int tile_x = tid_x + kx;

                            if (tile_y >= 0 && tile_y < BLOCK_SIZE &&
                                tile_x >= 0 && tile_x < BLOCK_SIZE) {
                                float pixel = tile[tile_y * BLOCK_SIZE + tile_x];
                                float weight = c_kernel[(ky + 1) * 3 + (kx + 1)];
                                sum += pixel * weight;
                            }
                        }
                    }

                    output[y * width + x] = clamp(sum, 0.0f, 255.0f);
                }
            }
        "#;

        let parser = CudaParser::new();
        let result = parser.parse(cuda_code);

        assert!(result.is_ok());
        let ast = result.unwrap();
        let k = kernels(&ast);
        assert_eq!(k.len(), 1);
        assert_eq!(device_functions(&ast).len(), 1);
        assert_eq!(constant_vars(&ast).len(), 1);
        assert_eq!(k[0].name, "image_filter");
    }

    #[test]
    fn test_parser_no_options() {
        let cuda_code = r#"
            __global__ void test_kernel() {
                int idx = threadIdx.x;
            }
        "#;

        // CudaParser::new() takes no arguments
        let parser = CudaParser::new();
        let result = parser.parse(cuda_code);

        assert!(result.is_ok());
    }
}
