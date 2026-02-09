//! Integration tests for end-to-end CUDA transpilation and CPU kernel execution

#[cfg(test)]
mod integration_tests {
    use cuda_rust_wasm::{CudaRust, CudaParser};
    use cuda_rust_wasm::runtime::{
        Grid, Block, LaunchConfig, KernelFunction, ThreadContext, launch_kernel,
    };
    use cuda_rust_wasm::runtime::grid::Dim3;
    use cuda_rust_wasm::memory::MemoryPool;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    #[test]
    fn test_vector_add_transpile() {
        let cuda_code = r#"
            __global__ void vector_add(float* a, float* b, float* c, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    c[idx] = a[idx] + b[idx];
                }
            }
        "#;

        let transpiler = CudaRust::new();
        let result = transpiler.transpile(cuda_code);
        assert!(result.is_ok(), "Vector add transpilation failed: {:?}", result.err());
        let code = result.unwrap();
        assert!(!code.is_empty());
    }

    // CPU kernel that simulates vector addition
    struct VectorAddKernel {
        a: Vec<f32>,
        b: Vec<f32>,
        c: Arc<Mutex<Vec<f32>>>,
    }

    impl KernelFunction<()> for VectorAddKernel {
        fn execute(&self, _args: (), ctx: ThreadContext) {
            let idx = ctx.global_thread_id();
            if idx < self.a.len() {
                let mut c = self.c.lock().unwrap();
                c[idx] = self.a[idx] + self.b[idx];
            }
        }
        fn name(&self) -> &str { "vector_add" }
    }

    #[test]
    fn test_vector_add_end_to_end() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let c = Arc::new(Mutex::new(vec![0.0f32; n]));

        let kernel = VectorAddKernel { a: a.clone(), b: b.clone(), c: c.clone() };
        let blocks = ((n + 255) / 256) as u32;
        let config = LaunchConfig::new(Grid::new(blocks), Block::new(256u32));
        launch_kernel(kernel, config, ()).unwrap();

        let result = c.lock().unwrap();
        for i in 0..n {
            assert_eq!(result[i], a[i] + b[i], "Mismatch at index {}", i);
        }
    }

    // CPU kernel for matrix multiply
    struct MatMulKernel {
        a: Vec<f32>,
        b: Vec<f32>,
        c: Arc<Mutex<Vec<f32>>>,
        m: usize,
        n: usize,
        k: usize,
    }

    impl KernelFunction<()> for MatMulKernel {
        fn execute(&self, _args: (), ctx: ThreadContext) {
            let (col, row) = ctx.global_thread_id_2d();
            if row < self.m && col < self.n {
                let mut sum = 0.0f32;
                for l in 0..self.k {
                    sum += self.a[row * self.k + l] * self.b[l * self.n + col];
                }
                let mut c = self.c.lock().unwrap();
                c[row * self.n + col] = sum;
            }
        }
        fn name(&self) -> &str { "matrix_multiply" }
    }

    #[test]
    fn test_matrix_multiply_end_to_end() {
        let cuda_code = r#"
            __global__ void matrix_multiply(float* a, float* b, float* c, int m, int n, int k) {
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col = blockIdx.x * blockDim.x + threadIdx.x;
                if (row < m && col < n) {
                    float sum = 0.0f;
                    for (int i = 0; i < k; i++) {
                        sum += a[row * k + i] * b[i * n + col];
                    }
                    c[row * n + col] = sum;
                }
            }
        "#;

        // Verify transpilation
        let transpiler = CudaRust::new();
        assert!(transpiler.transpile(cuda_code).is_ok());

        // CPU kernel execution
        let m = 16;
        let n = 16;
        let k = 16;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 10) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 10) as f32).collect();
        let c = Arc::new(Mutex::new(vec![0.0f32; m * n]));

        let kernel = MatMulKernel {
            a: a.clone(), b: b.clone(), c: c.clone(),
            m, n, k,
        };
        let config = LaunchConfig::new(
            Grid::new((((n + 15) / 16) as u32, ((m + 15) / 16) as u32)),
            Block::new((16u32, 16u32)),
        );
        launch_kernel(kernel, config, ()).unwrap();

        let result = c.lock().unwrap();
        for i in 0..5 {
            for j in 0..5 {
                let mut expected = 0.0f32;
                for l in 0..k {
                    expected += a[i * k + l] * b[l * n + j];
                }
                assert!((result[i * n + j] - expected).abs() < 1e-3,
                    "MatMul mismatch at [{},{}]: expected {}, got {}", i, j, expected, result[i * n + j]);
            }
        }
    }

    #[test]
    fn test_reduction_transpile() {
        let cuda_code = r#"
            __global__ void reduction_sum(float* input, float* output, int n) {
                extern __shared__ float sdata[];
                unsigned int tid = threadIdx.x;
                unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
                sdata[tid] = (idx < n) ? input[idx] : 0.0f;
                __syncthreads();
                for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
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

        let transpiler = CudaRust::new();
        let result = transpiler.transpile(cuda_code);
        assert!(result.is_ok(), "Reduction transpilation failed: {:?}", result.err());
    }

    #[test]
    fn test_atomic_histogram_transpile() {
        let cuda_code = r#"
            __global__ void histogram(int* data, int* hist, int n, int nbins) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    int bin = data[idx] % nbins;
                    atomicAdd(&hist[bin], 1);
                }
            }
        "#;

        let transpiler = CudaRust::new();
        let result = transpiler.transpile(cuda_code);
        assert!(result.is_ok(), "Histogram transpilation failed: {:?}", result.err());
    }

    #[test]
    fn test_performance_measurement() {
        let n = 10000usize;
        let output = Arc::new(Mutex::new(vec![0.0f32; n]));

        struct ComputeKernel {
            output: Arc<Mutex<Vec<f32>>>,
        }
        impl KernelFunction<()> for ComputeKernel {
            fn execute(&self, _args: (), ctx: ThreadContext) {
                let idx = ctx.global_thread_id();
                let mut out = self.output.lock().unwrap();
                if idx < out.len() {
                    let mut x = idx as f32 / 1000.0;
                    for _ in 0..10 {
                        x = x.sin() + x.cos();
                    }
                    out[idx] = x;
                }
            }
            fn name(&self) -> &str { "compute_intensive" }
        }

        let kernel = ComputeKernel { output: output.clone() };
        let blocks = ((n + 255) / 256) as u32;
        let config = LaunchConfig::new(Grid::new(blocks), Block::new(256u32));

        let start = Instant::now();
        launch_kernel(kernel, config, ()).unwrap();
        let duration = start.elapsed();

        println!("Compute intensive kernel took: {:?}", duration);
        assert!(duration.as_secs() < 10, "Kernel took too long");
    }

    // Multi-kernel workflow
    struct ScaleKernel {
        data: Arc<Mutex<Vec<f32>>>,
        factor: f32,
    }
    impl KernelFunction<()> for ScaleKernel {
        fn execute(&self, _args: (), ctx: ThreadContext) {
            let idx = ctx.global_thread_id();
            let mut data = self.data.lock().unwrap();
            if idx < data.len() {
                data[idx] *= self.factor;
            }
        }
        fn name(&self) -> &str { "scale" }
    }

    struct BiasKernel {
        data: Arc<Mutex<Vec<f32>>>,
        bias: f32,
    }
    impl KernelFunction<()> for BiasKernel {
        fn execute(&self, _args: (), ctx: ThreadContext) {
            let idx = ctx.global_thread_id();
            let mut data = self.data.lock().unwrap();
            if idx < data.len() {
                data[idx] += self.bias;
            }
        }
        fn name(&self) -> &str { "add_bias" }
    }

    #[test]
    fn test_multi_kernel_workflow() {
        let n = 1000;
        let data = Arc::new(Mutex::new((0..n).map(|i| i as f32).collect::<Vec<f32>>()));
        let factor = 2.0f32;
        let bias = 10.0f32;

        let blocks = ((n + 255) / 256) as u32;

        // Scale kernel
        let kernel1 = ScaleKernel { data: data.clone(), factor };
        let config = LaunchConfig::new(Grid::new(blocks), Block::new(256u32));
        launch_kernel(kernel1, config, ()).unwrap();

        // Bias kernel
        let kernel2 = BiasKernel { data: data.clone(), bias };
        let config = LaunchConfig::new(Grid::new(blocks), Block::new(256u32));
        launch_kernel(kernel2, config, ()).unwrap();

        let result = data.lock().unwrap();
        for i in 0..n {
            let expected = (i as f32) * factor + bias;
            assert!((result[i] - expected).abs() < 1e-5,
                "Multi-kernel mismatch at {}: expected {}, got {}", i, expected, result[i]);
        }
    }

    #[test]
    fn test_error_handling() {
        let transpiler = CudaRust::new();

        // Invalid CUDA code should not panic
        let invalid_code = "__global__ void invalid( {}";
        let _ = transpiler.transpile(invalid_code);

        // Empty code should not panic
        let _ = transpiler.transpile("");

        // Garbage input should not panic
        let _ = transpiler.transpile("this is not valid CUDA code");
    }
}
