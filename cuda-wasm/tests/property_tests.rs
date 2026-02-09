//! Property-based tests for transpiler correctness and runtime behavior

#[cfg(test)]
mod property_tests {
    use cuda_rust_wasm::{CudaParser, CudaRust};
    use cuda_rust_wasm::runtime::{
        Grid, Block, Dim3, LaunchConfig, KernelFunction, ThreadContext, launch_kernel,
    };
    use cuda_rust_wasm::memory::MemoryPool;

    // Property: Valid CUDA kernels should parse and transpile successfully
    #[test]
    fn prop_valid_kernels_transpile() {
        let kernels = vec![
            r#"__global__ void add(float* a, float* b, float* c) { int i = threadIdx.x; c[i] = a[i] + b[i]; }"#,
            r#"__global__ void scale(float* data, float s, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) data[i] *= s; }"#,
            r#"__global__ void identity(float* in, float* out, int n) { int i = threadIdx.x; if (i < n) out[i] = in[i]; }"#,
            r#"__global__ void saxpy(float a, float* x, float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a * x[i] + y[i]; }"#,
        ];

        let transpiler = CudaRust::new();
        for (idx, kernel) in kernels.iter().enumerate() {
            let result = transpiler.transpile(kernel);
            assert!(result.is_ok(), "Kernel {} failed to transpile: {:?}", idx, result.err());
            let code = result.unwrap();
            assert!(!code.is_empty(), "Kernel {} produced empty output", idx);
        }
    }

    // Property: Parser should handle all valid CUDA types
    #[test]
    fn prop_parser_handles_types() {
        let parser = CudaParser::new();
        let typed_kernels = vec![
            r#"__global__ void k1(int* a) { int i = threadIdx.x; a[i] = i; }"#,
            r#"__global__ void k2(float* a) { int i = threadIdx.x; a[i] = 1.0f; }"#,
            r#"__global__ void k3(double* a) { int i = threadIdx.x; a[i] = 1.0; }"#,
            r#"__global__ void k4(unsigned int* a) { int i = threadIdx.x; a[i] = i; }"#,
        ];

        for (idx, kernel) in typed_kernels.iter().enumerate() {
            let result = parser.parse(kernel);
            assert!(result.is_ok(), "Type kernel {} failed: {:?}", idx, result.err());
        }
    }

    // Property: Vector addition via CPU kernel should be commutative
    struct VectorAddKernel {
        a: Vec<f32>,
        b: Vec<f32>,
        c: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
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
    fn prop_vector_add_commutative() {
        let n = 256;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.7).collect();

        // a + b
        let c1 = std::sync::Arc::new(std::sync::Mutex::new(vec![0.0f32; n]));
        let kernel1 = VectorAddKernel { a: a.clone(), b: b.clone(), c: c1.clone() };
        let config = LaunchConfig::new(Grid::new(1u32), Block::new(n as u32));
        launch_kernel(kernel1, config, ()).unwrap();

        // b + a
        let c2 = std::sync::Arc::new(std::sync::Mutex::new(vec![0.0f32; n]));
        let kernel2 = VectorAddKernel { a: b.clone(), b: a.clone(), c: c2.clone() };
        let config = LaunchConfig::new(Grid::new(1u32), Block::new(n as u32));
        launch_kernel(kernel2, config, ()).unwrap();

        let r1 = c1.lock().unwrap();
        let r2 = c2.lock().unwrap();
        for i in 0..n {
            assert!((r1[i] - r2[i]).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r1[i], r2[i]);
        }
    }

    // Property: Identity kernel preserves data
    struct IdentityKernel {
        input: Vec<f32>,
        output: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    }

    impl KernelFunction<()> for IdentityKernel {
        fn execute(&self, _args: (), ctx: ThreadContext) {
            let idx = ctx.global_thread_id();
            if idx < self.input.len() {
                let mut out = self.output.lock().unwrap();
                out[idx] = self.input[idx];
            }
        }
        fn name(&self) -> &str { "identity" }
    }

    #[test]
    fn prop_identity_preserves_data() {
        let data: Vec<f32> = (0..512).map(|i| (i as f32).sin()).collect();
        let output = std::sync::Arc::new(std::sync::Mutex::new(vec![0.0f32; 512]));

        let kernel = IdentityKernel { input: data.clone(), output: output.clone() };
        let config = LaunchConfig::new(Grid::new(2u32), Block::new(256u32));
        launch_kernel(kernel, config, ()).unwrap();

        let result = output.lock().unwrap();
        for i in 0..data.len() {
            assert_eq!(result[i], data[i], "Identity failed at index {}", i);
        }
    }

    // Property: Multi-block kernel covers all elements
    struct CountKernel {
        output: std::sync::Arc<std::sync::Mutex<Vec<u32>>>,
    }

    impl KernelFunction<()> for CountKernel {
        fn execute(&self, _args: (), ctx: ThreadContext) {
            let idx = ctx.global_thread_id();
            let mut out = self.output.lock().unwrap();
            if idx < out.len() {
                out[idx] = idx as u32;
            }
        }
        fn name(&self) -> &str { "count" }
    }

    #[test]
    fn prop_multi_block_coverage() {
        let n = 1024usize;
        let output = std::sync::Arc::new(std::sync::Mutex::new(vec![u32::MAX; n]));

        let kernel = CountKernel { output: output.clone() };
        let blocks = ((n + 255) / 256) as u32;
        let config = LaunchConfig::new(Grid::new(blocks), Block::new(256u32));
        launch_kernel(kernel, config, ()).unwrap();

        let result = output.lock().unwrap();
        for i in 0..n {
            assert_eq!(result[i], i as u32, "Thread {} not reached", i);
        }
    }

    // Property: Scalar multiplication is distributive
    struct ScalarMulKernel {
        data: Vec<f32>,
        scalar: f32,
        output: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    }

    impl KernelFunction<()> for ScalarMulKernel {
        fn execute(&self, _args: (), ctx: ThreadContext) {
            let idx = ctx.global_thread_id();
            if idx < self.data.len() {
                let mut out = self.output.lock().unwrap();
                out[idx] = self.data[idx] * self.scalar;
            }
        }
        fn name(&self) -> &str { "scalar_mul" }
    }

    #[test]
    fn prop_scalar_mult_correct() {
        let n = 256;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let scalar = 3.14f32;
        let output = std::sync::Arc::new(std::sync::Mutex::new(vec![0.0f32; n]));

        let kernel = ScalarMulKernel { data: data.clone(), scalar, output: output.clone() };
        let config = LaunchConfig::new(Grid::new(1u32), Block::new(n as u32));
        launch_kernel(kernel, config, ()).unwrap();

        let result = output.lock().unwrap();
        for i in 0..n {
            assert!((result[i] - data[i] * scalar).abs() < 1e-4,
                "Scalar mult mismatch at {}: expected {}, got {}", i, data[i] * scalar, result[i]);
        }
    }

    // Property: Memory pool allocation/deallocation is consistent
    #[test]
    fn prop_memory_pool_consistency() {
        let pool = MemoryPool::new();

        // Allocate various sizes and verify buffers are returned with correct length
        let sizes = [64usize, 128, 256, 512, 1024, 4096];
        for &size in &sizes {
            let buffer = pool.allocate(size);
            assert_eq!(buffer.len(), size, "Buffer size mismatch for allocation of {} bytes", size);
            // Return buffer to pool for reuse
            pool.deallocate(buffer);
        }

        // After allocating and deallocating, stats should show activity
        let stats = pool.stats();
        assert!(stats.total_allocations >= sizes.len() as u64,
            "Expected at least {} allocations, got {}", sizes.len(), stats.total_allocations);
    }

    // Property: Memory pool reuse works (cache hits after deallocation)
    #[test]
    fn prop_memory_pool_reuse() {
        let pool = MemoryPool::new();

        // Allocate and deallocate a poolable size (>= min_pooled_size of 1024)
        let buffer = pool.allocate(2048);
        assert_eq!(buffer.len(), 2048);
        pool.deallocate(buffer);

        // Allocate the same size again -- should be a cache hit
        let _buffer2 = pool.allocate(2048);
        let stats = pool.stats();
        assert!(stats.cache_hits > 0, "Expected cache hits after reuse, got {}", stats.cache_hits);
    }

    // Property: Grid/Block dimensions calculate correctly
    #[test]
    fn prop_grid_block_dimensions() {
        for x in [1u32, 2, 4, 8, 16, 32, 64, 128, 256] {
            let grid = Grid::new(x);
            assert_eq!(grid.num_blocks(), x);

            let block = Block::new(x);
            assert_eq!(block.num_threads(), x);
        }

        // 2D
        let grid_2d = Grid::new((4u32, 4u32));
        assert_eq!(grid_2d.num_blocks(), 16);

        // 3D
        let grid_3d = Grid::new((2u32, 3u32, 4u32));
        assert_eq!(grid_3d.num_blocks(), 24);
    }

    // Property: Block validation catches invalid sizes
    #[test]
    fn prop_block_validation() {
        // Valid blocks
        assert!(Block::new(1u32).validate().is_ok());
        assert!(Block::new(256u32).validate().is_ok());
        assert!(Block::new(1024u32).validate().is_ok());

        // Invalid: too many threads (max is 1024)
        assert!(Block::new(2048u32).validate().is_err());
    }

    // Property: ThreadContext computes correct global IDs
    #[test]
    fn prop_thread_context_global_ids() {
        let ctx = ThreadContext {
            thread_idx: Dim3 { x: 5, y: 0, z: 0 },
            block_idx: Dim3 { x: 2, y: 0, z: 0 },
            block_dim: Dim3 { x: 256, y: 1, z: 1 },
            grid_dim: Dim3 { x: 4, y: 1, z: 1 },
        };
        assert_eq!(ctx.global_thread_id(), 2 * 256 + 5);

        let ctx_2d = ThreadContext {
            thread_idx: Dim3 { x: 3, y: 7, z: 0 },
            block_idx: Dim3 { x: 1, y: 2, z: 0 },
            block_dim: Dim3 { x: 16, y: 16, z: 1 },
            grid_dim: Dim3 { x: 4, y: 4, z: 1 },
        };
        let (gx, gy) = ctx_2d.global_thread_id_2d();
        assert_eq!(gx, 1 * 16 + 3);
        assert_eq!(gy, 2 * 16 + 7);
    }

    // Property: Transpiler produces deterministic output
    #[test]
    fn prop_transpile_deterministic() {
        let kernel = r#"__global__ void add(float* a, float* b, float* c) {
            int i = threadIdx.x;
            c[i] = a[i] + b[i];
        }"#;

        let transpiler = CudaRust::new();
        let result1 = transpiler.transpile(kernel).unwrap();
        let result2 = transpiler.transpile(kernel).unwrap();
        assert_eq!(result1, result2, "Transpiler should produce deterministic output");
    }

    // Property: Empty or invalid kernels should fail gracefully
    #[test]
    fn prop_invalid_kernels_handled() {
        let transpiler = CudaRust::new();

        // Empty input
        let result = transpiler.transpile("");
        // Should either succeed with empty output or return an error, but not panic
        let _ = result;

        // Garbage input
        let result = transpiler.transpile("this is not valid CUDA code at all!!!");
        // Should not panic
        let _ = result;
    }
}
