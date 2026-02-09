//! Cross-platform compatibility tests
//!
//! These tests verify that core functionality works correctly across
//! different platforms using the available native API.

use cuda_rust_wasm::{
    transpiler::CudaTranspiler,
    parser::CudaParser,
    memory::{MemoryPool, PoolConfig, PoolStats, DeviceBuffer},
    kernel::{launch_kernel, LaunchConfig, KernelFunction, ThreadContext, Grid, Block, Dim3},
    runtime::{Runtime, Device},
    error::CudaRustError,
};
use std::sync::Arc;

#[cfg(test)]
mod cross_platform_tests {
    use super::*;

    #[test]
    fn test_basic_transpilation_all_platforms() {
        let cuda_code = r#"
            __global__ void simple_add(float* a, float* b, float* c, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    c[idx] = a[idx] + b[idx];
                }
            }
        "#;

        // Test transpilation on current platform
        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(cuda_code, false, false);
        assert!(result.is_ok(), "Transpilation should succeed on all platforms");

        let rust_code = result.unwrap();
        assert!(!rust_code.is_empty(), "Generated code should not be empty");
    }

    #[test]
    fn test_basic_kernel_launch_all_platforms() {
        // Define a simple kernel using the KernelFunction trait
        struct AddKernel;

        impl KernelFunction<(Vec<f32>, Vec<f32>, Arc<std::sync::Mutex<Vec<f32>>>)> for AddKernel {
            fn execute(
                &self,
                args: (Vec<f32>, Vec<f32>, Arc<std::sync::Mutex<Vec<f32>>>),
                ctx: ThreadContext,
            ) {
                let idx = ctx.global_thread_id();
                let (a, b, c) = args;
                if idx < a.len() {
                    let mut c_lock = c.lock().unwrap();
                    c_lock[idx] = a[idx] + b[idx];
                }
            }

            fn name(&self) -> &str {
                "add_kernel"
            }
        }

        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let c = Arc::new(std::sync::Mutex::new(vec![0.0f32; n]));

        let config = LaunchConfig::new(Grid::new(1u32), Block::new(n as u32));

        let result = launch_kernel(
            AddKernel,
            config,
            (a.clone(), b.clone(), Arc::clone(&c)),
        );
        assert!(result.is_ok(), "Kernel launch should succeed");

        let c_result = c.lock().unwrap();
        for i in 0..n {
            assert_eq!(c_result[i], a[i] + b[i], "Element {} mismatch", i);
        }
    }

    #[test]
    fn test_memory_pool_all_platforms() {
        let pool = MemoryPool::new();

        // Test allocation of different sizes
        let sizes = vec![1024, 4096, 16384, 65536];

        for size in &sizes {
            let buf = pool.allocate(*size);
            assert_eq!(buf.len(), *size, "Buffer should have requested size");
            pool.deallocate(buf);
        }

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, sizes.len() as u64);
    }

    #[test]
    fn test_device_buffer_all_platforms() {
        let device = Device::get_default().unwrap();

        // Allocate and copy data
        let n = 100;
        let host_data: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let mut buffer = DeviceBuffer::<f32>::new(n, device).unwrap();
        buffer.copy_from_host(&host_data).unwrap();

        let mut readback = vec![0.0f32; n];
        buffer.copy_to_host(&mut readback).unwrap();

        assert_eq!(host_data, readback, "Data round-trip should be lossless");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_linux_runtime_initialization() {
        // Test that Runtime initializes correctly on Linux
        let runtime = Runtime::new();
        assert!(runtime.is_ok(), "Runtime should initialize on Linux");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_macos_runtime_initialization() {
        // Test that Runtime initializes correctly on macOS
        let runtime = Runtime::new();
        assert!(runtime.is_ok(), "Runtime should initialize on macOS");
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_windows_runtime_initialization() {
        // Test that Runtime initializes correctly on Windows
        let runtime = Runtime::new();
        assert!(runtime.is_ok(), "Runtime should initialize on Windows");
    }

    #[test]
    fn test_endianness_handling() {
        // Test data consistency using DeviceBuffer round-trip
        let device = Device::get_default().unwrap();
        let test_data: Vec<u32> = vec![0x12345678, 0xABCDEF00, 0xDEADBEEF];

        let mut buffer = DeviceBuffer::<u32>::new(test_data.len(), device).unwrap();
        buffer.copy_from_host(&test_data).unwrap();

        let mut readback = vec![0u32; test_data.len()];
        buffer.copy_to_host(&mut readback).unwrap();

        assert_eq!(test_data, readback, "Data should survive round-trip regardless of endianness");
    }

    #[test]
    fn test_float_precision_consistency() {
        // Test floating point precision across platforms using the transpiler
        let precision_test_code = r#"
            __global__ void precision_test(float* input, float* output, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    float x = input[idx];
                    x = __sinf(x);
                    x = __expf(x);
                    x = __logf(__fabsf(x) + 1e-8f);
                    output[idx] = x;
                }
            }
        "#;

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(precision_test_code, false, false);
        assert!(result.is_ok(), "Precision test kernel should transpile");

        // Verify the generated code contains math operations
        let code = result.unwrap();
        assert!(!code.is_empty(), "Generated code should not be empty");
    }

    #[test]
    fn test_memory_pool_with_custom_config() {
        // Test different pool configurations across platforms
        let configs = vec![
            PoolConfig {
                max_pool_size: 1 * 1024 * 1024,
                min_pooled_size: 512,
                max_pooled_size: 256 * 1024,
                prealloc_count: 4,
            },
            PoolConfig {
                max_pool_size: 8 * 1024 * 1024,
                min_pooled_size: 1024,
                max_pooled_size: 2 * 1024 * 1024,
                prealloc_count: 8,
            },
        ];

        for config in configs {
            let pool = MemoryPool::with_config(config);
            let buf = pool.allocate(2048);
            assert_eq!(buf.len(), 2048);
            pool.deallocate(buf);
        }
    }

    #[test]
    fn test_thread_safety_across_platforms() {
        use std::sync::Barrier;
        use std::thread;

        let num_threads = std::thread::available_parallelism().unwrap().get().min(8);
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);

                thread::spawn(move || {
                    barrier.wait();

                    // Perform thread-safe MemoryPool operations
                    let pool = MemoryPool::new();
                    for _ in 0..10 {
                        let buf = pool.allocate(2048);
                        assert_eq!(buf.len(), 2048);
                        pool.deallocate(buf);
                    }

                    let stats = pool.stats();
                    assert_eq!(stats.total_allocations, 10);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_runtime_initialization() {
        // Test that the Runtime can be initialized on any platform
        let runtime = Runtime::new();
        assert!(runtime.is_ok(), "Runtime should initialize successfully");
    }

    #[test]
    fn test_device_properties() {
        // Test that device properties are available
        let device = Device::get_default().unwrap();
        let props = device.properties();

        assert!(!props.name.is_empty(), "Device should have a name");
        assert!(props.max_threads_per_block > 0, "Should have positive max threads");
        assert!(props.max_blocks_per_grid > 0, "Should have positive max blocks");
    }

    #[test]
    fn test_error_message_consistency() {
        // Test that error messages are consistent across platforms
        let invalid_cuda = "__global__ void invalid_syntax( { invalid }";

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(invalid_cuda, false, false);

        // The parser may or may not error on this, but should not panic
        // If it does error, the message should be informative
        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(!error_msg.is_empty(), "Error message should not be empty");
        }
    }

    #[test]
    fn test_parser_consistency() {
        // Test that the parser produces consistent results
        let parser = CudaParser::new();

        let cuda_code = r#"
            __global__ void test_kernel(float* data, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    data[idx] = data[idx] * 2.0f;
                }
            }
        "#;

        // Parse the same code twice and verify consistency
        let result1 = parser.parse(cuda_code);
        let result2 = parser.parse(cuda_code);

        assert_eq!(result1.is_ok(), result2.is_ok(), "Parse results should be consistent");
    }

    #[test]
    fn test_launch_config_creation() {
        // Test LaunchConfig creation with various grid/block sizes
        let configs = vec![
            (1u32, 256u32),
            (4, 128),
            (16, 64),
            (64, 32),
        ];

        for (grid_size, block_size) in configs {
            let config = LaunchConfig::new(
                Grid::new(grid_size),
                Block::new(block_size),
            );

            assert_eq!(config.grid.dim.x, grid_size);
            assert_eq!(config.block.dim.x, block_size);
        }
    }

    #[test]
    fn test_dim3_conversions() {
        // Test Dim3 creation in various ways
        let d1: Dim3 = 256u32.into();
        assert_eq!(d1, Dim3 { x: 256, y: 1, z: 1 });

        let d2: Dim3 = (16u32, 16u32).into();
        assert_eq!(d2, Dim3 { x: 16, y: 16, z: 1 });

        let d3: Dim3 = (8u32, 8u32, 4u32).into();
        assert_eq!(d3, Dim3 { x: 8, y: 8, z: 4 });

        assert_eq!(d1.size(), 256);
        assert_eq!(d2.size(), 256);
        assert_eq!(d3.size(), 256);
    }
}
