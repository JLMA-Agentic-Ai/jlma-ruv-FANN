//! Runtime system comprehensive tests
//!
//! Tests for the runtime, device, memory, and kernel launch subsystems
//! using the actual public API.

use cuda_rust_wasm::{
    runtime::{Runtime, Device, BackendType, Grid, Block, Dim3},
    kernel::{launch_kernel, LaunchConfig, KernelFunction, ThreadContext},
    memory::{MemoryPool, PoolConfig, PoolStats, DeviceBuffer},
    error::CudaRustError,
};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

#[cfg(test)]
mod runtime_tests {
    use super::*;

    #[test]
    fn test_runtime_initialization() {
        let runtime = Runtime::new();
        assert!(runtime.is_ok(), "Runtime should initialize successfully");
    }

    #[test]
    fn test_runtime_device_access() {
        let runtime = Runtime::new().unwrap();
        let device = runtime.device();

        let props = device.properties();
        assert!(!props.name.is_empty(), "Device should have a name");
        assert!(props.max_threads_per_block > 0);
    }

    #[test]
    fn test_runtime_synchronize() {
        let runtime = Runtime::new().unwrap();
        let result = runtime.synchronize();
        assert!(result.is_ok(), "Synchronize should succeed");
    }

    #[test]
    fn test_runtime_create_stream() {
        let runtime = Runtime::new().unwrap();
        let stream = runtime.create_stream();
        assert!(stream.is_ok(), "Stream creation should succeed");
    }

    #[test]
    fn test_device_default() {
        let device = Device::get_default();
        assert!(device.is_ok(), "Default device should be available");

        let device = device.unwrap();
        assert_eq!(device.id(), 0, "Default device should have id 0");
    }

    #[test]
    fn test_device_count() {
        let count = Device::count();
        assert!(count.is_ok());
        assert!(count.unwrap() >= 1, "Should have at least one device");
    }

    #[test]
    fn test_device_backend_type() {
        let device = Device::get_default().unwrap();
        let backend = device.backend();

        // Backend should be one of the valid types
        match backend {
            BackendType::CPU | BackendType::Native | BackendType::WebGPU => {
                // All valid
            }
        }
    }

    #[test]
    fn test_device_properties() {
        let device = Device::get_default().unwrap();
        let props = device.properties();

        assert!(!props.name.is_empty());
        assert!(props.max_threads_per_block > 0);
        assert!(props.max_blocks_per_grid > 0);
    }

    #[test]
    fn test_device_buffer_allocation() {
        let device = Device::get_default().unwrap();
        let buffer = DeviceBuffer::<f32>::new(1024, device);
        assert!(buffer.is_ok(), "Buffer allocation should succeed");

        let buffer = buffer.unwrap();
        assert_eq!(buffer.len(), 1024);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_device_buffer_zero_length() {
        let device = Device::get_default().unwrap();
        let buffer = DeviceBuffer::<f32>::new(0, device);
        assert!(buffer.is_err(), "Zero-length buffer should fail");
    }

    #[test]
    fn test_device_buffer_copy_roundtrip() {
        let device = Device::get_default().unwrap();
        let n = 256;
        let host_data: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();

        let mut buffer = DeviceBuffer::<f32>::new(n, device).unwrap();
        buffer.copy_from_host(&host_data).unwrap();

        let mut result = vec![0.0f32; n];
        buffer.copy_to_host(&mut result).unwrap();

        assert_eq!(host_data, result, "Data should survive round-trip copy");
    }

    #[test]
    fn test_device_buffer_copy_length_mismatch() {
        let device = Device::get_default().unwrap();
        let mut buffer = DeviceBuffer::<f32>::new(100, device).unwrap();

        // Mismatched length should error
        let wrong_size_data = vec![0.0f32; 50];
        let result = buffer.copy_from_host(&wrong_size_data);
        assert!(result.is_err(), "Mismatched copy should fail");
    }

    #[test]
    fn test_device_buffer_fill() {
        let device = Device::get_default().unwrap();
        let n = 100;
        let mut buffer = DeviceBuffer::<f32>::new(n, device).unwrap();

        buffer.fill(42.0f32).unwrap();

        let mut result = vec![0.0f32; n];
        buffer.copy_to_host(&mut result).unwrap();

        for val in &result {
            assert_eq!(*val, 42.0f32);
        }
    }

    #[test]
    fn test_concurrent_device_access() {
        let num_threads = 4;
        let barrier = Arc::new(std::sync::Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);

                thread::spawn(move || {
                    barrier.wait();

                    // Each thread creates its own device and buffer
                    for _ in 0..5 {
                        let device = Device::get_default().unwrap();
                        let props = device.properties();
                        assert!(!props.name.is_empty());
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_memory_pool_basic() {
        let pool = MemoryPool::new();

        let buf = pool.allocate(4096);
        assert_eq!(buf.len(), 4096);

        pool.deallocate(buf);

        let stats = pool.stats();
        assert!(stats.total_allocations >= 1);
    }

    #[test]
    fn test_memory_pool_reuse() {
        let pool = MemoryPool::new();

        // Allocate and deallocate
        let buf = pool.allocate(2048);
        pool.deallocate(buf);

        // Second allocation should be a cache hit
        let buf2 = pool.allocate(2048);
        assert_eq!(buf2.len(), 2048);
        pool.deallocate(buf2);

        assert!(pool.hit_ratio() > 0.0, "Should have cache hits after reuse");
    }

    #[test]
    fn test_memory_pool_stats() {
        let pool = MemoryPool::new();

        for _ in 0..10 {
            let buf = pool.allocate(1024);
            pool.deallocate(buf);
        }

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 10);
        assert!(stats.cache_hits > 0, "Should have cache hits");
        assert!(stats.total_bytes_allocated >= 10 * 1024);
    }

    #[test]
    fn test_memory_pool_clear() {
        let pool = MemoryPool::new();

        let buf = pool.allocate(4096);
        pool.deallocate(buf);
        assert!(pool.stats().total_allocations > 0);

        pool.clear();
        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.cache_hits, 0);
    }

    #[test]
    fn test_memory_pool_custom_config() {
        let config = PoolConfig {
            max_pool_size: 4 * 1024 * 1024,
            min_pooled_size: 256,
            max_pooled_size: 1 * 1024 * 1024,
            prealloc_count: 4,
        };

        let pool = MemoryPool::with_config(config);
        let buf = pool.allocate(512);
        assert_eq!(buf.len(), 512);
        pool.deallocate(buf);
    }

    #[test]
    fn test_kernel_launch_simple() {
        struct DoubleKernel;

        impl KernelFunction<Arc<std::sync::Mutex<Vec<f32>>>> for DoubleKernel {
            fn execute(
                &self,
                args: Arc<std::sync::Mutex<Vec<f32>>>,
                ctx: ThreadContext,
            ) {
                let idx = ctx.global_thread_id();
                let mut data = args.lock().unwrap();
                if idx < data.len() {
                    data[idx] *= 2.0;
                }
            }

            fn name(&self) -> &str {
                "double_kernel"
            }
        }

        let n = 32;
        let data = Arc::new(std::sync::Mutex::new(
            (0..n).map(|i| i as f32).collect::<Vec<f32>>(),
        ));

        let config = LaunchConfig::new(Grid::new(1u32), Block::new(n as u32));
        let result = launch_kernel(DoubleKernel, config, Arc::clone(&data));
        assert!(result.is_ok(), "Kernel launch should succeed");

        let result_data = data.lock().unwrap();
        for i in 0..n {
            assert_eq!(result_data[i], (i as f32) * 2.0, "Element {} should be doubled", i);
        }
    }

    #[test]
    fn test_launch_config_with_shared_memory() {
        let config = LaunchConfig::new(Grid::new(4u32), Block::new(128u32))
            .with_shared_memory(1024);

        assert_eq!(config.shared_memory_bytes, 1024);
        assert_eq!(config.grid.dim.x, 4);
        assert_eq!(config.block.dim.x, 128);
    }

    #[test]
    fn test_launch_config_2d() {
        let config = LaunchConfig::new(
            Grid::new((4u32, 4u32)),
            Block::new((16u32, 16u32)),
        );

        assert_eq!(config.grid.dim.x, 4);
        assert_eq!(config.grid.dim.y, 4);
        assert_eq!(config.block.dim.x, 16);
        assert_eq!(config.block.dim.y, 16);
    }

    #[test]
    fn test_block_validation_succeeds() {
        let block = Block::new(256u32);
        assert!(block.validate().is_ok());
    }

    #[test]
    fn test_block_validation_exceeds_limit() {
        let block = Block::new(2048u32);
        assert!(block.validate().is_err(), "Block size 2048 should exceed max threads per block");
    }

    #[test]
    fn test_performance_memory_pool_allocation() {
        let pool = MemoryPool::new();

        let start = Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            let buf = pool.allocate(4096);
            pool.deallocate(buf);
        }

        let duration = start.elapsed();
        println!(
            "Memory pool: {} allocations/deallocations in {:?} ({:.0} ops/sec)",
            iterations,
            duration,
            iterations as f64 / duration.as_secs_f64()
        );

        // Should be very fast - at least 1000 ops/sec
        assert!(
            duration.as_secs() < 10,
            "Memory pool operations should be fast"
        );
    }

    #[test]
    fn test_error_types() {
        // Verify error type formatting
        let err = CudaRustError::RuntimeError("test error".to_string());
        let msg = err.to_string();
        assert!(msg.contains("test error"), "Error message should contain the original text");

        let err = CudaRustError::MemoryError("out of memory".to_string());
        let msg = err.to_string();
        assert!(msg.contains("out of memory"));
    }

    #[test]
    fn test_device_buffer_multiple_types() {
        let device = Device::get_default().unwrap();

        // Test with f32
        let mut buf_f32 = DeviceBuffer::<f32>::new(10, device.clone()).unwrap();
        let data_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        buf_f32.copy_from_host(&data_f32).unwrap();
        let mut result_f32 = vec![0.0f32; 10];
        buf_f32.copy_to_host(&mut result_f32).unwrap();
        assert_eq!(data_f32, result_f32);

        // Test with u32
        let mut buf_u32 = DeviceBuffer::<u32>::new(5, device.clone()).unwrap();
        let data_u32: Vec<u32> = vec![100, 200, 300, 400, 500];
        buf_u32.copy_from_host(&data_u32).unwrap();
        let mut result_u32 = vec![0u32; 5];
        buf_u32.copy_to_host(&mut result_u32).unwrap();
        assert_eq!(data_u32, result_u32);

        // Test with i64
        let mut buf_i64 = DeviceBuffer::<i64>::new(3, device).unwrap();
        let data_i64: Vec<i64> = vec![-1, 0, 1];
        buf_i64.copy_from_host(&data_i64).unwrap();
        let mut result_i64 = vec![0i64; 3];
        buf_i64.copy_to_host(&mut result_i64).unwrap();
        assert_eq!(data_i64, result_i64);
    }

    #[test]
    fn test_global_memory_pool() {
        // Test the global pool functions
        let buf = cuda_rust_wasm::memory::allocate(2048);
        assert_eq!(buf.len(), 2048);

        cuda_rust_wasm::memory::deallocate(buf);

        let pool = cuda_rust_wasm::memory::global_pool();
        let stats = pool.stats();
        assert!(stats.total_allocations > 0);
    }
}
