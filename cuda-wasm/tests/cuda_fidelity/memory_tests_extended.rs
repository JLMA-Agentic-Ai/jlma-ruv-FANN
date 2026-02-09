//! Extended tests for memory management
//!
//! These tests verify device buffer allocation, unified memory, shared memory,
//! memory pool behavior, and proper cleanup semantics.

#[cfg(test)]
mod tests {
    use cuda_rust_wasm::memory::{
        DeviceBuffer, HostBuffer, UnifiedMemory,
        MemoryPool, PoolConfig, PoolStats, KernelMemoryManager,
        global_pool, allocate, deallocate,
    };
    use cuda_rust_wasm::runtime::Device;
    use std::sync::Arc;

    // ---------------------------------------------------------------
    // Test 1: Device buffer allocation and basic properties
    // ---------------------------------------------------------------
    #[test]
    fn test_device_buffer_allocation() {
        let device = Device::get_default().expect("Should get default device");
        let buffer = DeviceBuffer::<f32>::new(1024, device).expect("Should allocate buffer");

        assert_eq!(buffer.len(), 1024);
        assert!(!buffer.is_empty());
    }

    // ---------------------------------------------------------------
    // Test 2: Device buffer host-to-device and device-to-host copy
    // ---------------------------------------------------------------
    #[test]
    fn test_device_buffer_copy_roundtrip() {
        let device = Device::get_default().expect("Should get default device");
        let mut buffer = DeviceBuffer::<f32>::new(256, device).expect("Should allocate buffer");

        // Create test data
        let host_data: Vec<f32> = (0..256).map(|i| i as f32 * 1.5).collect();

        // Copy to device
        buffer.copy_from_host(&host_data).expect("Should copy to device");

        // Copy back from device
        let mut result = vec![0.0f32; 256];
        buffer.copy_to_host(&mut result).expect("Should copy from device");

        // Verify round-trip
        assert_eq!(host_data, result, "Data should survive host->device->host round-trip");
    }

    // ---------------------------------------------------------------
    // Test 3: Device buffer fill operation
    // ---------------------------------------------------------------
    #[test]
    fn test_device_buffer_fill() {
        let device = Device::get_default().expect("Should get default device");
        let mut buffer = DeviceBuffer::<f32>::new(100, device).expect("Should allocate buffer");

        buffer.fill(42.0).expect("Should fill buffer");

        let mut result = vec![0.0f32; 100];
        buffer.copy_to_host(&mut result).expect("Should copy from device");

        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, 42.0, "Element {} should be 42.0, got {}", i, val);
        }
    }

    // ---------------------------------------------------------------
    // Test 4: Device buffer zero-length allocation should fail
    // ---------------------------------------------------------------
    #[test]
    fn test_device_buffer_zero_length_fails() {
        let device = Device::get_default().expect("Should get default device");
        let result = DeviceBuffer::<f32>::new(0, device);
        assert!(result.is_err(), "Zero-length allocation should fail");
    }

    // ---------------------------------------------------------------
    // Test 5: Device buffer copy with mismatched lengths should fail
    // ---------------------------------------------------------------
    #[test]
    fn test_device_buffer_copy_mismatch() {
        let device = Device::get_default().expect("Should get default device");
        let mut buffer = DeviceBuffer::<f32>::new(100, device).expect("Should allocate buffer");

        // Too many elements
        let large_data = vec![0.0f32; 200];
        let result = buffer.copy_from_host(&large_data);
        assert!(result.is_err(), "Copying from oversized host buffer should fail");

        // Too few elements
        let small_data = vec![0.0f32; 50];
        let result = buffer.copy_from_host(&small_data);
        assert!(result.is_err(), "Copying from undersized host buffer should fail");
    }

    // ---------------------------------------------------------------
    // Test 6: Unified memory allocation and basic operations
    // ---------------------------------------------------------------
    #[test]
    fn test_unified_memory_allocation() {
        let mem = UnifiedMemory::new(1024).expect("Should allocate unified memory");
        assert_eq!(mem.size(), 1024);
        assert!(!mem.as_ptr().is_null());
    }

    // ---------------------------------------------------------------
    // Test 7: Unified memory read/write roundtrip
    // ---------------------------------------------------------------
    #[test]
    fn test_unified_memory_copy_roundtrip() {
        let mut mem = UnifiedMemory::new(256).expect("Should allocate unified memory");

        let data: Vec<u8> = (0..=255u8).collect();
        mem.copy_from_slice(&data).expect("Should copy data in");

        let mut output = vec![0u8; 256];
        mem.copy_to_slice(&mut output).expect("Should copy data out");

        assert_eq!(data, output, "Unified memory round-trip should preserve data");
    }

    // ---------------------------------------------------------------
    // Test 8: Unified memory zero-size allocation should fail
    // ---------------------------------------------------------------
    #[test]
    fn test_unified_memory_zero_size_fails() {
        let result = UnifiedMemory::new(0);
        assert!(result.is_err(), "Zero-size unified memory allocation should fail");
    }

    // ---------------------------------------------------------------
    // Test 9: Unified memory copy overflow protection
    // ---------------------------------------------------------------
    #[test]
    fn test_unified_memory_overflow_protection() {
        let mut mem = UnifiedMemory::new(100).expect("Should allocate");

        // Try to copy more data than the buffer can hold
        let large_data = vec![0u8; 200];
        let result = mem.copy_from_slice(&large_data);
        assert!(result.is_err(), "Should fail when copying more than buffer size");
    }

    // ---------------------------------------------------------------
    // Test 10: Memory pool basic allocation and deallocation
    // ---------------------------------------------------------------
    #[test]
    fn test_memory_pool_basic() {
        let pool = MemoryPool::new();

        let buffer = pool.allocate(2048);
        assert_eq!(buffer.len(), 2048, "Allocated buffer should be 2048 bytes");

        // Deallocate and reallocate - should get a cache hit
        pool.deallocate(buffer);

        let buffer2 = pool.allocate(2048);
        assert_eq!(buffer2.len(), 2048);

        let stats = pool.stats();
        assert!(stats.total_allocations >= 2, "Should have at least 2 allocations");
    }

    // ---------------------------------------------------------------
    // Test 11: Memory pool cache hit ratio
    // ---------------------------------------------------------------
    #[test]
    fn test_memory_pool_cache_hits() {
        let pool = MemoryPool::new();

        // First allocation (miss or pre-allocated hit)
        let buf1 = pool.allocate(4096);
        pool.deallocate(buf1);

        // Second allocation (should be a cache hit)
        let buf2 = pool.allocate(4096);
        pool.deallocate(buf2);

        let ratio = pool.hit_ratio();
        assert!(
            ratio > 0.0,
            "Hit ratio should be > 0 after reuse. Got: {}",
            ratio
        );
    }

    // ---------------------------------------------------------------
    // Test 12: Memory pool with custom configuration
    // ---------------------------------------------------------------
    #[test]
    fn test_memory_pool_custom_config() {
        let config = PoolConfig {
            max_pool_size: 8 * 1024 * 1024,
            min_pooled_size: 512,
            max_pooled_size: 2 * 1024 * 1024,
            prealloc_count: 4,
        };

        let pool = MemoryPool::with_config(config);

        // Allocate something within pooling range
        let buf = pool.allocate(1024);
        assert_eq!(buf.len(), 1024);
        pool.deallocate(buf);

        // Allocate something below min_pooled_size (should not be pooled)
        let small_buf = pool.allocate(100);
        assert_eq!(small_buf.len(), 100);
    }

    // ---------------------------------------------------------------
    // Test 13: Memory pool power-of-2 behavior (tested indirectly)
    // ---------------------------------------------------------------
    #[test]
    fn test_memory_pool_power_of_2_behavior() {
        let pool = MemoryPool::new();

        // Verify that the pool handles non-power-of-2 allocation sizes correctly.
        // The pool internally rounds to power of 2, so we test by allocating
        // and verifying the buffer size matches our request.
        let sizes = [1000, 1024, 1500, 2000, 3000, 4000, 5000];
        for &size in &sizes {
            let buf = pool.allocate(size);
            assert_eq!(buf.len(), size, "Allocated buffer should match requested size {}", size);
            pool.deallocate(buf);
        }

        // After returning all buffers, cache should contain some entries
        let pooled_memory = pool.total_pooled_memory();
        assert!(pooled_memory > 0, "Pool should retain some buffers for reuse");
    }

    // ---------------------------------------------------------------
    // Test 14: Memory pool clear and reset
    // ---------------------------------------------------------------
    #[test]
    fn test_memory_pool_clear() {
        let pool = MemoryPool::new();

        // Do some allocations
        for _ in 0..10 {
            let buf = pool.allocate(2048);
            pool.deallocate(buf);
        }

        let before = pool.stats();
        assert!(before.total_allocations > 0);

        pool.clear();

        let after = pool.stats();
        assert_eq!(after.total_allocations, 0, "Stats should be reset after clear");
        assert_eq!(after.cache_hits, 0);
        assert_eq!(after.cache_misses, 0);
    }

    // ---------------------------------------------------------------
    // Test 15: Memory pool total pooled memory tracking
    // ---------------------------------------------------------------
    #[test]
    fn test_memory_pool_total_pooled_memory() {
        let pool = MemoryPool::new();

        let initial = pool.total_pooled_memory();

        // Allocate and return to pool
        let buf = pool.allocate(4096);
        pool.deallocate(buf);

        let after = pool.total_pooled_memory();
        assert!(
            after >= initial,
            "Total pooled memory should not decrease after deallocation"
        );
    }

    // ---------------------------------------------------------------
    // Test 16: Memory pool shrink_to_fit
    // ---------------------------------------------------------------
    #[test]
    fn test_memory_pool_shrink_to_fit() {
        let pool = MemoryPool::new();

        // Do many allocations and deallocations
        for size in [1024, 2048, 4096, 8192, 16384] {
            for _ in 0..5 {
                let buf = pool.allocate(size);
                pool.deallocate(buf);
            }
        }

        // Shrink should not panic
        pool.shrink_to_fit();

        // Pool should still be functional after shrink
        let buf = pool.allocate(2048);
        assert_eq!(buf.len(), 2048);
    }

    // ---------------------------------------------------------------
    // Test 17: Global pool access
    // ---------------------------------------------------------------
    #[test]
    fn test_global_pool() {
        let buf = allocate(4096);
        assert_eq!(buf.len(), 4096);

        deallocate(buf);

        let stats = global_pool().stats();
        assert!(stats.total_allocations > 0, "Global pool should track allocations");
    }

    // ---------------------------------------------------------------
    // Test 18: KernelMemoryManager basic usage
    // ---------------------------------------------------------------
    #[test]
    fn test_kernel_memory_manager_basic() {
        let manager = KernelMemoryManager::new();

        unsafe {
            let ptr = manager.allocate_kernel_memory(4096, 16).expect("Should allocate");
            assert!(!ptr.is_null(), "Allocated pointer should not be null");

            let total = manager.total_kernel_memory();
            assert!(total > 0, "Should track allocated memory");

            manager.deallocate_kernel_memory(ptr).expect("Should deallocate");
        }
    }

    // ---------------------------------------------------------------
    // Test 19: HostBuffer allocation and operations
    // ---------------------------------------------------------------
    #[test]
    fn test_host_buffer_operations() {
        let mut buffer = HostBuffer::<f64>::new(100).expect("Should allocate host buffer");

        assert_eq!(buffer.len(), 100);
        assert!(!buffer.is_empty());

        // Fill with a value
        buffer.fill(2.718);

        // Verify via slice
        let slice = buffer.as_slice();
        for &val in slice {
            assert_eq!(val, 2.718);
        }

        // Copy from a slice
        let src: Vec<f64> = (0..100).map(|i| i as f64).collect();
        buffer.copy_from_slice(&src).expect("Should copy from slice");

        // Copy to a slice
        let mut dst = vec![0.0f64; 100];
        buffer.copy_to_slice(&mut dst).expect("Should copy to slice");
        assert_eq!(src, dst);
    }

    // ---------------------------------------------------------------
    // Test 20: HostBuffer indexing
    // ---------------------------------------------------------------
    #[test]
    fn test_host_buffer_indexing() {
        let mut buffer = HostBuffer::<i32>::new(10).expect("Should allocate");

        // Write via mut index
        for i in 0..10 {
            buffer[i] = i as i32 * 10;
        }

        // Read via index
        for i in 0..10 {
            assert_eq!(buffer[i], i as i32 * 10);
        }
    }

    // ---------------------------------------------------------------
    // Test 21: HostBuffer zero-length should fail
    // ---------------------------------------------------------------
    #[test]
    fn test_host_buffer_zero_length_fails() {
        let result = HostBuffer::<f32>::new(0);
        assert!(result.is_err(), "Zero-length host buffer should fail");
    }

    // ---------------------------------------------------------------
    // Test 22: HostBuffer copy length mismatch
    // ---------------------------------------------------------------
    #[test]
    fn test_host_buffer_copy_length_mismatch() {
        let mut buffer = HostBuffer::<i32>::new(10).expect("Should allocate");

        let wrong_size = vec![0i32; 5];
        assert!(
            buffer.copy_from_slice(&wrong_size).is_err(),
            "Copy from wrong-sized slice should fail"
        );

        let mut wrong_dst = vec![0i32; 20];
        assert!(
            buffer.copy_to_slice(&mut wrong_dst).is_err(),
            "Copy to wrong-sized slice should fail"
        );
    }

    // ---------------------------------------------------------------
    // Test 23: Proper cleanup on drop (no leaks or double-free)
    // ---------------------------------------------------------------
    #[test]
    fn test_memory_cleanup_on_drop() {
        // This test ensures that dropping memory objects does not panic
        {
            let device = Device::get_default().unwrap();
            let mut buffer = DeviceBuffer::<u8>::new(4096, device).unwrap();
            let data = vec![0u8; 4096];
            buffer.copy_from_host(&data).unwrap();
            // buffer is dropped here
        }

        {
            let mut mem = UnifiedMemory::new(1024).unwrap();
            let data = vec![0u8; 1024];
            mem.copy_from_slice(&data).unwrap();
            // mem is dropped here
        }

        {
            let mut buf = HostBuffer::<f32>::new(512).unwrap();
            buf.fill(1.0);
            // buf is dropped here
        }

        // If we reach here without panic, cleanup works correctly
    }

    // ---------------------------------------------------------------
    // Test 24: Multiple device buffers on same device
    // ---------------------------------------------------------------
    #[test]
    fn test_multiple_device_buffers() {
        let device = Device::get_default().unwrap();

        let mut buffers: Vec<DeviceBuffer<f32>> = Vec::new();
        for size in [64, 128, 256, 512, 1024] {
            let buf = DeviceBuffer::<f32>::new(size, device.clone())
                .expect(&format!("Should allocate buffer of size {}", size));
            assert_eq!(buf.len(), size);
            buffers.push(buf);
        }

        // All buffers coexist
        assert_eq!(buffers.len(), 5);

        // Drop all at once
        drop(buffers);
    }

    // ---------------------------------------------------------------
    // Test 25: PoolStats default values
    // ---------------------------------------------------------------
    #[test]
    fn test_pool_stats_default() {
        let stats = PoolStats::default();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.total_bytes_allocated, 0);
        assert_eq!(stats.pooled_bytes_served, 0);
        assert_eq!(stats.peak_memory_usage, 0);
        assert_eq!(stats.current_memory_usage, 0);
    }
}
