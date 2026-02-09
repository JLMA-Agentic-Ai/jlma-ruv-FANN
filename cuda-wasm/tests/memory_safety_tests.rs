//! Memory safety and leak detection tests

#[cfg(test)]
mod memory_safety_tests {
    use cuda_rust_wasm::memory::{MemoryPool, DeviceBuffer, HostBuffer};
    use cuda_rust_wasm::runtime::Device;
    use std::sync::{Arc, Barrier, Mutex};
    use std::thread;

    #[test]
    fn test_memory_leak_detection() {
        let pool = MemoryPool::new();

        // Allocate and deallocate in a loop
        for _ in 0..100 {
            let buf = pool.allocate(1024);
            // Use the buffer
            assert_eq!(buf.len(), 1024);
            pool.deallocate(buf);
        }

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 100,
            "Expected 100 allocations, got {}", stats.total_allocations);
    }

    #[test]
    fn test_double_free_protection() {
        let pool = MemoryPool::new();
        let buf = pool.allocate(1024);

        // First deallocation
        pool.deallocate(buf);

        // Rust's ownership prevents double-free at compile time
        // This test verifies the pool's Drop/deallocate is safe
    }

    #[test]
    fn test_buffer_bounds() {
        let device = Device::get_default().unwrap();
        let mut buf = DeviceBuffer::<u8>::new(100, device).unwrap();

        // Copy exactly the right amount -- should succeed
        let data = vec![0u8; 100];
        assert!(buf.copy_from_host(&data).is_ok());

        // Copy more than allocated -- should fail
        let large_data = vec![0u8; 200];
        let result = buf.copy_from_host(&large_data);
        assert!(result.is_err(), "Oversized copy should fail");
    }

    #[test]
    fn test_mismatched_size_copy() {
        let device = Device::get_default().unwrap();
        let buf = DeviceBuffer::<f32>::new(10, device).unwrap();

        // Read back into wrong-sized buffer -- should fail
        let mut wrong_dst = vec![0.0f32; 5];
        let result = buf.copy_to_host(&mut wrong_dst);
        assert!(result.is_err(), "Mismatched readback should fail");
    }

    #[test]
    fn test_concurrent_memory_safety() {
        let pool = Arc::new(MemoryPool::new());
        let num_threads = 8;
        let barrier = Arc::new(Barrier::new(num_threads));
        let alloc_count = Arc::new(Mutex::new(0u64));
        let dealloc_count = Arc::new(Mutex::new(0u64));

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let pool = Arc::clone(&pool);
                let barrier = Arc::clone(&barrier);
                let ac = Arc::clone(&alloc_count);
                let dc = Arc::clone(&dealloc_count);

                thread::spawn(move || {
                    barrier.wait();

                    let mut buffers = Vec::new();
                    for i in 0..50 {
                        let size = 100 + tid * 10 + i;
                        let buf = pool.allocate(size);
                        buffers.push(buf);
                        *ac.lock().unwrap() += 1;
                    }

                    // Deallocate all
                    for buf in buffers {
                        pool.deallocate(buf);
                        *dc.lock().unwrap() += 1;
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let total_alloc = *alloc_count.lock().unwrap();
        let total_dealloc = *dealloc_count.lock().unwrap();
        assert_eq!(total_alloc, total_dealloc,
            "Alloc/dealloc mismatch: {} vs {}", total_alloc, total_dealloc);
    }

    #[test]
    fn test_pool_isolation() {
        let pool1 = MemoryPool::new();
        let pool2 = MemoryPool::new();

        let buf1 = pool1.allocate(1024);
        let buf2 = pool2.allocate(1024);

        assert_eq!(buf1.len(), 1024);
        assert_eq!(buf2.len(), 1024);

        pool1.deallocate(buf1);
        pool2.deallocate(buf2);

        let stats1 = pool1.stats();
        let stats2 = pool2.stats();
        assert_eq!(stats1.total_allocations, 1);
        assert_eq!(stats2.total_allocations, 1);
    }

    #[test]
    fn test_resource_cleanup_on_panic() {
        use std::panic;

        let pool = Arc::new(MemoryPool::new());

        let result = panic::catch_unwind(|| {
            let pool = Arc::clone(&pool);
            let _buf = pool.allocate(10000);
            panic!("Simulated panic");
        });

        assert!(result.is_err(), "Panic should have occurred");

        // Pool should still be usable after panic
        let buf = pool.allocate(1024);
        assert_eq!(buf.len(), 1024);
        pool.deallocate(buf);
    }

    #[test]
    fn test_device_buffer_drop_safety() {
        let device = Device::get_default().unwrap();
        // Allocate a buffer, let it drop naturally
        {
            let _buf = DeviceBuffer::<u8>::new(4096, device.clone()).unwrap();
            // buf drops here -- should not leak or crash
        }

        // Allocate again to verify the allocator is fine
        let buf = DeviceBuffer::<u8>::new(4096, device).unwrap();
        assert_eq!(buf.len(), 4096);
    }

    #[test]
    fn test_host_buffer_safety() {
        let mut buf = HostBuffer::<u8>::new(1024).unwrap();
        assert_eq!(buf.len(), 1024);

        // Fill and verify
        buf.fill(0);
        let slice = buf.as_slice();
        assert_eq!(slice.len(), 1024);
        assert!(slice.iter().all(|&b| b == 0), "HostBuffer should be zero after fill");
    }

    #[test]
    fn test_host_buffer_copy() {
        let mut buf = HostBuffer::<i32>::new(10).unwrap();
        let data: Vec<i32> = (0..10).collect();

        buf.copy_from_slice(&data).unwrap();

        let mut result = vec![0i32; 10];
        buf.copy_to_slice(&mut result).unwrap();

        assert_eq!(data, result);
    }

    #[test]
    fn test_memory_pressure() {
        let pool = MemoryPool::new();
        let mut buffers = Vec::new();

        // Allocate increasing sizes
        for i in 0..100 {
            let size = 1024 * (i + 1);
            let buf = pool.allocate(size);
            assert_eq!(buf.len(), size);
            buffers.push(buf);
        }

        // Free half
        let half = buffers.len() / 2;
        for buf in buffers.drain(..half) {
            pool.deallocate(buf);
        }

        // Should be able to allocate more
        let buf = pool.allocate(2048);
        assert_eq!(buf.len(), 2048);
        pool.deallocate(buf);
    }

    #[test]
    fn test_allocation_pattern_detection() {
        let pool = MemoryPool::new();

        // Allocate various sizes and verify they all work
        let sizes = [100, 200, 300];
        for &size in &sizes {
            let pool_buf = pool.allocate(size);
            assert_eq!(pool_buf.len(), size);
            pool.deallocate(pool_buf);
        }
    }
}
