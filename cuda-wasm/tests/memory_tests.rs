//! Unit tests for memory management

#[cfg(test)]
mod memory_tests {
    use cuda_rust_wasm::memory::{MemoryPool, DeviceBuffer, HostBuffer};
    use cuda_rust_wasm::runtime::Device;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_memory_pool_creation() {
        let pool = MemoryPool::new();
        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.current_memory_usage, 0);
    }

    #[test]
    fn test_memory_pool_allocate() {
        let pool = MemoryPool::new();

        let buf = pool.allocate(1024);
        assert_eq!(buf.len(), 1024);

        let stats = pool.stats();
        assert!(stats.total_allocations >= 1);
    }

    #[test]
    fn test_memory_pool_deallocate() {
        let pool = MemoryPool::new();

        let buf = pool.allocate(2048);
        assert_eq!(buf.len(), 2048);
        pool.deallocate(buf);

        let stats = pool.stats();
        // After deallocation, memory usage should decrease
        assert!(stats.total_allocations >= 1);
    }

    #[test]
    fn test_memory_pool_multiple_allocations() {
        let pool = MemoryPool::new();

        let sizes = [64, 128, 256, 512, 1024, 4096, 8192];
        let mut buffers = Vec::new();

        for &size in &sizes {
            let buf = pool.allocate(size);
            assert_eq!(buf.len(), size, "Buffer size mismatch for {} bytes", size);
            buffers.push(buf);
        }

        let stats = pool.stats();
        assert!(stats.total_allocations >= sizes.len() as u64);
    }

    #[test]
    fn test_memory_pool_reuse() {
        let pool = MemoryPool::new();

        // Allocate a large-enough buffer that gets pooled
        let buf = pool.allocate(2048);
        pool.deallocate(buf);

        // Re-allocate same size -- should hit cache
        let _buf2 = pool.allocate(2048);
        let stats = pool.stats();
        assert!(stats.cache_hits > 0, "Expected cache hits after reuse");
    }

    #[test]
    fn test_memory_pool_stats_tracking() {
        let pool = MemoryPool::new();

        let buf1 = pool.allocate(1000);
        let buf2 = pool.allocate(2000);

        let stats = pool.stats();
        // total_bytes_allocated always tracks requested bytes
        assert!(stats.total_bytes_allocated >= 3000,
            "Expected >= 3000 bytes allocated, got {}", stats.total_bytes_allocated);
        assert_eq!(stats.total_allocations, 2);

        pool.deallocate(buf1);
        pool.deallocate(buf2);
    }

    #[test]
    fn test_device_buffer_basic() {
        let device = Device::get_default().unwrap();
        let buf = DeviceBuffer::<u8>::new(256, device).unwrap();
        assert_eq!(buf.len(), 256);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_device_buffer_copy_roundtrip() {
        let device = Device::get_default().unwrap();
        let mut buf = DeviceBuffer::<f32>::new(4, device).unwrap();
        let src = vec![1.0f32, 2.0, 3.0, 4.0];

        buf.copy_from_host(&src).unwrap();

        let mut dst = vec![0.0f32; 4];
        buf.copy_to_host(&mut dst).unwrap();
        assert_eq!(src, dst);
    }

    #[test]
    fn test_host_buffer_basic() {
        let buf = HostBuffer::<u8>::new(128).unwrap();
        assert_eq!(buf.len(), 128);
    }

    #[test]
    fn test_host_buffer_fill_and_read() {
        let mut buf = HostBuffer::<f64>::new(100).unwrap();
        buf.fill(3.14);

        let slice = buf.as_slice();
        for &val in slice {
            assert_eq!(val, 3.14);
        }
    }

    #[test]
    fn test_concurrent_pool_allocation() {
        let pool = Arc::new(MemoryPool::new());

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    for j in 0..10 {
                        let size = 100 * (i + 1) + j;
                        let buf = pool.allocate(size);
                        assert_eq!(buf.len(), size);
                        pool.deallocate(buf);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let stats = pool.stats();
        assert!(stats.total_allocations >= 80);
    }

    #[test]
    fn test_zero_size_allocation() {
        let pool = MemoryPool::new();
        let buf = pool.allocate(0);
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_large_allocation() {
        let pool = MemoryPool::new();
        let buf = pool.allocate(10 * 1024 * 1024); // 10 MB
        assert_eq!(buf.len(), 10 * 1024 * 1024);
        pool.deallocate(buf);
    }

    #[test]
    fn test_allocation_deallocation_cycle() {
        let pool = MemoryPool::new();

        for _ in 0..100 {
            let buf = pool.allocate(1024);
            pool.deallocate(buf);
        }

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 100);
    }
}
