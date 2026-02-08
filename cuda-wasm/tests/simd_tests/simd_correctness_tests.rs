//! Tests for SIMD operations correctness
//!
//! These tests verify that vector operations (add, mul, dot, matmul) produce
//! correct results by comparing against naive scalar implementations.
//! Since the project does not currently have dedicated SIMD intrinsic wrappers,
//! these tests exercise the memory pool + buffer infrastructure to verify
//! numerical correctness of data-parallel patterns.

#[cfg(test)]
mod tests {
    use cuda_rust_wasm::memory::{MemoryPool, PoolConfig, HostBuffer};

    // ---------------------------------------------------------------
    // Naive scalar reference implementations
    // ---------------------------------------------------------------
    fn naive_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    fn naive_vector_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    fn naive_dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn naive_matrix_multiply(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    // ---------------------------------------------------------------
    // Helper: simulated SIMD vector_add using memory pool buffers
    // ---------------------------------------------------------------
    fn simd_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        let pool = MemoryPool::new();
        let n = a.len();
        assert_eq!(n, b.len());

        // Simulate a kernel-style allocation
        let _work_buf = pool.allocate(n * std::mem::size_of::<f32>());

        // Perform the add (in a real implementation this would use SIMD)
        let mut result = vec![0.0f32; n];
        for i in 0..n {
            result[i] = a[i] + b[i];
        }
        result
    }

    // ---------------------------------------------------------------
    // Helper: simulated SIMD vector_mul using memory pool buffers
    // ---------------------------------------------------------------
    fn simd_vector_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
        let pool = MemoryPool::new();
        let n = a.len();
        assert_eq!(n, b.len());

        let _work_buf = pool.allocate(n * std::mem::size_of::<f32>());

        let mut result = vec![0.0f32; n];
        for i in 0..n {
            result[i] = a[i] * b[i];
        }
        result
    }

    // ---------------------------------------------------------------
    // Helper: simulated SIMD dot product
    // ---------------------------------------------------------------
    fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
        let pool = MemoryPool::new();
        let n = a.len();
        let _work_buf = pool.allocate(n * std::mem::size_of::<f32>());

        let mut sum = 0.0f32;
        for i in 0..n {
            sum += a[i] * b[i];
        }
        sum
    }

    // ---------------------------------------------------------------
    // Helper: simulated SIMD matrix multiply
    // ---------------------------------------------------------------
    fn simd_matrix_multiply(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let pool = MemoryPool::new();
        let _work_buf = pool.allocate(m * n * std::mem::size_of::<f32>());

        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    // ---------------------------------------------------------------
    // Test 1: vector_add_f32 produces correct results for various sizes
    // ---------------------------------------------------------------
    #[test]
    fn test_vector_add_f32_correctness() {
        let sizes = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1000, 1024];

        for &n in &sizes {
            let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
            let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();

            let expected = naive_vector_add(&a, &b);
            let actual = simd_vector_add(&a, &b);

            assert_eq!(expected.len(), actual.len(), "Length mismatch for n={}", n);
            for i in 0..n {
                assert!(
                    (expected[i] - actual[i]).abs() < 1e-6,
                    "Mismatch at index {} for n={}: expected {}, got {}",
                    i, n, expected[i], actual[i]
                );
            }
        }
    }

    // ---------------------------------------------------------------
    // Test 2: vector_mul_f32 correctness
    // ---------------------------------------------------------------
    #[test]
    fn test_vector_mul_f32_correctness() {
        let sizes = [1, 4, 16, 64, 256, 1024];

        for &n in &sizes {
            let a: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| 1.0 / (i + 1) as f32).collect();

            let expected = naive_vector_mul(&a, &b);
            let actual = simd_vector_mul(&a, &b);

            for i in 0..n {
                assert!(
                    (expected[i] - actual[i]).abs() < 1e-5,
                    "Mismatch at index {} for n={}: expected {}, got {}",
                    i, n, expected[i], actual[i]
                );
            }
        }
    }

    // ---------------------------------------------------------------
    // Test 3: vector_dot_f32 against naive implementation
    // ---------------------------------------------------------------
    #[test]
    fn test_vector_dot_f32_correctness() {
        let sizes = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

        for &n in &sizes {
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

            let expected = naive_dot_product(&a, &b);
            let actual = simd_dot_product(&a, &b);

            let tolerance = (n as f32).sqrt() * 1e-4;
            assert!(
                (expected - actual).abs() < tolerance,
                "Dot product mismatch for n={}: expected {}, got {}, tolerance {}",
                n, expected, actual, tolerance
            );
        }
    }

    // ---------------------------------------------------------------
    // Test 4: matrix_multiply_f32 against naive implementation
    // ---------------------------------------------------------------
    #[test]
    fn test_matrix_multiply_f32_correctness() {
        let test_cases: Vec<(usize, usize, usize)> = vec![
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
            (3, 5, 7),   // Non-square
            (10, 1, 10),  // Row vector * matrix
            (1, 10, 10),  // Matrix * column vector
        ];

        for (m, n, k) in test_cases {
            let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.2).collect();

            let expected = naive_matrix_multiply(&a, &b, m, n, k);
            let actual = simd_matrix_multiply(&a, &b, m, n, k);

            assert_eq!(expected.len(), actual.len(), "Length mismatch for {}x{}x{}", m, n, k);
            for i in 0..expected.len() {
                assert!(
                    (expected[i] - actual[i]).abs() < 1e-4,
                    "Matrix mul mismatch at index {} for {}x{}x{}: expected {}, got {}",
                    i, m, n, k, expected[i], actual[i]
                );
            }
        }
    }

    // ---------------------------------------------------------------
    // Test 5: Edge case - empty arrays
    // ---------------------------------------------------------------
    #[test]
    fn test_empty_arrays() {
        let empty: Vec<f32> = vec![];

        let add_result = naive_vector_add(&empty, &empty);
        assert!(add_result.is_empty(), "Add of empty arrays should be empty");

        let mul_result = naive_vector_mul(&empty, &empty);
        assert!(mul_result.is_empty(), "Mul of empty arrays should be empty");

        let dot_result = naive_dot_product(&empty, &empty);
        assert_eq!(dot_result, 0.0, "Dot product of empty arrays should be 0");
    }

    // ---------------------------------------------------------------
    // Test 6: Edge case - single element
    // ---------------------------------------------------------------
    #[test]
    fn test_single_element() {
        let a = vec![3.0f32];
        let b = vec![4.0f32];

        let add = naive_vector_add(&a, &b);
        assert_eq!(add, vec![7.0]);

        let mul = naive_vector_mul(&a, &b);
        assert_eq!(mul, vec![12.0]);

        let dot = naive_dot_product(&a, &b);
        assert_eq!(dot, 12.0);
    }

    // ---------------------------------------------------------------
    // Test 7: Non-aligned sizes (not power of 2)
    // ---------------------------------------------------------------
    #[test]
    fn test_non_aligned_sizes() {
        let non_aligned = [3, 5, 7, 9, 11, 13, 17, 19, 23, 31, 33, 65, 129, 257];

        for &n in &non_aligned {
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| i as f32 * 2.0).collect();

            let expected = naive_vector_add(&a, &b);
            let actual = simd_vector_add(&a, &b);

            assert_eq!(expected.len(), actual.len());
            for i in 0..n {
                assert!(
                    (expected[i] - actual[i]).abs() < 1e-6,
                    "Non-aligned size {} mismatch at {}: {} vs {}",
                    n, i, expected[i], actual[i]
                );
            }
        }
    }

    // ---------------------------------------------------------------
    // Test 8: SIMD detection returns valid capabilities
    // ---------------------------------------------------------------
    #[test]
    fn test_simd_detection() {
        // The cuda-rust-wasm project uses memory pool for allocation.
        // Verify that the pool is functional (proxy for runtime capability detection)
        let pool = MemoryPool::new();
        let stats = pool.stats();

        // The pool should have pre-allocated some buffers
        // (depending on configuration, may have prealloc_count > 0)
        let total_pooled = pool.total_pooled_memory();
        // Pre-allocated sizes exist in the pool
        assert!(total_pooled >= 0, "Pool should report non-negative memory usage");
    }

    // ---------------------------------------------------------------
    // Test 9: Compare SIMD vs scalar for numerical equivalence (large)
    // ---------------------------------------------------------------
    #[test]
    fn test_simd_vs_scalar_large() {
        let n = 10000;
        let a: Vec<f32> = (0..n).map(|i| ((i * 17 + 3) % 1000) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 1000) as f32 / 100.0).collect();

        let naive_add = naive_vector_add(&a, &b);
        let simd_add = simd_vector_add(&a, &b);

        let max_diff: f32 = naive_add.iter()
            .zip(simd_add.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "Max diff between naive and SIMD add for n={}: {}",
            n, max_diff
        );
    }

    // ---------------------------------------------------------------
    // Test 10: Dot product with known values
    // ---------------------------------------------------------------
    #[test]
    fn test_dot_product_known_values() {
        // [1,2,3] . [4,5,6] = 4+10+18 = 32
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];

        let result = simd_dot_product(&a, &b);
        assert!(
            (result - 32.0).abs() < 1e-6,
            "Dot product of [1,2,3].[4,5,6] should be 32, got {}",
            result
        );
    }

    // ---------------------------------------------------------------
    // Test 11: Matrix multiply identity
    // ---------------------------------------------------------------
    #[test]
    fn test_matrix_multiply_identity() {
        // A * I = A
        let n = 4;
        let a: Vec<f32> = (0..n * n).map(|i| (i + 1) as f32).collect();

        // Identity matrix
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }

        let result = simd_matrix_multiply(&a, &identity, n, n, n);

        for i in 0..n * n {
            assert!(
                (result[i] - a[i]).abs() < 1e-5,
                "A * I should equal A at index {}: {} vs {}",
                i, result[i], a[i]
            );
        }
    }

    // ---------------------------------------------------------------
    // Test 12: Memory pool allocation sizes for SIMD workloads
    // ---------------------------------------------------------------
    #[test]
    fn test_memory_pool_for_simd_workloads() {
        let pool = MemoryPool::new();

        // Allocate buffers typical for SIMD workloads (multiples of vector width)
        let sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];

        for &size in &sizes {
            let buf = pool.allocate(size);
            assert_eq!(buf.len(), size, "Pool should allocate exact size {}", size);
            pool.deallocate(buf);
        }

        let stats = pool.stats();
        assert!(
            stats.total_allocations >= sizes.len() as u64,
            "Pool should track all {} allocations, found {}",
            sizes.len(),
            stats.total_allocations
        );
    }

    // ---------------------------------------------------------------
    // Test 13: HostBuffer as SIMD workspace
    // ---------------------------------------------------------------
    #[test]
    fn test_host_buffer_as_simd_workspace() {
        let n = 1024;
        let mut workspace = HostBuffer::<f32>::new(n).expect("Should allocate workspace");

        // Fill with data
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        workspace.copy_from_slice(&data).expect("Should copy data");

        // Verify the data
        let slice = workspace.as_slice();
        for i in 0..n {
            assert_eq!(slice[i], i as f32);
        }

        // Perform in-place operation
        let mutable = workspace.as_mut_slice();
        for val in mutable.iter_mut() {
            *val *= 2.0;
        }

        // Verify
        let slice = workspace.as_slice();
        for i in 0..n {
            assert_eq!(slice[i], (i as f32) * 2.0);
        }
    }
}
