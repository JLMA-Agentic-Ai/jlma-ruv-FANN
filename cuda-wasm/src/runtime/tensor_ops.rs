//! Tensor Core / Matrix Multiply-Accumulate (MMA) operations
//!
//! Emulates NVIDIA Tensor Core operations for mixed-precision matrix
//! multiplication. On real hardware (SM 7.0+), these map to WMMA/MMA
//! PTX instructions. In CPU fallback, we provide functionally-correct
//! tiled matrix multiply with the same API semantics.
//!
//! Supports: fp16×fp16→fp32, bf16×bf16→fp32, fp32→fp32, int8×int8→int32.

use super::half::Half;
use super::bfloat16::BFloat16;
use std::fmt;

/// Precision mode for tensor core operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MmaPrecision {
    /// fp16 inputs, fp32 accumulation (HMMA)
    Fp16Fp32,
    /// bf16 inputs, fp32 accumulation
    Bf16Fp32,
    /// fp32 inputs, fp32 accumulation (TF32 on Ampere+)
    Tf32,
    /// int8 inputs, int32 accumulation (IMMA)
    Int8Int32,
    /// Full fp32 (no tensor cores, standard GEMM)
    Fp32,
}

/// Fragment shape for WMMA operations.
/// Maps to hardware-supported shapes like 16×16×16, 8×32×16, etc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FragmentShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl FragmentShape {
    /// Standard 16×16×16 (SM 7.0+ Volta)
    pub const M16N16K16: Self = Self { m: 16, n: 16, k: 16 };
    /// Ampere 16×8×16
    pub const M16N8K16: Self = Self { m: 16, n: 8, k: 16 };
    /// INT8: 8×32×16
    pub const M8N32K16: Self = Self { m: 8, n: 32, k: 16 };
    /// Custom shape
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }
}

/// Matrix fragment — a tile of a matrix stored in registers.
/// On GPU, these map to warp-distributed register fragments.
#[derive(Debug, Clone)]
pub struct Fragment {
    /// Data stored as f32 (accumulator format).
    pub data: Vec<f32>,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl Fragment {
    /// Create a zero-initialized fragment.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create from f32 data.
    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> crate::Result<Self> {
        if data.len() != rows * cols {
            return Err(crate::error::CudaRustError::RuntimeError(
                format!("Fragment size mismatch: {}×{} needs {} elements, got {}",
                    rows, cols, rows * cols, data.len()),
            ));
        }
        Ok(Self {
            data: data.to_vec(),
            rows,
            cols,
        })
    }

    /// Load from fp16 data (converting to f32 accumulator format).
    pub fn from_half(data: &[Half], rows: usize, cols: usize) -> crate::Result<Self> {
        if data.len() != rows * cols {
            return Err(crate::error::CudaRustError::RuntimeError(
                format!("Fragment size mismatch: expected {} elements, got {}", rows * cols, data.len()),
            ));
        }
        Ok(Self {
            data: data.iter().map(|h| h.to_f32()).collect(),
            rows,
            cols,
        })
    }

    /// Load from bf16 data.
    pub fn from_bf16(data: &[BFloat16], rows: usize, cols: usize) -> crate::Result<Self> {
        if data.len() != rows * cols {
            return Err(crate::error::CudaRustError::RuntimeError(
                format!("Fragment size mismatch: expected {} elements, got {}", rows * cols, data.len()),
            ));
        }
        Ok(Self {
            data: data.iter().map(|b| b.to_f32()).collect(),
            rows,
            cols,
        })
    }

    /// Get element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[row * self.cols + col] = val;
    }

    /// Store to fp16.
    pub fn to_half(&self) -> Vec<Half> {
        self.data.iter().map(|&v| Half::from_f32(v)).collect()
    }

    /// Store to bf16.
    pub fn to_bf16(&self) -> Vec<BFloat16> {
        self.data.iter().map(|&v| BFloat16::from_f32(v)).collect()
    }
}

/// Tensor Core MMA engine.
///
/// Provides `mma()` (D = A·B + C) matching the semantics of CUDA's
/// `nvcuda::wmma::mma_sync` and PTX `mma.sync` instructions.
pub struct TensorCoreEngine {
    precision: MmaPrecision,
    shape: FragmentShape,
}

impl TensorCoreEngine {
    /// Create a new engine with specified precision and fragment shape.
    pub fn new(precision: MmaPrecision, shape: FragmentShape) -> Self {
        Self { precision, shape }
    }

    /// Matrix multiply-accumulate: D = A · B + C
    ///
    /// A: (m × k), B: (k × n), C: (m × n) → D: (m × n)
    pub fn mma(&self, a: &Fragment, b: &Fragment, c: &Fragment) -> crate::Result<Fragment> {
        if a.rows != self.shape.m || a.cols != self.shape.k {
            return Err(crate::error::CudaRustError::RuntimeError(
                format!("Fragment A shape {}×{} doesn't match MMA {}×{}",
                    a.rows, a.cols, self.shape.m, self.shape.k),
            ));
        }
        if b.rows != self.shape.k || b.cols != self.shape.n {
            return Err(crate::error::CudaRustError::RuntimeError(
                format!("Fragment B shape {}×{} doesn't match MMA {}×{}",
                    b.rows, b.cols, self.shape.k, self.shape.n),
            ));
        }
        if c.rows != self.shape.m || c.cols != self.shape.n {
            return Err(crate::error::CudaRustError::RuntimeError(
                format!("Fragment C shape {}×{} doesn't match MMA {}×{}",
                    c.rows, c.cols, self.shape.m, self.shape.n),
            ));
        }

        let m = self.shape.m;
        let n = self.shape.n;
        let k = self.shape.k;

        let mut d = Fragment::zeros(m, n);

        // D = A · B + C (standard GEMM)
        for i in 0..m {
            for j in 0..n {
                let mut acc = c.get(i, j);
                for p in 0..k {
                    acc += a.get(i, p) * b.get(p, j);
                }
                d.set(i, j, acc);
            }
        }

        Ok(d)
    }

    /// Full GEMM using tiled MMA: C = alpha * A · B + beta * C
    ///
    /// A: (m × k), B: (k × n), C: (m × n) — arbitrary sizes, tiled internally.
    pub fn gemm(
        &self,
        a: &[f32], b: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize,
        alpha: f32, beta: f32,
    ) -> crate::Result<GemmStats> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(crate::error::CudaRustError::RuntimeError("GEMM dimension mismatch".into()));
        }

        let tm = self.shape.m;
        let tn = self.shape.n;
        let tk = self.shape.k;
        let mut mma_count = 0u64;

        // Scale C by beta
        for val in c.iter_mut() {
            *val *= beta;
        }

        // Tile over M, N, K
        let m_tiles = (m + tm - 1) / tm;
        let n_tiles = (n + tn - 1) / tn;
        let k_tiles = (k + tk - 1) / tk;

        for mi in 0..m_tiles {
            let m_start = mi * tm;
            let m_end = (m_start + tm).min(m);
            let actual_m = m_end - m_start;

            for ni in 0..n_tiles {
                let n_start = ni * tn;
                let n_end = (n_start + tn).min(n);
                let actual_n = n_end - n_start;

                for ki in 0..k_tiles {
                    let k_start = ki * tk;
                    let k_end = (k_start + tk).min(k);
                    let actual_k = k_end - k_start;

                    // Extract tiles
                    for i in 0..actual_m {
                        for j in 0..actual_n {
                            let mut acc = 0.0f32;
                            for p in 0..actual_k {
                                acc += a[(m_start + i) * k + (k_start + p)]
                                     * b[(k_start + p) * n + (n_start + j)];
                            }
                            c[(m_start + i) * n + (n_start + j)] += alpha * acc;
                        }
                    }
                    mma_count += 1;
                }
            }
        }

        let flops = 2 * (m as u64) * (n as u64) * (k as u64);
        Ok(GemmStats { mma_count, flops, precision: self.precision })
    }
}

/// Statistics from a GEMM operation.
#[derive(Debug, Clone)]
pub struct GemmStats {
    /// Number of MMA (tile) operations performed.
    pub mma_count: u64,
    /// Total floating-point operations.
    pub flops: u64,
    /// Precision mode used.
    pub precision: MmaPrecision,
}

impl fmt::Display for GemmStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GEMM: {} MMA ops, {:.2}M FLOPs, {:?}",
            self.mma_count, self.flops as f64 / 1e6, self.precision)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fragment_zeros() {
        let frag = Fragment::zeros(4, 4);
        assert_eq!(frag.data.len(), 16);
        assert!(frag.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_fragment_from_f32() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let frag = Fragment::from_f32(&data, 4, 4).unwrap();
        assert_eq!(frag.get(0, 0), 0.0);
        assert_eq!(frag.get(1, 2), 6.0);
        assert_eq!(frag.get(3, 3), 15.0);
    }

    #[test]
    fn test_mma_identity() {
        let engine = TensorCoreEngine::new(MmaPrecision::Fp32, FragmentShape::new(2, 2, 2));

        // A = I (identity)
        let a = Fragment::from_f32(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        // B = some matrix
        let b = Fragment::from_f32(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
        // C = zeros
        let c = Fragment::zeros(2, 2);

        let d = engine.mma(&a, &b, &c).unwrap();
        assert!((d.get(0, 0) - 5.0).abs() < 1e-6);
        assert!((d.get(0, 1) - 6.0).abs() < 1e-6);
        assert!((d.get(1, 0) - 7.0).abs() < 1e-6);
        assert!((d.get(1, 1) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_mma_accumulate() {
        let engine = TensorCoreEngine::new(MmaPrecision::Fp16Fp32, FragmentShape::new(2, 2, 2));

        let a = Fragment::from_f32(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = Fragment::from_f32(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
        let c = Fragment::from_f32(&[10.0, 10.0, 10.0, 10.0], 2, 2).unwrap();

        // D = A·B + C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] + 10
        let d = engine.mma(&a, &b, &c).unwrap();
        assert!((d.get(0, 0) - 29.0).abs() < 1e-6); // 19 + 10
        assert!((d.get(0, 1) - 32.0).abs() < 1e-6); // 22 + 10
        assert!((d.get(1, 0) - 53.0).abs() < 1e-6); // 43 + 10
        assert!((d.get(1, 1) - 60.0).abs() < 1e-6); // 50 + 10
    }

    #[test]
    fn test_mma_shape_validation() {
        let engine = TensorCoreEngine::new(MmaPrecision::Fp32, FragmentShape::new(4, 4, 4));
        let a = Fragment::zeros(2, 2); // Wrong shape
        let b = Fragment::zeros(4, 4);
        let c = Fragment::zeros(4, 4);
        assert!(engine.mma(&a, &b, &c).is_err());
    }

    #[test]
    fn test_gemm_basic() {
        let engine = TensorCoreEngine::new(MmaPrecision::Fp32, FragmentShape::new(2, 2, 2));
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2×2
        let mut c = vec![0.0; 4]; // 2×2

        let stats = engine.gemm(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0).unwrap();
        assert!((c[0] - 19.0).abs() < 1e-4); // 1*5+2*7
        assert!((c[1] - 22.0).abs() < 1e-4);
        assert!((c[2] - 43.0).abs() < 1e-4);
        assert!((c[3] - 50.0).abs() < 1e-4);
        assert_eq!(stats.flops, 16); // 2*2*2*2
    }

    #[test]
    fn test_gemm_alpha_beta() {
        let engine = TensorCoreEngine::new(MmaPrecision::Fp32, FragmentShape::new(2, 2, 2));
        let a = vec![1.0, 0.0, 0.0, 1.0]; // Identity
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![10.0, 10.0, 10.0, 10.0];

        // C = 2.0 * I * B + 0.5 * C
        engine.gemm(&a, &b, &mut c, 2, 2, 2, 2.0, 0.5).unwrap();
        assert!((c[0] - 7.0).abs() < 1e-4); // 2*1 + 0.5*10 = 7
        assert!((c[1] - 9.0).abs() < 1e-4); // 2*2 + 0.5*10 = 9
    }

    #[test]
    fn test_gemm_non_square() {
        let engine = TensorCoreEngine::new(MmaPrecision::Fp32, FragmentShape::new(2, 2, 2));
        // A: 3×2, B: 2×4 → C: 3×4
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 12];

        engine.gemm(&a, &b, &mut c, 3, 4, 2, 1.0, 0.0).unwrap();
        // Row 0: [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
        assert!((c[0] - 11.0).abs() < 1e-4);
        assert!((c[1] - 14.0).abs() < 1e-4);
        assert!((c[2] - 17.0).abs() < 1e-4);
        assert!((c[3] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn test_fragment_half_roundtrip() {
        let data = vec![Half::from_f32(1.0), Half::from_f32(2.0), Half::from_f32(3.0), Half::from_f32(4.0)];
        let frag = Fragment::from_half(&data, 2, 2).unwrap();
        let back = frag.to_half();
        for i in 0..4 {
            assert!((back[i].to_f32() - data[i].to_f32()).abs() < 0.01);
        }
    }

    #[test]
    fn test_fragment_bf16_roundtrip() {
        let data = vec![BFloat16::from_f32(1.5), BFloat16::from_f32(2.5)];
        let frag = Fragment::from_bf16(&data, 1, 2).unwrap();
        let back = frag.to_bf16();
        assert!((back[0].to_f32() - 1.5).abs() < 0.1);
        assert!((back[1].to_f32() - 2.5).abs() < 0.1);
    }

    #[test]
    fn test_gemm_stats_display() {
        let stats = GemmStats { mma_count: 64, flops: 1_000_000, precision: MmaPrecision::Fp16Fp32 };
        let s = format!("{}", stats);
        assert!(s.contains("64 MMA"));
        assert!(s.contains("Fp16Fp32"));
    }
}
