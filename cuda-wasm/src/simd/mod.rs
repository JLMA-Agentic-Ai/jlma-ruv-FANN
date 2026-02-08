//! SIMD acceleration layer with runtime feature detection
//!
//! Provides optimized vector and matrix operations using platform-specific SIMD
//! instructions (AVX2/AVX-512 on x86_64, NEON/SVE on aarch64) with automatic
//! scalar fallback for unsupported architectures.

pub mod detection;
pub mod vector_ops;
pub mod matrix_ops;

pub use detection::SimdCapabilities;
