//! Runtime SIMD feature detection
//!
//! Detects available SIMD instruction sets at runtime and returns a capabilities
//! struct that can be queried to select optimal code paths.

use std::fmt;

/// Available SIMD instruction set levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdLevel {
    /// No SIMD support, scalar fallback only
    Scalar,
    /// SSE2 (128-bit, x86_64 baseline)
    Sse2,
    /// SSE4.1 (128-bit, enhanced integer ops)
    Sse41,
    /// AVX2 (256-bit, integer + float)
    Avx2,
    /// AVX-512 Foundation (512-bit)
    Avx512,
    /// ARM NEON (128-bit)
    Neon,
    /// ARM SVE (scalable vector extension)
    Sve,
    /// WebAssembly SIMD (128-bit)
    WasmSimd128,
}

impl fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimdLevel::Scalar => write!(f, "Scalar"),
            SimdLevel::Sse2 => write!(f, "SSE2"),
            SimdLevel::Sse41 => write!(f, "SSE4.1"),
            SimdLevel::Avx2 => write!(f, "AVX2"),
            SimdLevel::Avx512 => write!(f, "AVX-512"),
            SimdLevel::Neon => write!(f, "NEON"),
            SimdLevel::Sve => write!(f, "SVE"),
            SimdLevel::WasmSimd128 => write!(f, "WASM SIMD128"),
        }
    }
}

/// Runtime SIMD capabilities of the current platform
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    /// Whether SSE2 is available (x86_64 baseline, always true on x86_64)
    pub has_sse2: bool,
    /// Whether SSE4.1 is available
    pub has_sse41: bool,
    /// Whether AVX2 is available (256-bit integer + float)
    pub has_avx2: bool,
    /// Whether AVX-512 Foundation is available
    pub has_avx512f: bool,
    /// Whether FMA (fused multiply-add) is available
    pub has_fma: bool,
    /// Whether ARM NEON is available
    pub has_neon: bool,
    /// Whether ARM SVE is available
    pub has_sve: bool,
    /// Whether WASM SIMD128 is available
    pub has_wasm_simd128: bool,
    /// The maximum vector width in bytes supported by the platform
    pub max_vector_width_bytes: usize,
    /// The best available SIMD level
    pub best_level: SimdLevel,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities at runtime for the current platform.
    pub fn detect() -> Self {
        let mut caps = SimdCapabilities {
            has_sse2: false,
            has_sse41: false,
            has_avx2: false,
            has_avx512f: false,
            has_fma: false,
            has_neon: false,
            has_sve: false,
            has_wasm_simd128: false,
            max_vector_width_bytes: 0,
            best_level: SimdLevel::Scalar,
        };

        #[cfg(target_arch = "x86_64")]
        {
            caps.detect_x86_64();
        }

        #[cfg(target_arch = "aarch64")]
        {
            caps.detect_aarch64();
        }

        #[cfg(target_arch = "wasm32")]
        {
            caps.detect_wasm();
        }

        // If nothing was detected, we still have scalar
        if caps.max_vector_width_bytes == 0 {
            // Scalar: process one f32 at a time
            caps.max_vector_width_bytes = 4;
        }

        caps
    }

    /// Detect x86_64 SIMD features using `is_x86_feature_detected!`
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64(&mut self) {
        // SSE2 is always available on x86_64
        self.has_sse2 = true;
        self.best_level = SimdLevel::Sse2;
        self.max_vector_width_bytes = 16; // 128-bit

        if is_x86_feature_detected!("sse4.1") {
            self.has_sse41 = true;
            self.best_level = SimdLevel::Sse41;
        }

        if is_x86_feature_detected!("fma") {
            self.has_fma = true;
        }

        if is_x86_feature_detected!("avx2") {
            self.has_avx2 = true;
            self.best_level = SimdLevel::Avx2;
            self.max_vector_width_bytes = 32; // 256-bit
        }

        if is_x86_feature_detected!("avx512f") {
            self.has_avx512f = true;
            self.best_level = SimdLevel::Avx512;
            self.max_vector_width_bytes = 64; // 512-bit
        }
    }

    /// Detect aarch64 SIMD features
    #[cfg(target_arch = "aarch64")]
    fn detect_aarch64(&mut self) {
        // NEON is mandatory on aarch64
        self.has_neon = true;
        self.best_level = SimdLevel::Neon;
        self.max_vector_width_bytes = 16; // 128-bit

        // SVE detection via std::arch feature detection
        #[cfg(target_feature = "sve")]
        {
            self.has_sve = true;
            self.best_level = SimdLevel::Sve;
            // SVE vector length is implementation-defined (128-2048 bits)
            // Use a conservative estimate; actual length queried at runtime would
            // require inline assembly (cntb instruction).
            self.max_vector_width_bytes = 32; // conservative 256-bit estimate
        }
    }

    /// Detect WebAssembly SIMD features
    #[cfg(target_arch = "wasm32")]
    fn detect_wasm(&mut self) {
        #[cfg(target_feature = "simd128")]
        {
            self.has_wasm_simd128 = true;
            self.best_level = SimdLevel::WasmSimd128;
            self.max_vector_width_bytes = 16; // 128-bit
        }
    }

    /// Returns the number of f32 elements that can be processed in a single
    /// SIMD operation.
    pub fn f32_lane_count(&self) -> usize {
        self.max_vector_width_bytes / std::mem::size_of::<f32>()
    }

    /// Returns true if any SIMD acceleration is available beyond scalar.
    pub fn has_simd(&self) -> bool {
        self.best_level != SimdLevel::Scalar
    }

    /// Returns a human-readable summary of detected capabilities.
    pub fn summary(&self) -> String {
        let mut features = Vec::new();

        if self.has_sse2 {
            features.push("SSE2");
        }
        if self.has_sse41 {
            features.push("SSE4.1");
        }
        if self.has_avx2 {
            features.push("AVX2");
        }
        if self.has_avx512f {
            features.push("AVX-512F");
        }
        if self.has_fma {
            features.push("FMA");
        }
        if self.has_neon {
            features.push("NEON");
        }
        if self.has_sve {
            features.push("SVE");
        }
        if self.has_wasm_simd128 {
            features.push("WASM SIMD128");
        }

        if features.is_empty() {
            "Scalar only (no SIMD)".to_string()
        } else {
            format!(
                "Best: {} | Features: {} | Vector width: {} bytes ({} f32 lanes)",
                self.best_level,
                features.join(", "),
                self.max_vector_width_bytes,
                self.f32_lane_count()
            )
        }
    }
}

impl fmt::Display for SimdCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_returns_valid_capabilities() {
        let caps = SimdCapabilities::detect();
        // Should always have at least scalar width
        assert!(caps.max_vector_width_bytes >= 4);
        assert!(caps.f32_lane_count() >= 1);
    }

    #[test]
    fn test_summary_not_empty() {
        let caps = SimdCapabilities::detect();
        let summary = caps.summary();
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_display_impl() {
        let caps = SimdCapabilities::detect();
        let display = format!("{caps}");
        assert!(!display.is_empty());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_x86_64_has_sse2() {
        let caps = SimdCapabilities::detect();
        // SSE2 is mandatory on x86_64
        assert!(caps.has_sse2);
        assert!(caps.has_simd());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_aarch64_has_neon() {
        let caps = SimdCapabilities::detect();
        // NEON is mandatory on aarch64
        assert!(caps.has_neon);
        assert!(caps.has_simd());
    }
}
