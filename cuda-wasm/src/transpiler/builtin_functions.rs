//! CUDA built-in function mapping to Rust and WGSL equivalents
//!
//! Maps CUDA intrinsic functions (math, atomic, warp, sync, and type conversion)
//! to their corresponding representations in Rust (for CPU fallback) and WGSL
//! (for WebGPU compute shaders).

/// Maps CUDA built-in function calls to Rust or WGSL equivalents.
pub struct BuiltinMapper;

impl BuiltinMapper {
    // -----------------------------------------------------------------------
    // Rust target mapping
    // -----------------------------------------------------------------------

    /// Map a CUDA built-in function call to its Rust equivalent.
    ///
    /// Returns `Some(rust_code)` if the function is a recognized CUDA builtin,
    /// or `None` if it is not a builtin and should be treated as a user function.
    pub fn map_to_rust(name: &str, args: &[String]) -> Option<String> {
        // Math functions
        if let Some(mapped) = Self::map_math_to_rust(name, args) {
            return Some(mapped);
        }

        // Atomic operations
        if let Some(mapped) = Self::map_atomic_to_rust(name, args) {
            return Some(mapped);
        }

        // Warp-level primitives
        if let Some(mapped) = Self::map_warp_to_rust(name, args) {
            return Some(mapped);
        }

        // Synchronization
        if let Some(mapped) = Self::map_sync_to_rust(name, args) {
            return Some(mapped);
        }

        // Type conversion
        if let Some(mapped) = Self::map_type_conversion_to_rust(name, args) {
            return Some(mapped);
        }

        None
    }

    // -----------------------------------------------------------------------
    // WGSL target mapping
    // -----------------------------------------------------------------------

    /// Map a CUDA built-in function call to its WGSL equivalent.
    ///
    /// Returns `Some(wgsl_code)` if the function is a recognized CUDA builtin,
    /// or `None` if it is not a builtin.
    pub fn map_to_wgsl(name: &str, args: &[String]) -> Option<String> {
        // Math functions
        if let Some(mapped) = Self::map_math_to_wgsl(name, args) {
            return Some(mapped);
        }

        // Atomic operations
        if let Some(mapped) = Self::map_atomic_to_wgsl(name, args) {
            return Some(mapped);
        }

        // Warp-level primitives
        if let Some(mapped) = Self::map_warp_to_wgsl(name, args) {
            return Some(mapped);
        }

        // Synchronization
        if let Some(mapped) = Self::map_sync_to_wgsl(name, args) {
            return Some(mapped);
        }

        // Type conversion
        if let Some(mapped) = Self::map_type_conversion_to_wgsl(name, args) {
            return Some(mapped);
        }

        None
    }

    // -----------------------------------------------------------------------
    // Math functions -> Rust
    // -----------------------------------------------------------------------
    fn map_math_to_rust(name: &str, args: &[String]) -> Option<String> {
        match name {
            // Fast single-precision math intrinsics -> standard Rust f32 methods
            "__sinf" | "sinf" => {
                let a = args.first()?;
                Some(format!("({a} as f32).sin()"))
            }
            "__cosf" | "cosf" => {
                let a = args.first()?;
                Some(format!("({a} as f32).cos()"))
            }
            "__expf" | "expf" => {
                let a = args.first()?;
                Some(format!("({a} as f32).exp()"))
            }
            "__logf" | "logf" => {
                let a = args.first()?;
                Some(format!("({a} as f32).ln()"))
            }
            "__powf" | "powf" => {
                let base = args.first()?;
                let exp = args.get(1)?;
                Some(format!("({base} as f32).powf({exp} as f32)"))
            }
            "sqrtf" | "__fsqrt_rn" => {
                let a = args.first()?;
                Some(format!("({a} as f32).sqrt()"))
            }
            "fabsf" | "__fabsf" => {
                let a = args.first()?;
                Some(format!("({a} as f32).abs()"))
            }
            "fminf" => {
                let a = args.first()?;
                let b = args.get(1)?;
                Some(format!("({a} as f32).min({b} as f32)"))
            }
            "fmaxf" => {
                let a = args.first()?;
                let b = args.get(1)?;
                Some(format!("({a} as f32).max({b} as f32)"))
            }
            "ceilf" | "__ceilf" => {
                let a = args.first()?;
                Some(format!("({a} as f32).ceil()"))
            }
            "floorf" | "__floorf" => {
                let a = args.first()?;
                Some(format!("({a} as f32).floor()"))
            }
            "roundf" | "__roundf" => {
                let a = args.first()?;
                Some(format!("({a} as f32).round()"))
            }
            // Double-precision variants
            "sin" => {
                let a = args.first()?;
                Some(format!("({a} as f64).sin()"))
            }
            "cos" => {
                let a = args.first()?;
                Some(format!("({a} as f64).cos()"))
            }
            "exp" => {
                let a = args.first()?;
                Some(format!("({a} as f64).exp()"))
            }
            "log" => {
                let a = args.first()?;
                Some(format!("({a} as f64).ln()"))
            }
            "sqrt" => {
                let a = args.first()?;
                Some(format!("({a} as f64).sqrt()"))
            }
            "fabs" => {
                let a = args.first()?;
                Some(format!("({a} as f64).abs()"))
            }
            "fmin" => {
                let a = args.first()?;
                let b = args.get(1)?;
                Some(format!("({a} as f64).min({b} as f64)"))
            }
            "fmax" => {
                let a = args.first()?;
                let b = args.get(1)?;
                Some(format!("({a} as f64).max({b} as f64)"))
            }
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Atomic operations -> Rust
    // -----------------------------------------------------------------------
    fn map_atomic_to_rust(name: &str, args: &[String]) -> Option<String> {
        // CUDA atomics: first arg is pointer to memory location, second is value
        let addr = args.first()?;
        let val = args.get(1);

        match name {
            "atomicAdd" => {
                let v = val?;
                Some(format!(
                    "{{ let ptr = {addr} as *mut _ as *mut std::sync::atomic::AtomicI32; \
                     unsafe {{ (*ptr).fetch_add({v} as i32, std::sync::atomic::Ordering::Relaxed) }} }}"
                ))
            }
            "atomicSub" => {
                let v = val?;
                Some(format!(
                    "{{ let ptr = {addr} as *mut _ as *mut std::sync::atomic::AtomicI32; \
                     unsafe {{ (*ptr).fetch_sub({v} as i32, std::sync::atomic::Ordering::Relaxed) }} }}"
                ))
            }
            "atomicMin" => {
                let v = val?;
                Some(format!(
                    "{{ let ptr = {addr} as *mut _ as *mut std::sync::atomic::AtomicI32; \
                     unsafe {{ (*ptr).fetch_min({v} as i32, std::sync::atomic::Ordering::Relaxed) }} }}"
                ))
            }
            "atomicMax" => {
                let v = val?;
                Some(format!(
                    "{{ let ptr = {addr} as *mut _ as *mut std::sync::atomic::AtomicI32; \
                     unsafe {{ (*ptr).fetch_max({v} as i32, std::sync::atomic::Ordering::Relaxed) }} }}"
                ))
            }
            "atomicExch" => {
                let v = val?;
                Some(format!(
                    "{{ let ptr = {addr} as *mut _ as *mut std::sync::atomic::AtomicI32; \
                     unsafe {{ (*ptr).swap({v} as i32, std::sync::atomic::Ordering::Relaxed) }} }}"
                ))
            }
            "atomicCAS" => {
                // atomicCAS(addr, compare, val) -> old
                let compare = args.get(1)?;
                let v = args.get(2)?;
                Some(format!(
                    "{{ let ptr = {addr} as *mut _ as *mut std::sync::atomic::AtomicI32; \
                     unsafe {{ (*ptr).compare_exchange(\
                     {compare} as i32, {v} as i32, \
                     std::sync::atomic::Ordering::Relaxed, \
                     std::sync::atomic::Ordering::Relaxed\
                     ).unwrap_or_else(|old| old) }} }}"
                ))
            }
            "atomicAnd" => {
                let v = val?;
                Some(format!(
                    "{{ let ptr = {addr} as *mut _ as *mut std::sync::atomic::AtomicI32; \
                     unsafe {{ (*ptr).fetch_and({v} as i32, std::sync::atomic::Ordering::Relaxed) }} }}"
                ))
            }
            "atomicOr" => {
                let v = val?;
                Some(format!(
                    "{{ let ptr = {addr} as *mut _ as *mut std::sync::atomic::AtomicI32; \
                     unsafe {{ (*ptr).fetch_or({v} as i32, std::sync::atomic::Ordering::Relaxed) }} }}"
                ))
            }
            "atomicXor" => {
                let v = val?;
                Some(format!(
                    "{{ let ptr = {addr} as *mut _ as *mut std::sync::atomic::AtomicI32; \
                     unsafe {{ (*ptr).fetch_xor({v} as i32, std::sync::atomic::Ordering::Relaxed) }} }}"
                ))
            }
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Warp-level primitives -> Rust
    // -----------------------------------------------------------------------
    fn map_warp_to_rust(name: &str, args: &[String]) -> Option<String> {
        match name {
            "__shfl_sync" => {
                // __shfl_sync(mask, var, srcLane, width=32)
                let _mask = args.first()?;
                let var = args.get(1)?;
                let src_lane = args.get(2)?;
                Some(format!(
                    "cuda_rust_wasm::kernel::warp::WarpState::shuffle({var}, {src_lane} as u32)"
                ))
            }
            "__shfl_xor_sync" => {
                let _mask = args.first()?;
                let var = args.get(1)?;
                let lane_mask = args.get(2)?;
                Some(format!(
                    "cuda_rust_wasm::kernel::warp::WarpState::shuffle_xor({var}, {lane_mask} as u32)"
                ))
            }
            "__shfl_up_sync" => {
                let _mask = args.first()?;
                let var = args.get(1)?;
                let delta = args.get(2)?;
                Some(format!(
                    "cuda_rust_wasm::kernel::warp::WarpState::shuffle_up({var}, {delta} as u32)"
                ))
            }
            "__shfl_down_sync" => {
                let _mask = args.first()?;
                let var = args.get(1)?;
                let delta = args.get(2)?;
                Some(format!(
                    "cuda_rust_wasm::kernel::warp::WarpState::shuffle_down({var}, {delta} as u32)"
                ))
            }
            "__ballot_sync" => {
                let _mask = args.first()?;
                let predicate = args.get(1)?;
                Some(format!(
                    "cuda_rust_wasm::kernel::warp::WarpState::ballot({predicate})"
                ))
            }
            "__all_sync" => {
                let _mask = args.first()?;
                let predicate = args.get(1)?;
                Some(format!(
                    "cuda_rust_wasm::kernel::warp::WarpState::vote_all({predicate})"
                ))
            }
            "__any_sync" => {
                let _mask = args.first()?;
                let predicate = args.get(1)?;
                Some(format!(
                    "cuda_rust_wasm::kernel::warp::WarpState::vote_any({predicate})"
                ))
            }
            "__activemask" => {
                Some("cuda_rust_wasm::kernel::warp::WarpState::active_mask()".to_string())
            }
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Synchronization -> Rust
    // -----------------------------------------------------------------------
    fn map_sync_to_rust(name: &str, _args: &[String]) -> Option<String> {
        match name {
            "__syncthreads" => {
                Some("cuda_rust_wasm::runtime::sync_threads()".to_string())
            }
            "__threadfence" => {
                Some("std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst)".to_string())
            }
            "__threadfence_block" => {
                Some("std::sync::atomic::fence(std::sync::atomic::Ordering::AcqRel)".to_string())
            }
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Type conversion -> Rust
    // -----------------------------------------------------------------------
    fn map_type_conversion_to_rust(name: &str, args: &[String]) -> Option<String> {
        let a = args.first()?;
        match name {
            "__float2int_rn" => Some(format!("({a} as f32).round() as i32")),
            "__int2float_rn" => Some(format!("({a} as i32) as f32")),
            "__float2half" => {
                // Rust does not have native f16; truncate to f32 and note this
                // would require the `half` crate for true f16 support.
                Some(format!("({a} as f32) /* f16 requires half crate */"))
            }
            "__half2float" => {
                Some(format!("({a} as f32) /* from f16 */"))
            }
            "__float2uint_rn" => Some(format!("({a} as f32).round() as u32")),
            "__uint2float_rn" => Some(format!("({a} as u32) as f32")),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Math functions -> WGSL
    // -----------------------------------------------------------------------
    fn map_math_to_wgsl(name: &str, args: &[String]) -> Option<String> {
        match name {
            "__sinf" | "sinf" | "sin" => {
                let a = args.first()?;
                Some(format!("sin({a})"))
            }
            "__cosf" | "cosf" | "cos" => {
                let a = args.first()?;
                Some(format!("cos({a})"))
            }
            "__expf" | "expf" | "exp" => {
                let a = args.first()?;
                Some(format!("exp({a})"))
            }
            "__logf" | "logf" | "log" => {
                let a = args.first()?;
                Some(format!("log({a})"))
            }
            "__powf" | "powf" | "pow" => {
                let base = args.first()?;
                let exp = args.get(1)?;
                Some(format!("pow({base}, {exp})"))
            }
            "sqrtf" | "sqrt" | "__fsqrt_rn" => {
                let a = args.first()?;
                Some(format!("sqrt({a})"))
            }
            "fabsf" | "fabs" | "__fabsf" => {
                let a = args.first()?;
                Some(format!("abs({a})"))
            }
            "fminf" | "fmin" => {
                let a = args.first()?;
                let b = args.get(1)?;
                Some(format!("min({a}, {b})"))
            }
            "fmaxf" | "fmax" => {
                let a = args.first()?;
                let b = args.get(1)?;
                Some(format!("max({a}, {b})"))
            }
            "ceilf" | "ceil" | "__ceilf" => {
                let a = args.first()?;
                Some(format!("ceil({a})"))
            }
            "floorf" | "floor" | "__floorf" => {
                let a = args.first()?;
                Some(format!("floor({a})"))
            }
            "roundf" | "round" | "__roundf" => {
                let a = args.first()?;
                Some(format!("round({a})"))
            }
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Atomic operations -> WGSL
    // -----------------------------------------------------------------------
    fn map_atomic_to_wgsl(name: &str, args: &[String]) -> Option<String> {
        let addr = args.first()?;
        let val = args.get(1);

        match name {
            "atomicAdd" => {
                let v = val?;
                Some(format!("atomicAdd(&{addr}, {v})"))
            }
            "atomicSub" => {
                let v = val?;
                Some(format!("atomicSub(&{addr}, {v})"))
            }
            "atomicMin" => {
                let v = val?;
                Some(format!("atomicMin(&{addr}, {v})"))
            }
            "atomicMax" => {
                let v = val?;
                Some(format!("atomicMax(&{addr}, {v})"))
            }
            "atomicExch" => {
                let v = val?;
                Some(format!("atomicExchange(&{addr}, {v})"))
            }
            "atomicCAS" => {
                let compare = args.get(1)?;
                let v = args.get(2)?;
                Some(format!("atomicCompareExchangeWeak(&{addr}, {compare}, {v}).old_value"))
            }
            "atomicAnd" => {
                let v = val?;
                Some(format!("atomicAnd(&{addr}, {v})"))
            }
            "atomicOr" => {
                let v = val?;
                Some(format!("atomicOr(&{addr}, {v})"))
            }
            "atomicXor" => {
                let v = val?;
                Some(format!("atomicXor(&{addr}, {v})"))
            }
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Warp-level primitives -> WGSL
    // -----------------------------------------------------------------------
    fn map_warp_to_wgsl(name: &str, args: &[String]) -> Option<String> {
        // WGSL does not have direct warp/subgroup primitives in all implementations.
        // We emit subgroup operations where available (WGSL extensions) and
        // fall back to comments + placeholders otherwise.
        match name {
            "__shfl_sync" => {
                let _mask = args.first()?;
                let var = args.get(1)?;
                let src_lane = args.get(2)?;
                // WGSL subgroup_shuffle is an extension (not universally available)
                Some(format!(
                    "/* __shfl_sync */ subgroupShuffle({var}, {src_lane})"
                ))
            }
            "__shfl_xor_sync" => {
                let _mask = args.first()?;
                let var = args.get(1)?;
                let lane_mask = args.get(2)?;
                Some(format!(
                    "/* __shfl_xor_sync */ subgroupShuffleXor({var}, {lane_mask})"
                ))
            }
            "__shfl_up_sync" => {
                let _mask = args.first()?;
                let var = args.get(1)?;
                let delta = args.get(2)?;
                Some(format!(
                    "/* __shfl_up_sync */ subgroupShuffleUp({var}, {delta})"
                ))
            }
            "__shfl_down_sync" => {
                let _mask = args.first()?;
                let var = args.get(1)?;
                let delta = args.get(2)?;
                Some(format!(
                    "/* __shfl_down_sync */ subgroupShuffleDown({var}, {delta})"
                ))
            }
            "__ballot_sync" => {
                let _mask = args.first()?;
                let predicate = args.get(1)?;
                Some(format!(
                    "/* __ballot_sync */ subgroupBallot({predicate})"
                ))
            }
            "__all_sync" => {
                let _mask = args.first()?;
                let predicate = args.get(1)?;
                Some(format!(
                    "/* __all_sync */ subgroupAll({predicate})"
                ))
            }
            "__any_sync" => {
                let _mask = args.first()?;
                let predicate = args.get(1)?;
                Some(format!(
                    "/* __any_sync */ subgroupAny({predicate})"
                ))
            }
            "__activemask" => {
                Some("/* __activemask */ subgroupBallot(true)".to_string())
            }
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Synchronization -> WGSL
    // -----------------------------------------------------------------------
    fn map_sync_to_wgsl(name: &str, _args: &[String]) -> Option<String> {
        match name {
            "__syncthreads" => Some("workgroupBarrier()".to_string()),
            "__threadfence" => Some("storageBarrier()".to_string()),
            "__threadfence_block" => Some("workgroupBarrier()".to_string()),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Type conversion -> WGSL
    // -----------------------------------------------------------------------
    fn map_type_conversion_to_wgsl(name: &str, args: &[String]) -> Option<String> {
        let a = args.first()?;
        match name {
            "__float2int_rn" => Some(format!("i32(round({a}))")),
            "__int2float_rn" => Some(format!("f32({a})")),
            "__float2half" => {
                // WGSL f16 support via enable f16; extension
                Some(format!("f16({a})"))
            }
            "__half2float" => {
                Some(format!("f32({a})"))
            }
            "__float2uint_rn" => Some(format!("u32(round({a}))")),
            "__uint2float_rn" => Some(format!("f32({a})")),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Query helpers
    // -----------------------------------------------------------------------

    /// Returns true if the given function name is a recognized CUDA built-in.
    pub fn is_builtin(name: &str) -> bool {
        Self::is_math_builtin(name)
            || Self::is_atomic_builtin(name)
            || Self::is_warp_builtin(name)
            || Self::is_sync_builtin(name)
            || Self::is_type_conversion_builtin(name)
    }

    /// Returns true if the name is a CUDA math built-in.
    pub fn is_math_builtin(name: &str) -> bool {
        matches!(
            name,
            "__sinf" | "sinf" | "sin"
                | "__cosf" | "cosf" | "cos"
                | "__expf" | "expf" | "exp"
                | "__logf" | "logf" | "log"
                | "__powf" | "powf" | "pow"
                | "sqrtf" | "sqrt" | "__fsqrt_rn"
                | "fabsf" | "fabs" | "__fabsf"
                | "fminf" | "fmin"
                | "fmaxf" | "fmax"
                | "ceilf" | "ceil" | "__ceilf"
                | "floorf" | "floor" | "__floorf"
                | "roundf" | "round" | "__roundf"
        )
    }

    /// Returns true if the name is a CUDA atomic built-in.
    pub fn is_atomic_builtin(name: &str) -> bool {
        matches!(
            name,
            "atomicAdd"
                | "atomicSub"
                | "atomicMin"
                | "atomicMax"
                | "atomicExch"
                | "atomicCAS"
                | "atomicAnd"
                | "atomicOr"
                | "atomicXor"
        )
    }

    /// Returns true if the name is a CUDA warp-level built-in.
    pub fn is_warp_builtin(name: &str) -> bool {
        matches!(
            name,
            "__shfl_sync"
                | "__shfl_xor_sync"
                | "__shfl_up_sync"
                | "__shfl_down_sync"
                | "__ballot_sync"
                | "__all_sync"
                | "__any_sync"
                | "__activemask"
        )
    }

    /// Returns true if the name is a CUDA synchronization built-in.
    pub fn is_sync_builtin(name: &str) -> bool {
        matches!(
            name,
            "__syncthreads" | "__threadfence" | "__threadfence_block"
        )
    }

    /// Returns true if the name is a CUDA type conversion built-in.
    pub fn is_type_conversion_builtin(name: &str) -> bool {
        matches!(
            name,
            "__float2int_rn"
                | "__int2float_rn"
                | "__float2half"
                | "__half2float"
                | "__float2uint_rn"
                | "__uint2float_rn"
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create args vec from string slices
    fn args(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    // -- Math Rust --
    #[test]
    fn test_sinf_to_rust() {
        let result = BuiltinMapper::map_to_rust("__sinf", &args(&["x"]));
        assert_eq!(result, Some("(x as f32).sin()".to_string()));
    }

    #[test]
    fn test_powf_to_rust() {
        let result = BuiltinMapper::map_to_rust("__powf", &args(&["base", "exp"]));
        assert_eq!(
            result,
            Some("(base as f32).powf(exp as f32)".to_string())
        );
    }

    #[test]
    fn test_fminf_to_rust() {
        let result = BuiltinMapper::map_to_rust("fminf", &args(&["a", "b"]));
        assert_eq!(result, Some("(a as f32).min(b as f32)".to_string()));
    }

    // -- Math WGSL --
    #[test]
    fn test_sinf_to_wgsl() {
        let result = BuiltinMapper::map_to_wgsl("__sinf", &args(&["x"]));
        assert_eq!(result, Some("sin(x)".to_string()));
    }

    #[test]
    fn test_powf_to_wgsl() {
        let result = BuiltinMapper::map_to_wgsl("powf", &args(&["base", "exp"]));
        assert_eq!(result, Some("pow(base, exp)".to_string()));
    }

    // -- Atomics --
    #[test]
    fn test_atomic_add_to_wgsl() {
        let result = BuiltinMapper::map_to_wgsl("atomicAdd", &args(&["addr", "val"]));
        assert_eq!(result, Some("atomicAdd(&addr, val)".to_string()));
    }

    #[test]
    fn test_atomic_cas_to_wgsl() {
        let result = BuiltinMapper::map_to_wgsl("atomicCAS", &args(&["addr", "cmp", "val"]));
        assert!(result.is_some());
        assert!(result.unwrap().contains("atomicCompareExchangeWeak"));
    }

    // -- Warp --
    #[test]
    fn test_shfl_sync_to_rust() {
        let result = BuiltinMapper::map_to_rust("__shfl_sync", &args(&["0xFFFFFFFF", "var", "3"]));
        assert!(result.is_some());
        assert!(result.unwrap().contains("shuffle"));
    }

    #[test]
    fn test_ballot_to_wgsl() {
        let result =
            BuiltinMapper::map_to_wgsl("__ballot_sync", &args(&["0xFFFFFFFF", "pred"]));
        assert!(result.is_some());
        assert!(result.unwrap().contains("subgroupBallot"));
    }

    // -- Sync --
    #[test]
    fn test_syncthreads_to_rust() {
        let result = BuiltinMapper::map_to_rust("__syncthreads", &args(&[]));
        assert_eq!(
            result,
            Some("cuda_rust_wasm::runtime::sync_threads()".to_string())
        );
    }

    #[test]
    fn test_syncthreads_to_wgsl() {
        let result = BuiltinMapper::map_to_wgsl("__syncthreads", &args(&[]));
        assert_eq!(result, Some("workgroupBarrier()".to_string()));
    }

    // -- Type conversion --
    #[test]
    fn test_float2int_to_rust() {
        let result = BuiltinMapper::map_to_rust("__float2int_rn", &args(&["x"]));
        assert_eq!(result, Some("(x as f32).round() as i32".to_string()));
    }

    #[test]
    fn test_float2half_to_wgsl() {
        let result = BuiltinMapper::map_to_wgsl("__float2half", &args(&["x"]));
        assert_eq!(result, Some("f16(x)".to_string()));
    }

    // -- Query helpers --
    #[test]
    fn test_is_builtin() {
        assert!(BuiltinMapper::is_builtin("__sinf"));
        assert!(BuiltinMapper::is_builtin("atomicAdd"));
        assert!(BuiltinMapper::is_builtin("__shfl_sync"));
        assert!(BuiltinMapper::is_builtin("__syncthreads"));
        assert!(BuiltinMapper::is_builtin("__float2int_rn"));
        assert!(!BuiltinMapper::is_builtin("my_custom_function"));
    }

    // -- Unknown function --
    #[test]
    fn test_unknown_function_returns_none() {
        assert!(BuiltinMapper::map_to_rust("unknown_func", &args(&["x"])).is_none());
        assert!(BuiltinMapper::map_to_wgsl("unknown_func", &args(&["x"])).is_none());
    }
}
