//! CUDA memory space to target memory mapping
//!
//! Maps CUDA memory address spaces (global, shared, constant, register, local)
//! to their equivalents in Rust (for CPU fallback) and WGSL (for WebGPU compute
//! shaders).

use crate::parser::ast::StorageClass;

/// Memory space descriptor for a target platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryMapping {
    /// The target-specific storage qualifier / address space annotation.
    pub qualifier: String,
    /// Human-readable description of the mapping.
    pub description: String,
    /// Whether this memory space requires explicit synchronisation.
    pub requires_sync: bool,
    /// Whether the memory is read-only.
    pub read_only: bool,
}

/// Maps CUDA memory spaces to Rust and WGSL equivalents.
pub struct MemoryMapper;

impl MemoryMapper {
    // -----------------------------------------------------------------------
    // Rust target
    // -----------------------------------------------------------------------

    /// Map a CUDA `StorageClass` to its Rust representation.
    pub fn to_rust(storage: &StorageClass) -> MemoryMapping {
        match storage {
            StorageClass::Global => MemoryMapping {
                qualifier: "/* global */ ".to_string(),
                description: "Heap-allocated device buffer (Vec<T> or &mut [T])".to_string(),
                requires_sync: false,
                read_only: false,
            },
            StorageClass::Shared => MemoryMapping {
                qualifier: "#[shared] ".to_string(),
                description: "Thread-local shared memory (SharedMemory<T>)".to_string(),
                requires_sync: true,
                read_only: false,
            },
            StorageClass::Constant => MemoryMapping {
                qualifier: "const ".to_string(),
                description: "Compile-time constant (const or static)".to_string(),
                requires_sync: false,
                read_only: true,
            },
            StorageClass::Register => MemoryMapping {
                qualifier: "".to_string(),
                description: "Local variable (stack-allocated)".to_string(),
                requires_sync: false,
                read_only: false,
            },
            StorageClass::Local => MemoryMapping {
                qualifier: "".to_string(),
                description: "Local variable (stack-allocated)".to_string(),
                requires_sync: false,
                read_only: false,
            },
            StorageClass::Auto => MemoryMapping {
                qualifier: "let ".to_string(),
                description: "Auto storage (stack-allocated)".to_string(),
                requires_sync: false,
                read_only: false,
            },
        }
    }

    /// Generate a Rust variable declaration prefix for the given storage class.
    ///
    /// # Examples
    /// - `StorageClass::Shared` -> `"/* __shared__ */ let mut "`
    /// - `StorageClass::Constant` -> `"const "`
    /// - `StorageClass::Auto` -> `"let "`
    pub fn rust_var_prefix(storage: &StorageClass, mutable: bool) -> String {
        match storage {
            StorageClass::Shared => {
                if mutable {
                    "/* __shared__ */ let mut ".to_string()
                } else {
                    "/* __shared__ */ let ".to_string()
                }
            }
            StorageClass::Constant => "const ".to_string(),
            StorageClass::Global => {
                if mutable {
                    "/* __device__ */ static mut ".to_string()
                } else {
                    "/* __device__ */ static ".to_string()
                }
            }
            StorageClass::Register | StorageClass::Local | StorageClass::Auto => {
                if mutable {
                    "let mut ".to_string()
                } else {
                    "let ".to_string()
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // WGSL target
    // -----------------------------------------------------------------------

    /// Map a CUDA `StorageClass` to its WGSL representation.
    pub fn to_wgsl(storage: &StorageClass) -> MemoryMapping {
        match storage {
            StorageClass::Global => MemoryMapping {
                qualifier: "var<storage, read_write>".to_string(),
                description: "Storage buffer (read_write)".to_string(),
                requires_sync: false,
                read_only: false,
            },
            StorageClass::Shared => MemoryMapping {
                qualifier: "var<workgroup>".to_string(),
                description: "Workgroup memory (shared within workgroup)".to_string(),
                requires_sync: true,
                read_only: false,
            },
            StorageClass::Constant => MemoryMapping {
                qualifier: "var<uniform>".to_string(),
                description: "Uniform buffer (read-only)".to_string(),
                requires_sync: false,
                read_only: true,
            },
            StorageClass::Register => MemoryMapping {
                qualifier: "var<private>".to_string(),
                description: "Private variable (per-invocation)".to_string(),
                requires_sync: false,
                read_only: false,
            },
            StorageClass::Local => MemoryMapping {
                qualifier: "var<private>".to_string(),
                description: "Private variable (per-invocation)".to_string(),
                requires_sync: false,
                read_only: false,
            },
            StorageClass::Auto => MemoryMapping {
                qualifier: "var".to_string(),
                description: "Function-scope variable".to_string(),
                requires_sync: false,
                read_only: false,
            },
        }
    }

    /// Generate a WGSL variable declaration for the given storage class.
    ///
    /// # Arguments
    /// * `storage` - The CUDA storage class
    /// * `name` - Variable name
    /// * `wgsl_type` - WGSL type string (e.g. "f32", "array<f32, 256>")
    ///
    /// # Returns
    /// A complete WGSL variable declaration string.
    pub fn wgsl_var_decl(storage: &StorageClass, name: &str, wgsl_type: &str) -> String {
        let mapping = Self::to_wgsl(storage);
        format!("{} {}: {};", mapping.qualifier, name, wgsl_type)
    }

    /// Generate a WGSL binding declaration for a kernel parameter.
    ///
    /// # Arguments
    /// * `storage` - The CUDA storage class
    /// * `group` - Binding group number
    /// * `binding` - Binding index
    /// * `name` - Variable name
    /// * `wgsl_type` - WGSL type string
    /// * `read_only` - Whether the binding is read-only
    pub fn wgsl_binding_decl(
        storage: &StorageClass,
        group: u32,
        binding: u32,
        name: &str,
        wgsl_type: &str,
        read_only: bool,
    ) -> String {
        let access = match storage {
            StorageClass::Constant => "var<storage, read>",
            StorageClass::Global => {
                if read_only {
                    "var<storage, read>"
                } else {
                    "var<storage, read_write>"
                }
            }
            _ => {
                let mapping = Self::to_wgsl(storage);
                return format!(
                    "@group({group}) @binding({binding})\n{} {name}: {wgsl_type};",
                    mapping.qualifier
                );
            }
        };

        format!(
            "@group({group}) @binding({binding})\n{access} {name}: {wgsl_type};"
        )
    }

    // -----------------------------------------------------------------------
    // CUDA memory space name -> StorageClass
    // -----------------------------------------------------------------------

    /// Parse a CUDA memory space qualifier string into a `StorageClass`.
    pub fn parse_cuda_qualifier(qualifier: &str) -> StorageClass {
        match qualifier.trim() {
            "__shared__" | "shared" => StorageClass::Shared,
            "__constant__" | "constant" => StorageClass::Constant,
            "__device__" | "device" => StorageClass::Global,
            "__managed__" | "managed" => StorageClass::Global,
            "register" => StorageClass::Register,
            "local" => StorageClass::Local,
            _ => StorageClass::Auto,
        }
    }

    // -----------------------------------------------------------------------
    // Query helpers
    // -----------------------------------------------------------------------

    /// Returns true if the storage class requires barrier synchronisation
    /// before other threads can see writes.
    pub fn requires_barrier(storage: &StorageClass) -> bool {
        matches!(storage, StorageClass::Shared)
    }

    /// Returns true if the storage class is read-only.
    pub fn is_read_only(storage: &StorageClass) -> bool {
        matches!(storage, StorageClass::Constant)
    }

    /// Returns the WGSL barrier function name needed after writes to this
    /// memory space, if any.
    pub fn wgsl_barrier(storage: &StorageClass) -> Option<&'static str> {
        match storage {
            StorageClass::Shared => Some("workgroupBarrier()"),
            StorageClass::Global => Some("storageBarrier()"),
            _ => None,
        }
    }

    /// Returns the Rust synchronization primitive needed after writes to this
    /// memory space, if any.
    pub fn rust_barrier(storage: &StorageClass) -> Option<&'static str> {
        match storage {
            StorageClass::Shared => {
                Some("std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst)")
            }
            StorageClass::Global => {
                Some("std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst)")
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_to_rust() {
        let mapping = MemoryMapper::to_rust(&StorageClass::Global);
        assert!(!mapping.requires_sync);
        assert!(!mapping.read_only);
    }

    #[test]
    fn test_shared_to_rust() {
        let mapping = MemoryMapper::to_rust(&StorageClass::Shared);
        assert!(mapping.requires_sync);
        assert!(!mapping.read_only);
    }

    #[test]
    fn test_constant_to_rust() {
        let mapping = MemoryMapper::to_rust(&StorageClass::Constant);
        assert!(!mapping.requires_sync);
        assert!(mapping.read_only);
        assert_eq!(mapping.qualifier, "const ");
    }

    #[test]
    fn test_register_to_rust() {
        let mapping = MemoryMapper::to_rust(&StorageClass::Register);
        assert!(!mapping.requires_sync);
        assert_eq!(mapping.qualifier, "");
    }

    #[test]
    fn test_rust_var_prefix_shared() {
        let prefix = MemoryMapper::rust_var_prefix(&StorageClass::Shared, true);
        assert!(prefix.contains("__shared__"));
        assert!(prefix.contains("let mut"));
    }

    #[test]
    fn test_rust_var_prefix_const() {
        let prefix = MemoryMapper::rust_var_prefix(&StorageClass::Constant, false);
        assert_eq!(prefix, "const ");
    }

    #[test]
    fn test_global_to_wgsl() {
        let mapping = MemoryMapper::to_wgsl(&StorageClass::Global);
        assert_eq!(mapping.qualifier, "var<storage, read_write>");
        assert!(!mapping.read_only);
    }

    #[test]
    fn test_shared_to_wgsl() {
        let mapping = MemoryMapper::to_wgsl(&StorageClass::Shared);
        assert_eq!(mapping.qualifier, "var<workgroup>");
        assert!(mapping.requires_sync);
    }

    #[test]
    fn test_constant_to_wgsl() {
        let mapping = MemoryMapper::to_wgsl(&StorageClass::Constant);
        assert_eq!(mapping.qualifier, "var<uniform>");
        assert!(mapping.read_only);
    }

    #[test]
    fn test_register_to_wgsl() {
        let mapping = MemoryMapper::to_wgsl(&StorageClass::Register);
        assert_eq!(mapping.qualifier, "var<private>");
    }

    #[test]
    fn test_wgsl_var_decl() {
        let decl = MemoryMapper::wgsl_var_decl(
            &StorageClass::Shared,
            "shared_data",
            "array<f32, 256>",
        );
        assert_eq!(decl, "var<workgroup> shared_data: array<f32, 256>;");
    }

    #[test]
    fn test_wgsl_binding_decl() {
        let decl = MemoryMapper::wgsl_binding_decl(
            &StorageClass::Global,
            0,
            0,
            "data",
            "array<f32>",
            false,
        );
        assert!(decl.contains("@group(0) @binding(0)"));
        assert!(decl.contains("read_write"));
    }

    #[test]
    fn test_wgsl_binding_decl_readonly() {
        let decl = MemoryMapper::wgsl_binding_decl(
            &StorageClass::Global,
            0,
            1,
            "input",
            "array<f32>",
            true,
        );
        assert!(decl.contains("read"));
        assert!(!decl.contains("read_write"));
    }

    #[test]
    fn test_parse_cuda_qualifier() {
        assert!(matches!(
            MemoryMapper::parse_cuda_qualifier("__shared__"),
            StorageClass::Shared
        ));
        assert!(matches!(
            MemoryMapper::parse_cuda_qualifier("__constant__"),
            StorageClass::Constant
        ));
        assert!(matches!(
            MemoryMapper::parse_cuda_qualifier("__device__"),
            StorageClass::Global
        ));
        assert!(matches!(
            MemoryMapper::parse_cuda_qualifier("register"),
            StorageClass::Register
        ));
        assert!(matches!(
            MemoryMapper::parse_cuda_qualifier("unknown"),
            StorageClass::Auto
        ));
    }

    #[test]
    fn test_requires_barrier() {
        assert!(MemoryMapper::requires_barrier(&StorageClass::Shared));
        assert!(!MemoryMapper::requires_barrier(&StorageClass::Global));
        assert!(!MemoryMapper::requires_barrier(&StorageClass::Register));
    }

    #[test]
    fn test_is_read_only() {
        assert!(MemoryMapper::is_read_only(&StorageClass::Constant));
        assert!(!MemoryMapper::is_read_only(&StorageClass::Global));
        assert!(!MemoryMapper::is_read_only(&StorageClass::Shared));
    }

    #[test]
    fn test_wgsl_barrier() {
        assert_eq!(
            MemoryMapper::wgsl_barrier(&StorageClass::Shared),
            Some("workgroupBarrier()")
        );
        assert_eq!(
            MemoryMapper::wgsl_barrier(&StorageClass::Global),
            Some("storageBarrier()")
        );
        assert_eq!(MemoryMapper::wgsl_barrier(&StorageClass::Register), None);
    }

    #[test]
    fn test_rust_barrier() {
        assert!(MemoryMapper::rust_barrier(&StorageClass::Shared).is_some());
        assert!(MemoryMapper::rust_barrier(&StorageClass::Global).is_some());
        assert!(MemoryMapper::rust_barrier(&StorageClass::Register).is_none());
    }
}
