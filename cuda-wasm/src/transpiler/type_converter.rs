//! CUDA type to Rust/WGSL type conversion
//!
//! Converts CUDA type names (as strings or AST `Type` nodes) to their
//! corresponding Rust and WGSL representations. Handles scalar types,
//! vector types (float2, float3, float4, int2, etc.), and half precision.

use crate::parser::ast::{Type, IntType, FloatType, VectorType};

/// Converts CUDA types to target language types.
pub struct TypeConverter;

impl TypeConverter {
    // -----------------------------------------------------------------------
    // AST Type -> Rust string
    // -----------------------------------------------------------------------

    /// Convert an AST `Type` to its Rust string representation.
    pub fn to_rust(ty: &Type) -> String {
        match ty {
            Type::Void => "()".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Int(int_ty) => Self::int_type_to_rust(int_ty),
            Type::Float(float_ty) => Self::float_type_to_rust(float_ty),
            Type::Pointer(inner) => {
                let inner_str = Self::to_rust(inner);
                format!("*mut {inner_str}")
            }
            Type::Array(inner, size) => {
                let inner_str = Self::to_rust(inner);
                match size {
                    Some(n) => format!("[{inner_str}; {n}]"),
                    None => format!("&[{inner_str}]"),
                }
            }
            Type::Vector(vec_ty) => Self::vector_type_to_rust(vec_ty),
            Type::Named(name) => Self::cuda_named_type_to_rust(name),
            Type::Texture(_) => "/* texture type */ ()".to_string(),
        }
    }

    /// Convert an AST `Type` to its WGSL string representation.
    pub fn to_wgsl(ty: &Type) -> Result<String, String> {
        match ty {
            Type::Void => Err("void type not supported in WGSL".to_string()),
            Type::Bool => Ok("bool".to_string()),
            Type::Int(int_ty) => Self::int_type_to_wgsl(int_ty),
            Type::Float(float_ty) => Self::float_type_to_wgsl(float_ty),
            Type::Pointer(inner) => {
                let inner_str = Self::to_wgsl(inner)?;
                Ok(format!("ptr<storage, {inner_str}>"))
            }
            Type::Array(inner, size) => {
                let inner_str = Self::to_wgsl(inner)?;
                match size {
                    Some(n) => Ok(format!("array<{inner_str}, {n}>")),
                    None => Ok(format!("array<{inner_str}>")),
                }
            }
            Type::Vector(vec_ty) => Self::vector_type_to_wgsl(vec_ty),
            Type::Named(name) => Self::cuda_named_type_to_wgsl(name),
            Type::Texture(_) => Err("Texture types not yet supported in WGSL".to_string()),
        }
    }

    // -----------------------------------------------------------------------
    // CUDA type name string -> Rust
    // -----------------------------------------------------------------------

    /// Convert a CUDA type name string to its Rust equivalent.
    pub fn cuda_name_to_rust(name: &str) -> String {
        match name {
            // Scalar types
            "void" => "()".to_string(),
            "bool" => "bool".to_string(),
            "char" | "signed char" => "i8".to_string(),
            "unsigned char" | "uchar" => "u8".to_string(),
            "short" | "signed short" => "i16".to_string(),
            "unsigned short" | "ushort" => "u16".to_string(),
            "int" | "signed int" | "signed" => "i32".to_string(),
            "unsigned int" | "unsigned" | "uint" => "u32".to_string(),
            "long" | "signed long" => "i64".to_string(),
            "unsigned long" | "ulong" => "u64".to_string(),
            "long long" | "signed long long" => "i64".to_string(),
            "unsigned long long" => "u64".to_string(),
            "float" => "f32".to_string(),
            "double" => "f64".to_string(),
            "half" | "__half" => "f32".to_string(), // f16 not stable; use f32 as fallback
            "size_t" => "usize".to_string(),
            "ptrdiff_t" => "isize".to_string(),

            // Vector types (CUDA built-in)
            "float1" => "[f32; 1]".to_string(),
            "float2" => "[f32; 2]".to_string(),
            "float3" => "[f32; 3]".to_string(),
            "float4" => "[f32; 4]".to_string(),
            "double1" => "[f64; 1]".to_string(),
            "double2" => "[f64; 2]".to_string(),
            "double3" => "[f64; 3]".to_string(),
            "double4" => "[f64; 4]".to_string(),
            "int1" => "[i32; 1]".to_string(),
            "int2" => "[i32; 2]".to_string(),
            "int3" => "[i32; 3]".to_string(),
            "int4" => "[i32; 4]".to_string(),
            "uint1" => "[u32; 1]".to_string(),
            "uint2" => "[u32; 2]".to_string(),
            "uint3" => "[u32; 3]".to_string(),
            "uint4" => "[u32; 4]".to_string(),
            "short1" => "[i16; 1]".to_string(),
            "short2" => "[i16; 2]".to_string(),
            "short3" => "[i16; 3]".to_string(),
            "short4" => "[i16; 4]".to_string(),
            "ushort1" => "[u16; 1]".to_string(),
            "ushort2" => "[u16; 2]".to_string(),
            "ushort3" => "[u16; 3]".to_string(),
            "ushort4" => "[u16; 4]".to_string(),
            "char1" => "[i8; 1]".to_string(),
            "char2" => "[i8; 2]".to_string(),
            "char3" => "[i8; 3]".to_string(),
            "char4" => "[i8; 4]".to_string(),
            "uchar1" => "[u8; 1]".to_string(),
            "uchar2" => "[u8; 2]".to_string(),
            "uchar3" => "[u8; 3]".to_string(),
            "uchar4" => "[u8; 4]".to_string(),
            "long1" => "[i64; 1]".to_string(),
            "long2" => "[i64; 2]".to_string(),
            "long3" => "[i64; 3]".to_string(),
            "long4" => "[i64; 4]".to_string(),
            "ulong1" => "[u64; 1]".to_string(),
            "ulong2" => "[u64; 2]".to_string(),
            "ulong3" => "[u64; 3]".to_string(),
            "ulong4" => "[u64; 4]".to_string(),
            "longlong1" => "[i64; 1]".to_string(),
            "longlong2" => "[i64; 2]".to_string(),
            "ulonglong1" => "[u64; 1]".to_string(),
            "ulonglong2" => "[u64; 2]".to_string(),

            // Half-precision vector types
            "half2" | "__half2" => "[f32; 2]".to_string(),

            // CUDA dim3 type
            "dim3" => "(u32, u32, u32)".to_string(),

            // Fallback: use the name as-is (user-defined type)
            other => other.to_string(),
        }
    }

    /// Convert a CUDA type name string to its WGSL equivalent.
    pub fn cuda_name_to_wgsl(name: &str) -> Result<String, String> {
        match name {
            // Scalar types
            "void" => Err("void type not supported in WGSL".to_string()),
            "bool" => Ok("bool".to_string()),
            "char" | "signed char" | "short" | "signed short"
            | "int" | "signed int" | "signed" => Ok("i32".to_string()),
            "unsigned char" | "uchar" | "unsigned short" | "ushort"
            | "unsigned int" | "unsigned" | "uint" => Ok("u32".to_string()),
            "long" | "signed long" | "long long" | "signed long long" => {
                Err("i64 not supported in WGSL".to_string())
            }
            "unsigned long" | "ulong" | "unsigned long long" => {
                Err("u64 not supported in WGSL".to_string())
            }
            "float" => Ok("f32".to_string()),
            "double" => Err("f64 not supported in WGSL".to_string()),
            "half" | "__half" => Ok("f16".to_string()),
            "size_t" => Ok("u32".to_string()),

            // Vector types
            "float2" => Ok("vec2<f32>".to_string()),
            "float3" => Ok("vec3<f32>".to_string()),
            "float4" => Ok("vec4<f32>".to_string()),
            "int2" => Ok("vec2<i32>".to_string()),
            "int3" => Ok("vec3<i32>".to_string()),
            "int4" => Ok("vec4<i32>".to_string()),
            "uint2" => Ok("vec2<u32>".to_string()),
            "uint3" => Ok("vec3<u32>".to_string()),
            "uint4" => Ok("vec4<u32>".to_string()),
            "short2" => Ok("vec2<i32>".to_string()),
            "short3" => Ok("vec3<i32>".to_string()),
            "short4" => Ok("vec4<i32>".to_string()),
            "ushort2" => Ok("vec2<u32>".to_string()),
            "ushort3" => Ok("vec3<u32>".to_string()),
            "ushort4" => Ok("vec4<u32>".to_string()),
            "half2" | "__half2" => Ok("vec2<f16>".to_string()),

            // dim3
            "dim3" => Ok("vec3<u32>".to_string()),

            // Fallback: use the name as-is
            other => Ok(other.to_string()),
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers: AST Type components
    // -----------------------------------------------------------------------

    fn int_type_to_rust(int_ty: &IntType) -> String {
        match int_ty {
            IntType::I8 => "i8".to_string(),
            IntType::I16 => "i16".to_string(),
            IntType::I32 => "i32".to_string(),
            IntType::I64 => "i64".to_string(),
            IntType::U8 => "u8".to_string(),
            IntType::U16 => "u16".to_string(),
            IntType::U32 => "u32".to_string(),
            IntType::U64 => "u64".to_string(),
        }
    }

    fn float_type_to_rust(float_ty: &FloatType) -> String {
        match float_ty {
            FloatType::F16 => "f32".to_string(), // f16 not stable; use f32
            FloatType::F32 => "f32".to_string(),
            FloatType::F64 => "f64".to_string(),
        }
    }

    fn int_type_to_wgsl(int_ty: &IntType) -> Result<String, String> {
        match int_ty {
            IntType::I8 | IntType::I16 | IntType::I32 => Ok("i32".to_string()),
            IntType::I64 => Err("i64 not supported in WGSL".to_string()),
            IntType::U8 | IntType::U16 | IntType::U32 => Ok("u32".to_string()),
            IntType::U64 => Err("u64 not supported in WGSL".to_string()),
        }
    }

    fn float_type_to_wgsl(float_ty: &FloatType) -> Result<String, String> {
        match float_ty {
            FloatType::F16 => Ok("f16".to_string()),
            FloatType::F32 => Ok("f32".to_string()),
            FloatType::F64 => Err("f64 not supported in WGSL".to_string()),
        }
    }

    fn vector_type_to_rust(vec_ty: &VectorType) -> String {
        let elem = Self::to_rust(&vec_ty.element);
        let size = vec_ty.size;
        format!("[{elem}; {size}]")
    }

    fn vector_type_to_wgsl(vec_ty: &VectorType) -> Result<String, String> {
        let elem = Self::to_wgsl(&vec_ty.element)?;
        let size = vec_ty.size;
        Ok(format!("vec{size}<{elem}>"))
    }

    fn cuda_named_type_to_rust(name: &str) -> String {
        // Try mapping as a CUDA type name first
        let mapped = Self::cuda_name_to_rust(name);
        if mapped != name {
            mapped
        } else {
            // User-defined type: use as-is
            name.to_string()
        }
    }

    fn cuda_named_type_to_wgsl(name: &str) -> Result<String, String> {
        Self::cuda_name_to_wgsl(name)
    }

    // -----------------------------------------------------------------------
    // Query helpers
    // -----------------------------------------------------------------------

    /// Returns true if the CUDA type name is a vector type (e.g. float4, int2).
    pub fn is_vector_type(name: &str) -> bool {
        matches!(
            name,
            "float1" | "float2" | "float3" | "float4"
                | "double1" | "double2" | "double3" | "double4"
                | "int1" | "int2" | "int3" | "int4"
                | "uint1" | "uint2" | "uint3" | "uint4"
                | "short1" | "short2" | "short3" | "short4"
                | "ushort1" | "ushort2" | "ushort3" | "ushort4"
                | "char1" | "char2" | "char3" | "char4"
                | "uchar1" | "uchar2" | "uchar3" | "uchar4"
                | "long1" | "long2" | "long3" | "long4"
                | "ulong1" | "ulong2" | "ulong3" | "ulong4"
                | "longlong1" | "longlong2"
                | "ulonglong1" | "ulonglong2"
                | "half2" | "__half2"
        )
    }

    /// Returns true if the CUDA type name is a half-precision type.
    pub fn is_half_type(name: &str) -> bool {
        matches!(name, "half" | "__half" | "half2" | "__half2")
    }

    /// Returns the number of components for a CUDA vector type name.
    /// Returns 1 for scalar types.
    pub fn vector_components(name: &str) -> u8 {
        if name.ends_with('4') && Self::is_vector_type(name) {
            4
        } else if name.ends_with('3') && Self::is_vector_type(name) {
            3
        } else if name.ends_with('2') && Self::is_vector_type(name) {
            2
        } else if name.ends_with('1') && Self::is_vector_type(name) {
            1
        } else {
            1
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // -- Scalar type Rust mapping --
    #[test]
    fn test_cuda_scalar_to_rust() {
        assert_eq!(TypeConverter::cuda_name_to_rust("int"), "i32");
        assert_eq!(TypeConverter::cuda_name_to_rust("unsigned int"), "u32");
        assert_eq!(TypeConverter::cuda_name_to_rust("float"), "f32");
        assert_eq!(TypeConverter::cuda_name_to_rust("double"), "f64");
        assert_eq!(TypeConverter::cuda_name_to_rust("void"), "()");
        assert_eq!(TypeConverter::cuda_name_to_rust("bool"), "bool");
        assert_eq!(TypeConverter::cuda_name_to_rust("size_t"), "usize");
        assert_eq!(TypeConverter::cuda_name_to_rust("half"), "f32");
    }

    // -- Vector type Rust mapping --
    #[test]
    fn test_cuda_vector_to_rust() {
        assert_eq!(TypeConverter::cuda_name_to_rust("float2"), "[f32; 2]");
        assert_eq!(TypeConverter::cuda_name_to_rust("float4"), "[f32; 4]");
        assert_eq!(TypeConverter::cuda_name_to_rust("int3"), "[i32; 3]");
        assert_eq!(TypeConverter::cuda_name_to_rust("uint4"), "[u32; 4]");
        assert_eq!(TypeConverter::cuda_name_to_rust("double2"), "[f64; 2]");
        assert_eq!(TypeConverter::cuda_name_to_rust("uchar4"), "[u8; 4]");
    }

    // -- Scalar type WGSL mapping --
    #[test]
    fn test_cuda_scalar_to_wgsl() {
        assert_eq!(TypeConverter::cuda_name_to_wgsl("int"), Ok("i32".to_string()));
        assert_eq!(TypeConverter::cuda_name_to_wgsl("float"), Ok("f32".to_string()));
        assert_eq!(TypeConverter::cuda_name_to_wgsl("half"), Ok("f16".to_string()));
        assert!(TypeConverter::cuda_name_to_wgsl("double").is_err());
        assert!(TypeConverter::cuda_name_to_wgsl("void").is_err());
    }

    // -- Vector type WGSL mapping --
    #[test]
    fn test_cuda_vector_to_wgsl() {
        assert_eq!(TypeConverter::cuda_name_to_wgsl("float2"), Ok("vec2<f32>".to_string()));
        assert_eq!(TypeConverter::cuda_name_to_wgsl("float4"), Ok("vec4<f32>".to_string()));
        assert_eq!(TypeConverter::cuda_name_to_wgsl("int3"), Ok("vec3<i32>".to_string()));
        assert_eq!(TypeConverter::cuda_name_to_wgsl("uint4"), Ok("vec4<u32>".to_string()));
        assert_eq!(TypeConverter::cuda_name_to_wgsl("half2"), Ok("vec2<f16>".to_string()));
    }

    // -- AST Type -> Rust --
    #[test]
    fn test_ast_type_to_rust() {
        assert_eq!(TypeConverter::to_rust(&Type::Void), "()");
        assert_eq!(TypeConverter::to_rust(&Type::Bool), "bool");
        assert_eq!(TypeConverter::to_rust(&Type::Int(IntType::I32)), "i32");
        assert_eq!(TypeConverter::to_rust(&Type::Float(FloatType::F32)), "f32");

        let ptr_ty = Type::Pointer(Box::new(Type::Float(FloatType::F32)));
        assert_eq!(TypeConverter::to_rust(&ptr_ty), "*mut f32");

        let arr_ty = Type::Array(Box::new(Type::Int(IntType::I32)), Some(16));
        assert_eq!(TypeConverter::to_rust(&arr_ty), "[i32; 16]");

        let vec_ty = Type::Vector(VectorType {
            element: Box::new(Type::Float(FloatType::F32)),
            size: 4,
        });
        assert_eq!(TypeConverter::to_rust(&vec_ty), "[f32; 4]");
    }

    // -- AST Type -> WGSL --
    #[test]
    fn test_ast_type_to_wgsl() {
        assert_eq!(TypeConverter::to_wgsl(&Type::Bool), Ok("bool".to_string()));
        assert_eq!(TypeConverter::to_wgsl(&Type::Int(IntType::I32)), Ok("i32".to_string()));
        assert_eq!(TypeConverter::to_wgsl(&Type::Float(FloatType::F32)), Ok("f32".to_string()));
        assert!(TypeConverter::to_wgsl(&Type::Void).is_err());
        assert!(TypeConverter::to_wgsl(&Type::Float(FloatType::F64)).is_err());

        let vec_ty = Type::Vector(VectorType {
            element: Box::new(Type::Float(FloatType::F32)),
            size: 3,
        });
        assert_eq!(TypeConverter::to_wgsl(&vec_ty), Ok("vec3<f32>".to_string()));
    }

    // -- Query helpers --
    #[test]
    fn test_is_vector_type() {
        assert!(TypeConverter::is_vector_type("float4"));
        assert!(TypeConverter::is_vector_type("int2"));
        assert!(TypeConverter::is_vector_type("uchar4"));
        assert!(TypeConverter::is_vector_type("half2"));
        assert!(!TypeConverter::is_vector_type("float"));
        assert!(!TypeConverter::is_vector_type("int"));
        assert!(!TypeConverter::is_vector_type("MyStruct"));
    }

    #[test]
    fn test_is_half_type() {
        assert!(TypeConverter::is_half_type("half"));
        assert!(TypeConverter::is_half_type("__half"));
        assert!(TypeConverter::is_half_type("half2"));
        assert!(!TypeConverter::is_half_type("float"));
    }

    #[test]
    fn test_vector_components() {
        assert_eq!(TypeConverter::vector_components("float4"), 4);
        assert_eq!(TypeConverter::vector_components("int3"), 3);
        assert_eq!(TypeConverter::vector_components("uint2"), 2);
        assert_eq!(TypeConverter::vector_components("char1"), 1);
        assert_eq!(TypeConverter::vector_components("float"), 1);
    }

    // -- User-defined type passthrough --
    #[test]
    fn test_user_defined_type() {
        assert_eq!(TypeConverter::cuda_name_to_rust("MyStruct"), "MyStruct");
        assert_eq!(
            TypeConverter::cuda_name_to_wgsl("MyStruct"),
            Ok("MyStruct".to_string())
        );
    }

    // -- dim3 --
    #[test]
    fn test_dim3() {
        assert_eq!(TypeConverter::cuda_name_to_rust("dim3"), "(u32, u32, u32)");
        assert_eq!(TypeConverter::cuda_name_to_wgsl("dim3"), Ok("vec3<u32>".to_string()));
    }
}
