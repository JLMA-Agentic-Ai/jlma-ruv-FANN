//! Tests for CUDA transpiler fidelity - verifying correct transpilation to Rust and WGSL
//!
//! These tests parse CUDA kernels, transpile them to Rust code or WGSL, and verify
//! that the generated output contains the expected constructs.

#[cfg(test)]
mod tests {
    use cuda_rust_wasm::parser::CudaParser;
    use cuda_rust_wasm::parser::ast::*;
    use cuda_rust_wasm::transpiler::Transpiler;
    use cuda_rust_wasm::transpiler::wgsl::WgslGenerator;

    // ---------------------------------------------------------------
    // Helper: parse and transpile to Rust
    // ---------------------------------------------------------------
    fn transpile_to_rust(source: &str) -> String {
        let parser = CudaParser::new();
        let ast = parser.parse(source).expect("Failed to parse CUDA source");
        let transpiler = Transpiler::new();
        transpiler.transpile(ast).expect("Failed to transpile to Rust")
    }

    // ---------------------------------------------------------------
    // Helper: parse and transpile to WGSL
    // ---------------------------------------------------------------
    fn transpile_to_wgsl(source: &str) -> String {
        let parser = CudaParser::new();
        let ast = parser.parse(source).expect("Failed to parse CUDA source");
        let transpiler = Transpiler::new();
        transpiler.to_wgsl(ast).expect("Failed to transpile to WGSL")
    }

    // ---------------------------------------------------------------
    // Helper: construct AST directly and transpile to WGSL
    // ---------------------------------------------------------------
    fn wgsl_from_ast(ast: Ast) -> String {
        let mut gen = WgslGenerator::new();
        gen.generate(ast).expect("Failed to generate WGSL")
    }

    // ---------------------------------------------------------------
    // Test 1: Transpile vectorAdd to Rust - verify Rust constructs
    // ---------------------------------------------------------------
    #[test]
    fn test_transpile_vector_add_to_rust() {
        let source = r#"
            __global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
            }
        "#;

        let rust_code = transpile_to_rust(source);

        // The generated Rust code should contain kernel annotation
        assert!(
            rust_code.contains("#[kernel]") || rust_code.contains("pub fn vectorAdd"),
            "Rust output should contain kernel annotation or function name. Got:\n{}",
            rust_code
        );

        // Should contain thread index mapping
        assert!(
            rust_code.contains("thread") || rust_code.contains("block"),
            "Rust output should reference thread or block indexing. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 2: Transpile vectorAdd to WGSL - verify WebGPU constructs
    // ---------------------------------------------------------------
    #[test]
    fn test_transpile_vector_add_to_wgsl() {
        let source = r#"
            __global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
            }
        "#;

        let wgsl_code = transpile_to_wgsl(source);

        // WGSL should contain compute shader annotation
        assert!(
            wgsl_code.contains("@compute"),
            "WGSL output should contain @compute annotation. Got:\n{}",
            wgsl_code
        );

        // WGSL should contain workgroup_size
        assert!(
            wgsl_code.contains("@workgroup_size"),
            "WGSL output should contain @workgroup_size. Got:\n{}",
            wgsl_code
        );

        // WGSL should contain function definition
        assert!(
            wgsl_code.contains("fn vectorAdd"),
            "WGSL output should contain function name. Got:\n{}",
            wgsl_code
        );
    }

    // ---------------------------------------------------------------
    // Test 3: threadIdx/blockIdx mapping in WGSL
    // ---------------------------------------------------------------
    #[test]
    fn test_thread_block_idx_mapping_wgsl() {
        let source = r#"
            __global__ void kernel(float* data) {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
            }
        "#;

        let wgsl_code = transpile_to_wgsl(source);

        // WGSL should map CUDA builtins to WGSL builtins
        assert!(
            wgsl_code.contains("local_invocation_id") || wgsl_code.contains("threadIdx"),
            "WGSL should map threadIdx to local_invocation_id or use threadIdx alias. Got:\n{}",
            wgsl_code
        );

        assert!(
            wgsl_code.contains("workgroup_id") || wgsl_code.contains("blockIdx"),
            "WGSL should map blockIdx to workgroup_id or use blockIdx alias. Got:\n{}",
            wgsl_code
        );
    }

    // ---------------------------------------------------------------
    // Test 4: Shared memory mapping in WGSL
    // ---------------------------------------------------------------
    #[test]
    fn test_shared_memory_mapping_wgsl() {
        // Build AST with shared memory variable directly
        let ast = Ast {
            items: vec![
                Item::GlobalVar(GlobalVar {
                    name: "shared_data".to_string(),
                    ty: Type::Array(Box::new(Type::Float(FloatType::F32)), Some(256)),
                    storage: StorageClass::Shared,
                    init: None,
                }),
                Item::Kernel(KernelDef {
                    name: "shared_test".to_string(),
                    params: vec![
                        Parameter {
                            name: "output".to_string(),
                            ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                            qualifiers: vec![],
                        },
                    ],
                    body: Block {
                        statements: vec![],
                    },
                    attributes: vec![],
                }),
            ],
        };

        let wgsl = wgsl_from_ast(ast);

        // Shared memory should map to var<workgroup>
        assert!(
            wgsl.contains("var<workgroup>"),
            "Shared memory should map to var<workgroup> in WGSL. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 5: __syncthreads -> workgroupBarrier mapping
    // ---------------------------------------------------------------
    #[test]
    fn test_syncthreads_to_workgroup_barrier() {
        let ast = Ast {
            items: vec![
                Item::Kernel(KernelDef {
                    name: "sync_test".to_string(),
                    params: vec![
                        Parameter {
                            name: "data".to_string(),
                            ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                            qualifiers: vec![],
                        },
                    ],
                    body: Block {
                        statements: vec![
                            Statement::SyncThreads,
                        ],
                    },
                    attributes: vec![],
                }),
            ],
        };

        let wgsl = wgsl_from_ast(ast);

        assert!(
            wgsl.contains("workgroupBarrier()"),
            "__syncthreads should map to workgroupBarrier() in WGSL. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 6: Type conversions (float -> f32, int -> i32, etc.)
    // ---------------------------------------------------------------
    #[test]
    fn test_type_conversions_in_wgsl() {
        let gen = WgslGenerator::new();

        // Test via AST with various types - we verify through full generation
        let ast = Ast {
            items: vec![
                Item::Kernel(KernelDef {
                    name: "type_test".to_string(),
                    params: vec![
                        Parameter {
                            name: "f_data".to_string(),
                            ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                            qualifiers: vec![],
                        },
                        Parameter {
                            name: "i_data".to_string(),
                            ty: Type::Pointer(Box::new(Type::Int(IntType::I32))),
                            qualifiers: vec![],
                        },
                        Parameter {
                            name: "u_data".to_string(),
                            ty: Type::Pointer(Box::new(Type::Int(IntType::U32))),
                            qualifiers: vec![],
                        },
                    ],
                    body: Block {
                        statements: vec![
                            Statement::VarDecl {
                                name: "f".to_string(),
                                ty: Type::Float(FloatType::F32),
                                init: Some(Expression::Literal(Literal::Float(1.0))),
                                storage: StorageClass::Auto,
                            },
                            Statement::VarDecl {
                                name: "i".to_string(),
                                ty: Type::Int(IntType::I32),
                                init: Some(Expression::Literal(Literal::Int(42))),
                                storage: StorageClass::Auto,
                            },
                            Statement::VarDecl {
                                name: "u".to_string(),
                                ty: Type::Int(IntType::U32),
                                init: Some(Expression::Literal(Literal::UInt(100))),
                                storage: StorageClass::Auto,
                            },
                            Statement::VarDecl {
                                name: "b".to_string(),
                                ty: Type::Bool,
                                init: Some(Expression::Literal(Literal::Bool(true))),
                                storage: StorageClass::Auto,
                            },
                        ],
                    },
                    attributes: vec![],
                }),
            ],
        };

        let wgsl = wgsl_from_ast(ast);

        // Check that WGSL type names appear in the output
        assert!(wgsl.contains("f32"), "Should contain f32 type. Got:\n{}", wgsl);
        assert!(wgsl.contains("i32"), "Should contain i32 type. Got:\n{}", wgsl);
        assert!(wgsl.contains("u32"), "Should contain u32 type. Got:\n{}", wgsl);
        assert!(wgsl.contains("bool"), "Should contain bool type. Got:\n{}", wgsl);
    }

    // ---------------------------------------------------------------
    // Test 7: For loop -> while loop conversion in WGSL
    // ---------------------------------------------------------------
    #[test]
    fn test_for_loop_to_while_in_wgsl() {
        let ast = Ast {
            items: vec![
                Item::Kernel(KernelDef {
                    name: "loop_test".to_string(),
                    params: vec![
                        Parameter {
                            name: "data".to_string(),
                            ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                            qualifiers: vec![],
                        },
                    ],
                    body: Block {
                        statements: vec![
                            Statement::For {
                                init: Some(Box::new(Statement::VarDecl {
                                    name: "i".to_string(),
                                    ty: Type::Int(IntType::I32),
                                    init: Some(Expression::Literal(Literal::Int(0))),
                                    storage: StorageClass::Auto,
                                })),
                                condition: Some(Expression::Binary {
                                    op: BinaryOp::Lt,
                                    left: Box::new(Expression::Var("i".to_string())),
                                    right: Box::new(Expression::Literal(Literal::Int(10))),
                                }),
                                update: Some(Expression::Binary {
                                    op: BinaryOp::Assign,
                                    left: Box::new(Expression::Var("i".to_string())),
                                    right: Box::new(Expression::Binary {
                                        op: BinaryOp::Add,
                                        left: Box::new(Expression::Var("i".to_string())),
                                        right: Box::new(Expression::Literal(Literal::Int(1))),
                                    }),
                                }),
                                body: Box::new(Statement::Expr(Expression::Binary {
                                    op: BinaryOp::Assign,
                                    left: Box::new(Expression::Index {
                                        array: Box::new(Expression::Var("data".to_string())),
                                        index: Box::new(Expression::Var("i".to_string())),
                                    }),
                                    right: Box::new(Expression::Literal(Literal::Float(0.0))),
                                })),
                            },
                        ],
                    },
                    attributes: vec![],
                }),
            ],
        };

        let wgsl = wgsl_from_ast(ast);

        // WGSL doesn't have for loops; the generator converts them to while loops
        assert!(
            wgsl.contains("while"),
            "For loop should be converted to while loop in WGSL. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 8: Multiple kernels transpilation
    // ---------------------------------------------------------------
    #[test]
    fn test_multiple_kernels_transpilation() {
        let ast = Ast {
            items: vec![
                Item::Kernel(KernelDef {
                    name: "kernel_a".to_string(),
                    params: vec![Parameter {
                        name: "data".to_string(),
                        ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                        qualifiers: vec![],
                    }],
                    body: Block {
                        statements: vec![Statement::VarDecl {
                            name: "idx".to_string(),
                            ty: Type::Int(IntType::I32),
                            init: Some(Expression::ThreadIdx(Dimension::X)),
                            storage: StorageClass::Auto,
                        }],
                    },
                    attributes: vec![],
                }),
                Item::Kernel(KernelDef {
                    name: "kernel_b".to_string(),
                    params: vec![Parameter {
                        name: "output".to_string(),
                        ty: Type::Pointer(Box::new(Type::Int(IntType::I32))),
                        qualifiers: vec![],
                    }],
                    body: Block {
                        statements: vec![Statement::VarDecl {
                            name: "tid".to_string(),
                            ty: Type::Int(IntType::I32),
                            init: Some(Expression::ThreadIdx(Dimension::X)),
                            storage: StorageClass::Auto,
                        }],
                    },
                    attributes: vec![],
                }),
            ],
        };

        // Transpile to Rust
        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast.clone()).expect("Failed to transpile to Rust");

        // Both kernel names should appear in the Rust output
        assert!(
            rust_code.contains("kernel_a"),
            "Rust output should contain kernel_a. Got:\n{}",
            rust_code
        );
        assert!(
            rust_code.contains("kernel_b"),
            "Rust output should contain kernel_b. Got:\n{}",
            rust_code
        );

        // Transpile to WGSL
        let wgsl = wgsl_from_ast(ast);

        // Both kernel names should appear in the WGSL output
        assert!(
            wgsl.contains("fn kernel_a"),
            "WGSL output should contain fn kernel_a. Got:\n{}",
            wgsl
        );
        assert!(
            wgsl.contains("fn kernel_b"),
            "WGSL output should contain fn kernel_b. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 9: Device function transpilation to WGSL
    // ---------------------------------------------------------------
    #[test]
    fn test_device_function_transpilation_wgsl() {
        let ast = Ast {
            items: vec![
                Item::DeviceFunction(FunctionDef {
                    name: "helper".to_string(),
                    return_type: Type::Float(FloatType::F32),
                    params: vec![
                        Parameter {
                            name: "x".to_string(),
                            ty: Type::Float(FloatType::F32),
                            qualifiers: vec![],
                        },
                    ],
                    body: Block {
                        statements: vec![
                            Statement::Return(Some(Expression::Binary {
                                op: BinaryOp::Mul,
                                left: Box::new(Expression::Var("x".to_string())),
                                right: Box::new(Expression::Var("x".to_string())),
                            })),
                        ],
                    },
                    qualifiers: vec![FunctionQualifier::Device],
                }),
                Item::Kernel(KernelDef {
                    name: "main_kernel".to_string(),
                    params: vec![Parameter {
                        name: "data".to_string(),
                        ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                        qualifiers: vec![],
                    }],
                    body: Block {
                        statements: vec![],
                    },
                    attributes: vec![],
                }),
            ],
        };

        let wgsl = wgsl_from_ast(ast);

        // The device function should appear as a regular function in WGSL
        assert!(
            wgsl.contains("fn helper"),
            "WGSL should contain device function 'helper'. Got:\n{}",
            wgsl
        );

        // It should have the correct return type
        assert!(
            wgsl.contains("f32"),
            "WGSL helper function should return f32. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 10: WGSL binding generation for pointer parameters
    // ---------------------------------------------------------------
    #[test]
    fn test_wgsl_binding_generation() {
        let ast = Ast {
            items: vec![
                Item::Kernel(KernelDef {
                    name: "binding_test".to_string(),
                    params: vec![
                        Parameter {
                            name: "input".to_string(),
                            ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                            qualifiers: vec![ParamQualifier::Const],
                        },
                        Parameter {
                            name: "output".to_string(),
                            ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                            qualifiers: vec![],
                        },
                    ],
                    body: Block { statements: vec![] },
                    attributes: vec![],
                }),
            ],
        };

        let wgsl = wgsl_from_ast(ast);

        // Should have @group(0) @binding annotations
        assert!(
            wgsl.contains("@group(0) @binding(0)"),
            "WGSL should contain binding(0). Got:\n{}",
            wgsl
        );
        assert!(
            wgsl.contains("@group(0) @binding(1)"),
            "WGSL should contain binding(1). Got:\n{}",
            wgsl
        );

        // Const pointer should generate read-only storage
        assert!(
            wgsl.contains("var<storage, read>"),
            "Const pointer should generate read-only storage. Got:\n{}",
            wgsl
        );

        // Non-const pointer should generate read-write storage
        assert!(
            wgsl.contains("var<storage, read_write>"),
            "Non-const pointer should generate read_write storage. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 11: WGSL workgroup size configuration
    // ---------------------------------------------------------------
    #[test]
    fn test_wgsl_workgroup_size_configuration() {
        let ast = Ast {
            items: vec![
                Item::Kernel(KernelDef {
                    name: "wg_test".to_string(),
                    params: vec![],
                    body: Block { statements: vec![] },
                    attributes: vec![],
                }),
            ],
        };

        // Default workgroup size
        let mut gen = WgslGenerator::new();
        let wgsl = gen.generate(ast.clone()).expect("Failed to generate WGSL");
        assert!(
            wgsl.contains("@workgroup_size(64, 1, 1)"),
            "Default workgroup size should be 64,1,1. Got:\n{}",
            wgsl
        );

        // Custom workgroup size
        let mut gen2 = WgslGenerator::new().with_workgroup_size(256, 1, 1);
        let wgsl2 = gen2.generate(ast).expect("Failed to generate WGSL");
        assert!(
            wgsl2.contains("@workgroup_size(256, 1, 1)"),
            "Custom workgroup size should be 256,1,1. Got:\n{}",
            wgsl2
        );
    }

    // ---------------------------------------------------------------
    // Test 12: Transpiler handles GlobalVar with const storage
    // ---------------------------------------------------------------
    #[test]
    fn test_constant_memory_wgsl() {
        let ast = Ast {
            items: vec![
                Item::GlobalVar(GlobalVar {
                    name: "weights".to_string(),
                    ty: Type::Array(Box::new(Type::Float(FloatType::F32)), Some(5)),
                    storage: StorageClass::Constant,
                    init: None,
                }),
                Item::Kernel(KernelDef {
                    name: "const_test".to_string(),
                    params: vec![],
                    body: Block { statements: vec![] },
                    attributes: vec![],
                }),
            ],
        };

        let wgsl = wgsl_from_ast(ast);

        assert!(
            wgsl.contains("const"),
            "Constant memory should use 'const' in WGSL. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 13: Rust transpiler generates proper imports
    // ---------------------------------------------------------------
    #[test]
    fn test_rust_transpiler_generates_imports() {
        let source = r#"
            __global__ void testKernel(float* data) {
                int idx = threadIdx.x;
            }
        "#;

        let rust_code = transpile_to_rust(source);

        assert!(
            rust_code.contains("use"),
            "Rust output should contain import statements. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 14: WGSL binary operator mapping
    // ---------------------------------------------------------------
    #[test]
    fn test_wgsl_binary_operator_mapping() {
        let ast = Ast {
            items: vec![
                Item::Kernel(KernelDef {
                    name: "op_test".to_string(),
                    params: vec![Parameter {
                        name: "data".to_string(),
                        ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                        qualifiers: vec![],
                    }],
                    body: Block {
                        statements: vec![
                            // a + b
                            Statement::Expr(Expression::Binary {
                                op: BinaryOp::Add,
                                left: Box::new(Expression::Var("a".to_string())),
                                right: Box::new(Expression::Var("b".to_string())),
                            }),
                            // a < b
                            Statement::Expr(Expression::Binary {
                                op: BinaryOp::Lt,
                                left: Box::new(Expression::Var("a".to_string())),
                                right: Box::new(Expression::Var("b".to_string())),
                            }),
                            // a && b
                            Statement::Expr(Expression::Binary {
                                op: BinaryOp::LogicalAnd,
                                left: Box::new(Expression::Var("a".to_string())),
                                right: Box::new(Expression::Var("b".to_string())),
                            }),
                        ],
                    },
                    attributes: vec![],
                }),
            ],
        };

        let wgsl = wgsl_from_ast(ast);

        assert!(wgsl.contains("+"), "WGSL should contain + operator. Got:\n{}", wgsl);
        assert!(wgsl.contains("<"), "WGSL should contain < operator. Got:\n{}", wgsl);
        assert!(wgsl.contains("&&"), "WGSL should contain && operator. Got:\n{}", wgsl);
    }

    // ---------------------------------------------------------------
    // Test 15: WGSL literal formatting
    // ---------------------------------------------------------------
    #[test]
    fn test_wgsl_literal_formatting() {
        let ast = Ast {
            items: vec![
                Item::Kernel(KernelDef {
                    name: "lit_test".to_string(),
                    params: vec![Parameter {
                        name: "data".to_string(),
                        ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                        qualifiers: vec![],
                    }],
                    body: Block {
                        statements: vec![
                            Statement::VarDecl {
                                name: "fval".to_string(),
                                ty: Type::Float(FloatType::F32),
                                init: Some(Expression::Literal(Literal::Float(3.14))),
                                storage: StorageClass::Auto,
                            },
                            Statement::VarDecl {
                                name: "ival".to_string(),
                                ty: Type::Int(IntType::I32),
                                init: Some(Expression::Literal(Literal::Int(42))),
                                storage: StorageClass::Auto,
                            },
                            Statement::VarDecl {
                                name: "uval".to_string(),
                                ty: Type::Int(IntType::U32),
                                init: Some(Expression::Literal(Literal::UInt(100))),
                                storage: StorageClass::Auto,
                            },
                            Statement::VarDecl {
                                name: "bval".to_string(),
                                ty: Type::Bool,
                                init: Some(Expression::Literal(Literal::Bool(true))),
                                storage: StorageClass::Auto,
                            },
                        ],
                    },
                    attributes: vec![],
                }),
            ],
        };

        let wgsl = wgsl_from_ast(ast);

        // Float literals should have 'f' suffix in WGSL
        assert!(wgsl.contains("3.14f"), "Float literal should have 'f' suffix. Got:\n{}", wgsl);
        // Int literals should have 'i' suffix
        assert!(wgsl.contains("42i"), "Int literal should have 'i' suffix. Got:\n{}", wgsl);
        // Uint literals should have 'u' suffix
        assert!(wgsl.contains("100u"), "Uint literal should have 'u' suffix. Got:\n{}", wgsl);
        // Bool literals
        assert!(wgsl.contains("true"), "Bool literal 'true' should be present. Got:\n{}", wgsl);
    }

    // ---------------------------------------------------------------
    // Test 16: CudaTranspiler high-level API
    // ---------------------------------------------------------------
    #[test]
    fn test_cuda_transpiler_high_level_api() {
        use cuda_rust_wasm::CudaTranspiler;

        let transpiler = CudaTranspiler::new();

        let source = r#"
            __global__ void myKernel(float* data, int n) {
                int idx = threadIdx.x;
                data[idx] = 0.0f;
            }
        "#;

        // Test transpile method
        let result = transpiler.transpile(source, false, false);
        assert!(result.is_ok(), "CudaTranspiler::transpile should succeed");
        let rust_code = result.unwrap();
        assert!(!rust_code.is_empty(), "Generated Rust code should not be empty");
    }

    // ---------------------------------------------------------------
    // Test 17: CudaRust high-level API
    // ---------------------------------------------------------------
    #[test]
    fn test_cuda_rust_high_level_api() {
        use cuda_rust_wasm::CudaRust;

        let cuda_rust = CudaRust::new();

        let source = r#"
            __global__ void add(float* a, float* b, float* c) {
                int i = threadIdx.x;
                c[i] = a[i] + b[i];
            }
        "#;

        let result = cuda_rust.transpile(source);
        assert!(result.is_ok(), "CudaRust::transpile should succeed");
    }

    // ---------------------------------------------------------------
    // Test 18: Transpiler handles while loop in WGSL
    // ---------------------------------------------------------------
    #[test]
    fn test_while_loop_wgsl() {
        let ast = Ast {
            items: vec![
                Item::Kernel(KernelDef {
                    name: "while_test".to_string(),
                    params: vec![Parameter {
                        name: "data".to_string(),
                        ty: Type::Pointer(Box::new(Type::Float(FloatType::F32))),
                        qualifiers: vec![],
                    }],
                    body: Block {
                        statements: vec![
                            Statement::While {
                                condition: Expression::Binary {
                                    op: BinaryOp::Lt,
                                    left: Box::new(Expression::Var("i".to_string())),
                                    right: Box::new(Expression::Literal(Literal::Int(100))),
                                },
                                body: Box::new(Statement::Expr(Expression::Binary {
                                    op: BinaryOp::Assign,
                                    left: Box::new(Expression::Var("i".to_string())),
                                    right: Box::new(Expression::Binary {
                                        op: BinaryOp::Add,
                                        left: Box::new(Expression::Var("i".to_string())),
                                        right: Box::new(Expression::Literal(Literal::Int(1))),
                                    }),
                                })),
                            },
                        ],
                    },
                    attributes: vec![],
                }),
            ],
        };

        let wgsl = wgsl_from_ast(ast);
        assert!(wgsl.contains("while"), "Should contain while loop. Got:\n{}", wgsl);
    }

    // ---------------------------------------------------------------
    // Test 19: Transpiler handles break and continue
    // ---------------------------------------------------------------
    #[test]
    fn test_break_continue_wgsl() {
        let ast = Ast {
            items: vec![
                Item::Kernel(KernelDef {
                    name: "flow_test".to_string(),
                    params: vec![],
                    body: Block {
                        statements: vec![
                            Statement::Break,
                            Statement::Continue,
                        ],
                    },
                    attributes: vec![],
                }),
            ],
        };

        let wgsl = wgsl_from_ast(ast);
        assert!(wgsl.contains("break;"), "Should contain break. Got:\n{}", wgsl);
        assert!(wgsl.contains("continue;"), "Should contain continue. Got:\n{}", wgsl);
    }

    // ---------------------------------------------------------------
    // Test 20: Transpiler handles return statement
    // ---------------------------------------------------------------
    #[test]
    fn test_return_statement_wgsl() {
        let ast = Ast {
            items: vec![
                Item::Kernel(KernelDef {
                    name: "return_test".to_string(),
                    params: vec![],
                    body: Block {
                        statements: vec![
                            Statement::Return(None),
                        ],
                    },
                    attributes: vec![],
                }),
            ],
        };

        let wgsl = wgsl_from_ast(ast);
        assert!(wgsl.contains("return"), "Should contain return statement. Got:\n{}", wgsl);
    }
}
