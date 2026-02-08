//! Tests for atomic operation fidelity
//!
//! These tests verify that CUDA atomic operations (atomicAdd, atomicCAS,
//! atomicMin, atomicMax, etc.) are correctly represented in the AST and
//! produce correct transpiled output in both Rust and WGSL.

#[cfg(test)]
mod tests {
    use cuda_rust_wasm::parser::ast::*;
    use cuda_rust_wasm::transpiler::Transpiler;
    use cuda_rust_wasm::transpiler::wgsl::WgslGenerator;

    // ---------------------------------------------------------------
    // Helper: build a kernel with an atomic call expression
    // ---------------------------------------------------------------
    fn kernel_with_atomic(name: &str, atomic_fn: &str, args: Vec<Expression>) -> Ast {
        Ast {
            items: vec![Item::Kernel(KernelDef {
                name: name.to_string(),
                params: vec![
                    Parameter {
                        name: "data".to_string(),
                        ty: Type::Pointer(Box::new(Type::Int(IntType::I32))),
                        qualifiers: vec![],
                    },
                    Parameter {
                        name: "result".to_string(),
                        ty: Type::Pointer(Box::new(Type::Int(IntType::I32))),
                        qualifiers: vec![],
                    },
                ],
                body: Block {
                    statements: vec![
                        Statement::VarDecl {
                            name: "idx".to_string(),
                            ty: Type::Int(IntType::I32),
                            init: Some(Expression::Binary {
                                op: BinaryOp::Add,
                                left: Box::new(Expression::Binary {
                                    op: BinaryOp::Mul,
                                    left: Box::new(Expression::BlockIdx(Dimension::X)),
                                    right: Box::new(Expression::BlockDim(Dimension::X)),
                                }),
                                right: Box::new(Expression::ThreadIdx(Dimension::X)),
                            }),
                            storage: StorageClass::Auto,
                        },
                        Statement::Expr(Expression::Call {
                            name: atomic_fn.to_string(),
                            args,
                        }),
                    ],
                },
                attributes: vec![],
            })],
        }
    }

    // ---------------------------------------------------------------
    // Test 1: atomicAdd mapping in Rust
    // ---------------------------------------------------------------
    #[test]
    fn test_atomic_add_to_rust() {
        let ast = kernel_with_atomic(
            "atomic_add_test",
            "atomicAdd",
            vec![
                Expression::Unary {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(Expression::Index {
                        array: Box::new(Expression::Var("result".to_string())),
                        index: Box::new(Expression::Literal(Literal::Int(0))),
                    }),
                },
                Expression::Literal(Literal::Int(1)),
            ],
        );

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile");

        // The generated code should contain the atomicAdd call
        assert!(
            rust_code.contains("atomicAdd"),
            "Rust output should contain atomicAdd call. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 2: atomicCAS mapping in Rust
    // ---------------------------------------------------------------
    #[test]
    fn test_atomic_cas_to_rust() {
        let ast = kernel_with_atomic(
            "atomic_cas_test",
            "atomicCAS",
            vec![
                Expression::Unary {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(Expression::Index {
                        array: Box::new(Expression::Var("data".to_string())),
                        index: Box::new(Expression::Var("idx".to_string())),
                    }),
                },
                Expression::Literal(Literal::Int(0)),   // expected
                Expression::Literal(Literal::Int(42)),   // desired
            ],
        );

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile");

        assert!(
            rust_code.contains("atomicCAS"),
            "Rust output should contain atomicCAS call. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 3: atomicMin mapping in Rust
    // ---------------------------------------------------------------
    #[test]
    fn test_atomic_min_to_rust() {
        let ast = kernel_with_atomic(
            "atomic_min_test",
            "atomicMin",
            vec![
                Expression::Unary {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(Expression::Index {
                        array: Box::new(Expression::Var("result".to_string())),
                        index: Box::new(Expression::Literal(Literal::Int(0))),
                    }),
                },
                Expression::Index {
                    array: Box::new(Expression::Var("data".to_string())),
                    index: Box::new(Expression::Var("idx".to_string())),
                },
            ],
        );

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile");

        assert!(
            rust_code.contains("atomicMin"),
            "Rust output should contain atomicMin call. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 4: atomicMax mapping in Rust
    // ---------------------------------------------------------------
    #[test]
    fn test_atomic_max_to_rust() {
        let ast = kernel_with_atomic(
            "atomic_max_test",
            "atomicMax",
            vec![
                Expression::Unary {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(Expression::Index {
                        array: Box::new(Expression::Var("result".to_string())),
                        index: Box::new(Expression::Literal(Literal::Int(0))),
                    }),
                },
                Expression::Index {
                    array: Box::new(Expression::Var("data".to_string())),
                    index: Box::new(Expression::Var("idx".to_string())),
                },
            ],
        );

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile");

        assert!(
            rust_code.contains("atomicMax"),
            "Rust output should contain atomicMax call. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 5: atomicAdd in WGSL output
    // ---------------------------------------------------------------
    #[test]
    fn test_atomic_add_to_wgsl() {
        let ast = kernel_with_atomic(
            "atomic_add_wgsl",
            "atomicAdd",
            vec![
                Expression::Unary {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(Expression::Index {
                        array: Box::new(Expression::Var("result".to_string())),
                        index: Box::new(Expression::Literal(Literal::Int(0))),
                    }),
                },
                Expression::Literal(Literal::Int(1)),
            ],
        );

        let mut gen = WgslGenerator::new();
        let wgsl = gen.generate(ast).expect("Failed to generate WGSL");

        // In WGSL, atomicAdd is a function call; verify it appears in output
        assert!(
            wgsl.contains("atomicAdd"),
            "WGSL output should contain atomicAdd. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 6: atomicCAS in WGSL output
    // ---------------------------------------------------------------
    #[test]
    fn test_atomic_cas_to_wgsl() {
        let ast = kernel_with_atomic(
            "atomic_cas_wgsl",
            "atomicCAS",
            vec![
                Expression::Unary {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(Expression::Index {
                        array: Box::new(Expression::Var("data".to_string())),
                        index: Box::new(Expression::Literal(Literal::Int(0))),
                    }),
                },
                Expression::Literal(Literal::Int(0)),
                Expression::Literal(Literal::Int(1)),
            ],
        );

        let mut gen = WgslGenerator::new();
        let wgsl = gen.generate(ast).expect("Failed to generate WGSL");

        assert!(
            wgsl.contains("atomicCAS") || wgsl.contains("atomicCompareExchangeWeak"),
            "WGSL output should contain atomic CAS equivalent. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 7: atomicMin in WGSL output
    // ---------------------------------------------------------------
    #[test]
    fn test_atomic_min_to_wgsl() {
        let ast = kernel_with_atomic(
            "atomic_min_wgsl",
            "atomicMin",
            vec![
                Expression::Unary {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(Expression::Index {
                        array: Box::new(Expression::Var("result".to_string())),
                        index: Box::new(Expression::Literal(Literal::Int(0))),
                    }),
                },
                Expression::Literal(Literal::Int(42)),
            ],
        );

        let mut gen = WgslGenerator::new();
        let wgsl = gen.generate(ast).expect("Failed to generate WGSL");

        assert!(
            wgsl.contains("atomicMin"),
            "WGSL output should contain atomicMin. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 8: atomicMax in WGSL output
    // ---------------------------------------------------------------
    #[test]
    fn test_atomic_max_to_wgsl() {
        let ast = kernel_with_atomic(
            "atomic_max_wgsl",
            "atomicMax",
            vec![
                Expression::Unary {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(Expression::Index {
                        array: Box::new(Expression::Var("result".to_string())),
                        index: Box::new(Expression::Literal(Literal::Int(0))),
                    }),
                },
                Expression::Literal(Literal::Int(42)),
            ],
        );

        let mut gen = WgslGenerator::new();
        let wgsl = gen.generate(ast).expect("Failed to generate WGSL");

        assert!(
            wgsl.contains("atomicMax"),
            "WGSL output should contain atomicMax. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 9: Atomic operation within if guard
    // ---------------------------------------------------------------
    #[test]
    fn test_atomic_within_bounds_check() {
        let ast = Ast {
            items: vec![Item::Kernel(KernelDef {
                name: "guarded_atomic".to_string(),
                params: vec![
                    Parameter {
                        name: "bins".to_string(),
                        ty: Type::Pointer(Box::new(Type::Int(IntType::I32))),
                        qualifiers: vec![],
                    },
                    Parameter {
                        name: "n".to_string(),
                        ty: Type::Int(IntType::I32),
                        qualifiers: vec![],
                    },
                ],
                body: Block {
                    statements: vec![
                        Statement::VarDecl {
                            name: "idx".to_string(),
                            ty: Type::Int(IntType::I32),
                            init: Some(Expression::ThreadIdx(Dimension::X)),
                            storage: StorageClass::Auto,
                        },
                        Statement::If {
                            condition: Expression::Binary {
                                op: BinaryOp::Lt,
                                left: Box::new(Expression::Var("idx".to_string())),
                                right: Box::new(Expression::Var("n".to_string())),
                            },
                            then_branch: Box::new(Statement::Expr(Expression::Call {
                                name: "atomicAdd".to_string(),
                                args: vec![
                                    Expression::Unary {
                                        op: UnaryOp::AddrOf,
                                        expr: Box::new(Expression::Index {
                                            array: Box::new(Expression::Var("bins".to_string())),
                                            index: Box::new(Expression::Var("idx".to_string())),
                                        }),
                                    },
                                    Expression::Literal(Literal::Int(1)),
                                ],
                            })),
                            else_branch: None,
                        },
                    ],
                },
                attributes: vec![],
            })],
        };

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast.clone()).expect("Failed to transpile");

        // Should contain both the bounds check and the atomic
        assert!(
            rust_code.contains("if") && rust_code.contains("atomicAdd"),
            "Should contain bounds check and atomicAdd. Got:\n{}",
            rust_code
        );

        let mut gen = WgslGenerator::new();
        let wgsl = gen.generate(ast).expect("Failed to generate WGSL");

        assert!(
            wgsl.contains("if") && wgsl.contains("atomicAdd"),
            "WGSL should contain bounds check and atomicAdd. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 10: Multiple atomics in one kernel
    // ---------------------------------------------------------------
    #[test]
    fn test_multiple_atomics_in_kernel() {
        let ast = Ast {
            items: vec![Item::Kernel(KernelDef {
                name: "multi_atomic".to_string(),
                params: vec![
                    Parameter {
                        name: "data".to_string(),
                        ty: Type::Pointer(Box::new(Type::Int(IntType::I32))),
                        qualifiers: vec![],
                    },
                ],
                body: Block {
                    statements: vec![
                        Statement::Expr(Expression::Call {
                            name: "atomicAdd".to_string(),
                            args: vec![
                                Expression::Var("data".to_string()),
                                Expression::Literal(Literal::Int(1)),
                            ],
                        }),
                        Statement::Expr(Expression::Call {
                            name: "atomicMin".to_string(),
                            args: vec![
                                Expression::Var("data".to_string()),
                                Expression::Literal(Literal::Int(0)),
                            ],
                        }),
                        Statement::Expr(Expression::Call {
                            name: "atomicMax".to_string(),
                            args: vec![
                                Expression::Var("data".to_string()),
                                Expression::Literal(Literal::Int(100)),
                            ],
                        }),
                    ],
                },
                attributes: vec![],
            })],
        };

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile");

        assert!(rust_code.contains("atomicAdd"), "Should contain atomicAdd");
        assert!(rust_code.contains("atomicMin"), "Should contain atomicMin");
        assert!(rust_code.contains("atomicMax"), "Should contain atomicMax");
    }
}
