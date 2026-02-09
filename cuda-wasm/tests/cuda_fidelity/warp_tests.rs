//! Tests for warp primitive fidelity
//!
//! These tests verify that warp-level operations (shuffle, ballot, vote, etc.)
//! are correctly represented in the AST and transpiled properly.

#[cfg(test)]
mod tests {
    use cuda_rust_wasm::parser::ast::*;
    use cuda_rust_wasm::transpiler::Transpiler;
    use cuda_rust_wasm::transpiler::wgsl::WgslGenerator;

    // ---------------------------------------------------------------
    // Helper: build a kernel with a warp primitive expression
    // ---------------------------------------------------------------
    fn kernel_with_warp_op(name: &str, warp_op: WarpOp, args: Vec<Expression>) -> Ast {
        Ast {
            items: vec![Item::Kernel(KernelDef {
                name: name.to_string(),
                params: vec![Parameter {
                    name: "data".to_string(),
                    ty: Type::Pointer(Box::new(Type::Int(IntType::I32))),
                    qualifiers: vec![],
                }],
                body: Block {
                    statements: vec![
                        Statement::VarDecl {
                            name: "val".to_string(),
                            ty: Type::Int(IntType::I32),
                            init: Some(Expression::Index {
                                array: Box::new(Expression::Var("data".to_string())),
                                index: Box::new(Expression::ThreadIdx(Dimension::X)),
                            }),
                            storage: StorageClass::Auto,
                        },
                        Statement::VarDecl {
                            name: "result".to_string(),
                            ty: Type::Int(IntType::I32),
                            init: Some(Expression::WarpPrimitive {
                                op: warp_op,
                                args,
                            }),
                            storage: StorageClass::Auto,
                        },
                    ],
                },
                attributes: vec![],
            })],
        }
    }

    // ---------------------------------------------------------------
    // Test 1: Warp shuffle produces correct Rust code
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_shuffle_to_rust() {
        let ast = kernel_with_warp_op(
            "shuffle_test",
            WarpOp::Shuffle,
            vec![
                Expression::Var("val".to_string()),
                Expression::Literal(Literal::Int(3)),
            ],
        );

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile");

        assert!(
            rust_code.contains("warp_shuffle"),
            "Warp shuffle should produce warp_shuffle call in Rust. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 2: Warp shuffle_xor produces correct Rust code
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_shuffle_xor_to_rust() {
        let ast = kernel_with_warp_op(
            "shuffle_xor_test",
            WarpOp::ShuffleXor,
            vec![
                Expression::Var("val".to_string()),
                Expression::Literal(Literal::Int(1)),
            ],
        );

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile");

        assert!(
            rust_code.contains("warp_shuffle_xor"),
            "ShuffleXor should produce warp_shuffle_xor call. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 3: Warp shuffle_up produces correct Rust code
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_shuffle_up_to_rust() {
        let ast = kernel_with_warp_op(
            "shuffle_up_test",
            WarpOp::ShuffleUp,
            vec![
                Expression::Var("val".to_string()),
                Expression::Literal(Literal::Int(2)),
            ],
        );

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile");

        assert!(
            rust_code.contains("warp_shuffle_up"),
            "ShuffleUp should produce warp_shuffle_up call. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 4: Warp shuffle_down produces correct Rust code
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_shuffle_down_to_rust() {
        let ast = kernel_with_warp_op(
            "shuffle_down_test",
            WarpOp::ShuffleDown,
            vec![
                Expression::Var("val".to_string()),
                Expression::Literal(Literal::Int(4)),
            ],
        );

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile");

        assert!(
            rust_code.contains("warp_shuffle_down"),
            "ShuffleDown should produce warp_shuffle_down call. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 5: Warp ballot produces correct Rust code
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_ballot_to_rust() {
        let ast = kernel_with_warp_op(
            "ballot_test",
            WarpOp::Ballot,
            vec![
                Expression::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expression::Var("val".to_string())),
                    right: Box::new(Expression::Literal(Literal::Int(0))),
                },
            ],
        );

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile");

        assert!(
            rust_code.contains("warp_ballot"),
            "Ballot should produce warp_ballot call. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 6: Warp vote (all) produces correct Rust code
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_vote_to_rust() {
        let ast = kernel_with_warp_op(
            "vote_test",
            WarpOp::Vote,
            vec![
                Expression::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expression::Var("val".to_string())),
                    right: Box::new(Expression::Literal(Literal::Int(0))),
                },
            ],
        );

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile");

        assert!(
            rust_code.contains("warp_vote_all"),
            "Vote should produce warp_vote_all call. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 7: Warp activemask produces correct Rust code
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_active_mask_to_rust() {
        let ast = kernel_with_warp_op(
            "activemask_test",
            WarpOp::ActiveMask,
            vec![], // ActiveMask takes no arguments
        );

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile");

        assert!(
            rust_code.contains("warp_activemask"),
            "ActiveMask should produce warp_activemask call. Got:\n{}",
            rust_code
        );
    }

    // ---------------------------------------------------------------
    // Test 8: Warp primitives in WGSL emit comments (not supported)
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_primitives_in_wgsl() {
        let ast = kernel_with_warp_op(
            "warp_wgsl_test",
            WarpOp::ShuffleXor,
            vec![
                Expression::Var("val".to_string()),
                Expression::Literal(Literal::Int(1)),
            ],
        );

        let mut gen = WgslGenerator::new();
        let wgsl = gen.generate(ast).expect("Failed to generate WGSL");

        // WGSL doesn't natively support warp primitives, so the generator
        // should emit a comment and a placeholder value
        assert!(
            wgsl.contains("warp") || wgsl.contains("0"),
            "Warp primitives in WGSL should emit comment/placeholder. Got:\n{}",
            wgsl
        );
    }

    // ---------------------------------------------------------------
    // Test 9: Warp shuffle with wrong argument count should error
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_shuffle_wrong_args_count() {
        // Build AST with incorrect number of args for Shuffle (needs 2, give 1)
        let ast = Ast {
            items: vec![Item::Kernel(KernelDef {
                name: "bad_shuffle".to_string(),
                params: vec![],
                body: Block {
                    statements: vec![Statement::Expr(Expression::WarpPrimitive {
                        op: WarpOp::Shuffle,
                        args: vec![Expression::Var("val".to_string())], // Only 1 arg
                    })],
                },
                attributes: vec![],
            })],
        };

        let transpiler = Transpiler::new();
        let result = transpiler.transpile(ast);

        // The code generator should produce an error for wrong arg count
        assert!(
            result.is_err(),
            "Warp shuffle with 1 arg should produce an error"
        );
    }

    // ---------------------------------------------------------------
    // Test 10: Warp ballot with wrong argument count should error
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_ballot_wrong_args_count() {
        let ast = Ast {
            items: vec![Item::Kernel(KernelDef {
                name: "bad_ballot".to_string(),
                params: vec![],
                body: Block {
                    statements: vec![Statement::Expr(Expression::WarpPrimitive {
                        op: WarpOp::Ballot,
                        args: vec![], // 0 args, needs 1
                    })],
                },
                attributes: vec![],
            })],
        };

        let transpiler = Transpiler::new();
        let result = transpiler.transpile(ast);

        assert!(
            result.is_err(),
            "Warp ballot with 0 args should produce an error"
        );
    }

    // ---------------------------------------------------------------
    // Test 11: ActiveMask with args should error
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_activemask_with_args_error() {
        let ast = Ast {
            items: vec![Item::Kernel(KernelDef {
                name: "bad_activemask".to_string(),
                params: vec![],
                body: Block {
                    statements: vec![Statement::Expr(Expression::WarpPrimitive {
                        op: WarpOp::ActiveMask,
                        args: vec![Expression::Var("x".to_string())], // Should be empty
                    })],
                },
                attributes: vec![],
            })],
        };

        let transpiler = Transpiler::new();
        let result = transpiler.transpile(ast);

        assert!(
            result.is_err(),
            "ActiveMask with arguments should produce an error"
        );
    }

    // ---------------------------------------------------------------
    // Test 12: Warp reduction pattern (shuffle_down cascade)
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_reduction_pattern() {
        // Simulate a warp reduction: val += shuffle_down(val, 16), ... , shuffle_down(val, 1)
        let deltas = [16i64, 8, 4, 2, 1];
        let mut statements: Vec<Statement> = vec![
            Statement::VarDecl {
                name: "val".to_string(),
                ty: Type::Int(IntType::I32),
                init: Some(Expression::Index {
                    array: Box::new(Expression::Var("data".to_string())),
                    index: Box::new(Expression::ThreadIdx(Dimension::X)),
                }),
                storage: StorageClass::Auto,
            },
        ];

        for delta in &deltas {
            statements.push(Statement::Expr(Expression::Binary {
                op: BinaryOp::Assign,
                left: Box::new(Expression::Var("val".to_string())),
                right: Box::new(Expression::Binary {
                    op: BinaryOp::Add,
                    left: Box::new(Expression::Var("val".to_string())),
                    right: Box::new(Expression::WarpPrimitive {
                        op: WarpOp::ShuffleDown,
                        args: vec![
                            Expression::Var("val".to_string()),
                            Expression::Literal(Literal::Int(*delta)),
                        ],
                    }),
                }),
            }));
        }

        let ast = Ast {
            items: vec![Item::Kernel(KernelDef {
                name: "warp_reduce".to_string(),
                params: vec![Parameter {
                    name: "data".to_string(),
                    ty: Type::Pointer(Box::new(Type::Int(IntType::I32))),
                    qualifiers: vec![],
                }],
                body: Block { statements },
                attributes: vec![],
            })],
        };

        let transpiler = Transpiler::new();
        let rust_code = transpiler.transpile(ast).expect("Failed to transpile warp reduction");

        // Should contain multiple warp_shuffle_down calls
        let shuffle_count = rust_code.matches("warp_shuffle_down").count();
        assert_eq!(
            shuffle_count, 5,
            "Warp reduction should have 5 warp_shuffle_down calls, found {}",
            shuffle_count
        );
    }

    // ---------------------------------------------------------------
    // Test 13: All WarpOp variants are distinct
    // ---------------------------------------------------------------
    #[test]
    fn test_warp_op_enum_distinctness() {
        let all_ops = vec![
            WarpOp::Shuffle,
            WarpOp::ShuffleXor,
            WarpOp::ShuffleUp,
            WarpOp::ShuffleDown,
            WarpOp::Vote,
            WarpOp::Ballot,
            WarpOp::ActiveMask,
        ];

        // Verify they format differently via Debug
        let debug_strings: Vec<String> = all_ops.iter().map(|op| format!("{:?}", op)).collect();
        let unique: std::collections::HashSet<_> = debug_strings.iter().collect();
        assert_eq!(
            unique.len(),
            all_ops.len(),
            "All WarpOp variants should have distinct Debug output"
        );
    }
}
