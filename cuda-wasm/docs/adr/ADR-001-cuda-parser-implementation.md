# ADR-001: Replace Hardcoded Parser Stub with Real nom/logos-Based CUDA C++ Parser

## Status

**Accepted**

Date: 2025-07-15

## Context

The cuda-rust-wasm project provides a CUDA-to-Rust/WGSL transpilation pipeline. The pipeline is structured as:

1. **Parser** (`src/parser/cuda_parser.rs`) -- parses CUDA C++ source into an AST
2. **Transpiler** (`src/transpiler/`) -- converts the AST into Rust or WGSL output
3. **Backend** (`src/backend/`) -- selects and executes on WebGPU, native GPU, or CPU

The current `CudaParser::parse()` implementation (in `src/parser/cuda_parser.rs`) **ignores its input entirely** and returns a hardcoded AST representing a single `vectorAdd` kernel:

```rust
pub fn parse(&self, source: &str) -> Result<Ast> {
    // TODO: Implement actual parsing logic
    // This is a stub implementation
    Ok(Ast {
        items: vec![
            Item::Kernel(KernelDef {
                name: "vectorAdd".to_string(),
                // ... hardcoded params and body ...
            }),
        ],
    })
}
```

This means:

- **No actual CUDA source code is parsed.** Every call to `CudaParser::parse()` returns the same AST regardless of input.
- **The transpiler pipeline is untestable** against real CUDA programs because the parser short-circuits all input.
- **The project already declares `nom = "7.1"` and `logos = "0.14"` as dependencies** in `Cargo.toml`, along with `lalrpop = "0.20"` as a build dependency, but none of these are used by the parser.
- **The AST type system** (`src/parser/ast.rs`) is already comprehensive, supporting kernels, device/host functions, global variables, type definitions, includes, warp primitives, storage classes, vector types, texture types, and a full expression/statement hierarchy.
- **A separate transpiler-level AST** exists in `src/transpiler/ast.rs` with its own `Program`, `Function`, `Stmt`, and `Expr` types, creating duplication.
- **A lexer module** exists at `src/parser/lexer.rs` but is currently empty.
- **A kernel extractor** exists at `src/parser/kernel_extractor.rs` but its integration with the main parse path is unclear.

The current test in `src/lib.rs` passes trivially because it never validates that the parsed AST matches the input:

```rust
#[test]
fn test_basic_transpilation() {
    let cuda_rust = CudaRust::new();
    let cuda_code = r#"
        __global__ void add(float* a, float* b, float* c) {
            int i = threadIdx.x;
            c[i] = a[i] + b[i];
        }
    "#;
    let result = cuda_rust.transpile(cuda_code);
    assert!(result.is_ok()); // Always passes; parser ignores input
}
```

## Decision

We will implement a real CUDA C++ parser using a two-phase architecture:

### Phase 1: Lexical Analysis (logos)

Implement a token lexer in `src/parser/lexer.rs` using the `logos` crate. The lexer will tokenize CUDA C++ source into a stream of typed tokens covering:

- **Keywords**: `__global__`, `__device__`, `__host__`, `__shared__`, `__constant__`, `void`, `int`, `float`, `double`, `char`, `unsigned`, `signed`, `long`, `short`, `bool`, `if`, `else`, `for`, `while`, `do`, `return`, `break`, `continue`, `struct`, `typedef`, `enum`, `const`, `volatile`, `restrict`, `static`, `extern`, `inline`
- **CUDA-specific tokens**: `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, `__syncthreads`, `atomicAdd`, `atomicCAS`, `atomicMin`, `atomicMax`, `__shfl_sync`, `__shfl_xor_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__ballot_sync`, `__activemask`
- **Operators**: all C++ arithmetic, logical, bitwise, comparison, assignment, and compound-assignment operators
- **Delimiters**: braces, parentheses, brackets, semicolons, commas, angle brackets (for templates)
- **Literals**: integer (decimal, hex, octal, binary), floating-point (with suffix), string, character
- **Identifiers**: standard C++ identifier rules
- **Comments**: line (`//`) and block (`/* */`)
- **Preprocessor directives**: `#include`, `#define`, `#ifdef`, `#ifndef`, `#endif`, `#pragma`
- **Launch syntax**: `<<<` and `>>>` for kernel launch configuration

### Phase 2: Syntactic Analysis (nom)

Implement a recursive descent parser in `src/parser/cuda_parser.rs` using `nom` parser combinators operating on the token stream from Phase 1. The parser will produce the existing `Ast` type from `src/parser/ast.rs`. Key grammar productions:

- **Translation unit**: sequence of top-level items (functions, variables, typedefs, includes)
- **Function declarations**: with `__global__`, `__device__`, `__host__` qualifiers; parameter lists with pointer, array, const, restrict qualifiers
- **Statements**: variable declarations (with storage class), expression statements, blocks, if/else, for, while, do-while, return, break, continue, `__syncthreads()`
- **Expressions**: full C-style expression grammar with correct operator precedence (assignment, ternary, logical-or, logical-and, bitwise-or, bitwise-xor, bitwise-and, equality, relational, shift, additive, multiplicative, unary, postfix, primary)
- **CUDA built-in expressions**: `threadIdx.x/y/z`, `blockIdx.x/y/z`, `blockDim.x/y/z`, `gridDim.x/y/z`
- **Warp primitives**: `__shfl_sync`, `__shfl_xor_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__ballot_sync`, `__activemask`
- **Kernel launch syntax**: `kernel<<<grid, block>>>(args)` parsed into a function call with launch configuration metadata
- **Type parsing**: primitive types, pointer types (including pointer-to-pointer), array types, CUDA vector types (`float4`, `int2`, etc.), `const`/`volatile` qualifiers
- **Struct definitions**: member declarations, nested structs
- **Template syntax**: basic template parameter recognition for common patterns (`thrust::device_vector<float>`)

### AST Consolidation

- Consolidate the duplicate AST types. The parser-level AST (`src/parser/ast.rs`) will be the single source of truth.
- The transpiler-level AST (`src/transpiler/ast.rs`) will be deprecated and its consumers migrated to use `parser::ast` types directly, or a well-defined transformation pass will be documented.

### Error Reporting

- Parser errors will include source location (line, column) and contextual information.
- The `CudaRustError` type already has a `ParseError(String)` variant; this will be extended with structured location data.

### Testing Strategy

- Unit tests for every grammar production in the nom parser.
- Integration tests parsing real CUDA kernels (vectorAdd, matrixMul, reduction, stencil, histogram) and verifying the resulting AST structure.
- Round-trip tests: parse CUDA, generate Rust/WGSL, and verify semantic equivalence on known inputs.
- Fuzz testing using `proptest` (already a dev-dependency) to generate random token sequences and verify the parser does not panic.
- The existing `tests/parser_tests.rs` will be expanded to cover all grammar rules.

## Consequences

### Positive

- **Enables actual CUDA-to-WASM transpilation.** Users can provide real CUDA source code and receive correct Rust or WGSL output.
- **Unlocks the full transpiler pipeline.** The code generator, WGSL generator, type converter, memory mapper, and built-in function resolver can all be validated against real inputs.
- **Leverages existing dependencies.** `nom` and `logos` are already declared in `Cargo.toml` and need no additional dependency management.
- **Preserves the existing AST type system.** The parser populates the same `Ast`, `KernelDef`, `Statement`, `Expression`, `Type`, etc. types that the rest of the pipeline already consumes.
- **Enables CI validation.** Automated tests can verify that specific CUDA patterns produce expected AST structures.

### Negative

- **Significant implementation effort.** A full CUDA C++ parser is complex; the C++ grammar alone has hundreds of productions, and CUDA extends it further.
- **Incomplete grammar coverage is expected initially.** Obscure C++ features (template metaprogramming, operator overloading, multiple inheritance) will not be supported in the first iteration. The parser will target the CUDA kernel subset of C++.
- **Potential parsing ambiguities.** C++ is notoriously context-sensitive (e.g., `A * B` could be a multiplication or a pointer declaration). The parser will use heuristics and context tracking for these cases.
- **Performance overhead.** A full recursive descent parser is slower than the current hardcoded stub, though this is acceptable since parsing is not the bottleneck in a transpilation pipeline.

### Risks

- **Grammar completeness.** Real-world CUDA code may use C++ features not covered by the initial grammar. The parser must emit clear error messages for unsupported constructs.
- **Dual AST maintenance.** If the transpiler AST is not consolidated, future changes must be kept in sync across two type hierarchies.

## References

- `src/parser/cuda_parser.rs` -- Current stub parser
- `src/parser/ast.rs` -- Parser-level AST types (337 lines, comprehensive)
- `src/transpiler/ast.rs` -- Transpiler-level AST types (65 lines, duplicates parser AST)
- `src/parser/lexer.rs` -- Empty lexer module
- `src/parser/kernel_extractor.rs` -- Kernel extraction utilities
- `Cargo.toml` -- Declares `nom = "7.1"`, `logos = "0.14"`, `lalrpop = "0.20"` (unused)
- `tests/parser_tests.rs` -- Existing parser test suite
