# ADR-006: Implement Atomic Operations Support in AST, Parser, and All Code Generators

## Status

**Accepted**

Date: 2025-07-15

## Context

Atomic operations are fundamental to GPU programming. They provide thread-safe read-modify-write semantics for shared data structures and are essential for:

- **Histograms**: `atomicAdd(&hist[bin], 1)` to count occurrences
- **Counters**: `atomicAdd(&count, 1)` for global work counters
- **Reductions**: `atomicMin`/`atomicMax` for finding extrema
- **Locks and synchronization**: `atomicCAS` for implementing spin locks and lock-free data structures
- **Scatter operations**: Thread-safe writes to arbitrary locations
- **Neural network training**: Gradient accumulation via `atomicAdd`

CUDA provides the following atomic operations:

| CUDA Function | Description | Types |
|--------------|-------------|-------|
| `atomicAdd(addr, val)` | `*addr += val; return old` | `int`, `unsigned int`, `float`, `double` |
| `atomicSub(addr, val)` | `*addr -= val; return old` | `int`, `unsigned int` |
| `atomicExch(addr, val)` | `*addr = val; return old` | `int`, `unsigned int`, `float` |
| `atomicMin(addr, val)` | `*addr = min(*addr, val); return old` | `int`, `unsigned int` |
| `atomicMax(addr, val)` | `*addr = max(*addr, val); return old` | `int`, `unsigned int` |
| `atomicAnd(addr, val)` | `*addr &= val; return old` | `int`, `unsigned int` |
| `atomicOr(addr, val)` | `*addr |= val; return old` | `int`, `unsigned int` |
| `atomicXor(addr, val)` | `*addr ^= val; return old` | `int`, `unsigned int` |
| `atomicCAS(addr, compare, val)` | CAS operation, return old | `int`, `unsigned int`, `unsigned long long` |
| `atomicInc(addr, val)` | Increment with wrap | `unsigned int` |
| `atomicDec(addr, val)` | Decrement with wrap | `unsigned int` |

### Current State in cuda-rust-wasm

**Parser AST (`src/parser/ast.rs`)**: No atomic operation types exist. The `Expression` enum has no variant for atomic operations. Atomic calls would be parsed as generic `Expression::Call` nodes, losing their atomic semantics.

**WGSL Generator (`src/transpiler/wgsl.rs`)**: No handling for atomic operations. A generic function call like `atomicAdd(addr, val)` would be emitted as-is, which is invalid WGSL.

**Rust Code Generator (`src/transpiler/code_generator.rs`)**: No atomic-aware code generation. Generic function calls would be emitted without the necessary `std::sync::atomic` mappings.

**Builtin Functions (`src/transpiler/builtin_functions.rs`)**: Unknown current state, but no atomic function mappings are documented.

WGSL does support atomic operations natively:

| WGSL Function | CUDA Equivalent |
|---------------|-----------------|
| `atomicAdd(&val, n)` | `atomicAdd(&val, n)` |
| `atomicSub(&val, n)` | `atomicSub(&val, n)` |
| `atomicMin(&val, n)` | `atomicMin(&val, n)` |
| `atomicMax(&val, n)` | `atomicMax(&val, n)` |
| `atomicAnd(&val, n)` | `atomicAnd(&val, n)` |
| `atomicOr(&val, n)` | `atomicOr(&val, n)` |
| `atomicXor(&val, n)` | `atomicXor(&val, n)` |
| `atomicExchange(&val, n)` | `atomicExch(&val, n)` |
| `atomicCompareExchangeWeak(&val, cmp, n)` | `atomicCAS(&val, cmp, n)` |
| `atomicLoad(&val)` | (implicit via read) |
| `atomicStore(&val, n)` | (implicit via write) |

WGSL atomics operate on `atomic<i32>` and `atomic<u32>` types only. There is no `atomic<f32>` in WGSL -- floating-point atomics must be emulated using `atomicCompareExchangeWeak` with bitcasting.

## Decision

We will add comprehensive atomic operations support across the AST, parser, and all code generators.

### 1. AST Extension (`src/parser/ast.rs`)

Add a new `AtomicOp` enum and an `AtomicOperation` expression variant:

```rust
/// Atomic operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AtomicOp {
    Add,
    Sub,
    Exch,
    Min,
    Max,
    And,
    Or,
    Xor,
    CAS,
    Inc,
    Dec,
    Load,
    Store,
}

/// Memory ordering for atomic operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOrder {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

// Add to Expression enum:
pub enum Expression {
    // ... existing variants ...

    /// Atomic operation
    AtomicOperation {
        op: AtomicOp,
        /// Address operand (pointer to atomic variable)
        address: Box<Expression>,
        /// Value operand (for Add, Sub, Exch, Min, Max, And, Or, Xor, Store)
        value: Option<Box<Expression>>,
        /// Compare operand (for CAS only)
        compare: Option<Box<Expression>>,
        /// Memory ordering
        ordering: MemoryOrder,
    },
}
```

### 2. Parser Support

The parser (once ADR-001 is implemented) will recognize atomic function calls and produce `AtomicOperation` expression nodes:

| CUDA Source | Parsed AST |
|-------------|------------|
| `atomicAdd(&x, 1)` | `AtomicOperation { op: Add, address: AddrOf(x), value: Some(1), .. }` |
| `atomicCAS(&x, old, new)` | `AtomicOperation { op: CAS, address: AddrOf(x), compare: Some(old), value: Some(new), .. }` |
| `atomicExch(&x, val)` | `AtomicOperation { op: Exch, address: AddrOf(x), value: Some(val), .. }` |

Until ADR-001 is implemented, the current hardcoded parser will not produce atomic operations. However, the AST types and code generators will be ready.

### 3. WGSL Code Generation

The WGSL generator will map atomic operations to WGSL builtins:

```rust
Expression::AtomicOperation { op, address, value, compare, .. } => {
    match op {
        AtomicOp::Add => {
            self.write("atomicAdd(")?;
            self.generate_expression(address)?;
            self.write(", ")?;
            self.generate_expression(value.as_ref().unwrap())?;
            self.write(")")?;
        },
        AtomicOp::CAS => {
            self.write("atomicCompareExchangeWeak(")?;
            self.generate_expression(address)?;
            self.write(", ")?;
            self.generate_expression(compare.as_ref().unwrap())?;
            self.write(", ")?;
            self.generate_expression(value.as_ref().unwrap())?;
            self.write(").old_value")?;
        },
        AtomicOp::Exch => {
            self.write("atomicExchange(")?;
            self.generate_expression(address)?;
            self.write(", ")?;
            self.generate_expression(value.as_ref().unwrap())?;
            self.write(")")?;
        },
        // ... similar for Min, Max, And, Or, Xor, Sub
    }
}
```

#### Floating-Point Atomic Emulation in WGSL

Since WGSL lacks `atomic<f32>`, floating-point `atomicAdd` will be emulated:

```wgsl
fn atomicAddFloat(addr: ptr<storage, atomic<u32>, read_write>, val: f32) -> f32 {
    var old_val: u32;
    var new_val: u32;
    loop {
        old_val = atomicLoad(addr);
        let old_f32 = bitcast<f32>(old_val);
        let new_f32 = old_f32 + val;
        new_val = bitcast<u32>(new_f32);
        let result = atomicCompareExchangeWeak(addr, old_val, new_val);
        if (result.exchanged) {
            return old_f32;
        }
    }
}
```

The WGSL generator will detect when `atomicAdd` is applied to a floating-point type and emit this helper function.

#### Variable Type Transformation

WGSL requires atomic variables to be declared with `atomic<T>` types. The WGSL generator will:

1. Track which variables are targets of atomic operations during AST traversal
2. Transform their type declarations from `i32`/`u32` to `atomic<i32>`/`atomic<u32>`
3. Ensure buffer bindings use the correct atomic types

### 4. Rust Code Generation

The Rust code generator will map atomics to `std::sync::atomic`:

```rust
AtomicOp::Add => {
    quote! {
        std::sync::atomic::AtomicI32::from_ptr(#address)
            .fetch_add(#value, std::sync::atomic::Ordering::Relaxed)
    }
}

AtomicOp::CAS => {
    quote! {
        std::sync::atomic::AtomicI32::from_ptr(#address)
            .compare_exchange(#compare, #value, std::sync::atomic::Ordering::Relaxed,
                             std::sync::atomic::Ordering::Relaxed)
            .unwrap_or_else(|old| old)
    }
}
```

For `f32` atomics on CPU, the code generator will use `AtomicU32` with `f32::to_bits()` / `f32::from_bits()` in a CAS loop, mirroring the WGSL emulation.

### 5. Memory Ordering

CUDA atomics do not specify memory ordering (they are implicitly sequentially consistent within a thread block). The transpilation will use:

- **WGSL**: No ordering specification needed (WGSL atomics are sequentially consistent)
- **Rust**: `Ordering::Relaxed` by default, with an option to upgrade to `SeqCst` for correctness-critical code

## Consequences

### Positive

- **Full CUDA atomic fidelity.** All CUDA atomic operations will be correctly represented in the AST and transpiled to semantically equivalent code on all backends.
- **Direct WGSL mapping.** 8 of 10 CUDA atomic operations have direct WGSL equivalents (`atomicAdd`, `atomicSub`, `atomicMin`, `atomicMax`, `atomicAnd`, `atomicOr`, `atomicXor`, `atomicExchange`). The mapping is straightforward.
- **Histogram and counter kernels work.** Common GPU patterns (histograms, global counters, reduction via atomics) will produce correct results.
- **Type-safe AST representation.** Atomic operations as a distinct AST variant (rather than generic function calls) enables type checking, optimization passes, and backend-specific code generation.
- **Foundation for lock-free algorithms.** `atomicCAS` enables transpilation of lock-free data structures (queues, stacks, hash maps) from CUDA to WGSL.

### Negative

- **Floating-point atomic overhead.** The CAS-loop emulation for `atomicAdd` on `f32` has significant overhead under contention. NVIDIA GPUs have hardware `atomicAdd` for `f32`; the emulated version may be 10-100x slower.
- **No 64-bit atomics in WGSL.** CUDA `atomicAdd` on `double` and `atomicCAS` on `unsigned long long` cannot be directly transpiled to WGSL, which only supports 32-bit atomics. These operations will require error reporting or multi-word emulation.
- **AST size increase.** Adding `AtomicOperation` to the `Expression` enum increases the match-arm count in every code generator by one.
- **Variable type transformation complexity.** The WGSL generator must perform a pre-pass to identify atomic variable usage and transform type declarations, adding complexity to the code generation pipeline.

### Risks

- **Correctness under contention.** The CAS-loop for floating-point atomics can suffer from starvation under high contention (many threads atomically adding to the same address). Performance testing under contention is required.
- **WGSL `atomicCompareExchangeWeak` semantics.** The "weak" variant may spuriously fail, requiring a retry loop. This matches the CUDA `atomicCAS` semantics (which is also a CAS), but the retry loop overhead must be considered.
- **`atomicInc` and `atomicDec` emulation.** CUDA's `atomicInc(addr, val)` performs `*addr = (*addr >= val) ? 0 : *addr + 1`. This wrapping increment has no direct WGSL equivalent and must be emulated with `atomicCompareExchangeWeak` in a loop.

## References

- `src/parser/ast.rs` -- AST types (no atomic operations currently)
- `src/transpiler/wgsl.rs` -- WGSL code generator (no atomic handling)
- `src/transpiler/code_generator.rs` -- Rust code generator (no atomic handling)
- `src/transpiler/builtin_functions.rs` -- Built-in function mappings
- CUDA Atomic Functions: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
- WGSL Atomic Built-in Functions: https://www.w3.org/TR/WGSL/#atomic-builtin-functions
- Rust `std::sync::atomic`: https://doc.rust-lang.org/std/sync/atomic/
