# ADR-003: Implement Warp Primitive Emulation for Non-CUDA Targets

## Status

**Accepted**

Date: 2025-07-15

## Context

CUDA warp primitives are a class of operations that enable communication and synchronization between threads within a warp (a group of 32 threads that execute in lockstep on NVIDIA GPUs). These primitives are heavily used in performance-critical GPU code for reductions, scans, sorting, and communication patterns that avoid shared memory overhead.

The cuda-rust-wasm project's AST (`src/parser/ast.rs`) already defines warp operations:

```rust
pub enum WarpOp {
    Shuffle,       // __shfl_sync
    ShuffleXor,    // __shfl_xor_sync
    ShuffleUp,     // __shfl_up_sync
    ShuffleDown,   // __shfl_down_sync
    Vote,          // __all_sync / __any_sync
    Ballot,        // __ballot_sync
    ActiveMask,    // __activemask
}
```

And warp primitives appear as AST expressions:

```rust
pub enum Expression {
    // ...
    WarpPrimitive {
        op: WarpOp,
        args: Vec<Expression>,
    },
}
```

However, the current code generation for warp primitives is broken or incomplete across all backends:

### WGSL Generator (`src/transpiler/wgsl.rs`, lines 377-389)

The WGSL generator emits a comment and a hardcoded `0` for all warp primitives:

```rust
Expression::WarpPrimitive { op, args } => {
    self.write(&format!("/* warp_{op:?}("))?;
    for (i, arg) in args.iter().enumerate() {
        if i > 0 { self.write(", ")?; }
        self.generate_expression(arg)?;
    }
    self.write(") */")?;
    self.write("0")?;  // Always returns 0!
},
```

This means any CUDA kernel using warp shuffle for reductions (a very common pattern) will produce silently incorrect results when transpiled to WGSL.

### Rust Code Generator (`src/transpiler/code_generator.rs`, lines 377-432)

The Rust code generator emits calls to `cuda_rust_wasm::runtime::warp_shuffle()`, `warp_shuffle_xor()`, etc., but these functions are not implemented in the runtime module. The `src/runtime/mod.rs` does not export any warp-related functions.

### Kernel Warp Module (`src/kernel/warp.rs`)

The kernel warp module exists but is empty (1 line), providing no warp emulation logic.

### Why This Matters

Warp shuffle is the standard mechanism for:

- **Warp-level reductions**: summing values across a warp without shared memory (e.g., parallel reduction final stage)
- **Warp-level scans**: prefix sums within a warp for stream compaction
- **Histogram computation**: combining partial histograms
- **Parallel sorting**: bitonic sort communication pattern via `__shfl_xor_sync`
- **Stencil operations**: exchanging boundary values between adjacent threads
- **Neural network inference**: fast reduction for dot products, softmax denominators

Without correct warp emulation, a significant class of optimized CUDA kernels will produce incorrect results when transpiled.

## Decision

We will implement warp primitive emulation for all non-CUDA targets using target-appropriate mechanisms.

### WGSL Backend: Workgroup Shared Memory Emulation

WGSL has no warp-level primitives. We will emulate warp semantics using workgroup shared memory and barriers:

```wgsl
// Emulated warp shuffle using shared memory
var<workgroup> warp_scratch: array<f32, 32>;

fn warp_shuffle(value: f32, src_lane: u32, lane_id: u32) -> f32 {
    warp_scratch[lane_id] = value;
    workgroupBarrier();
    let result = warp_scratch[src_lane % 32u];
    workgroupBarrier();
    return result;
}

fn warp_shuffle_xor(value: f32, mask: u32, lane_id: u32) -> f32 {
    warp_scratch[lane_id] = value;
    workgroupBarrier();
    let src_lane = lane_id ^ mask;
    let result = warp_scratch[src_lane % 32u];
    workgroupBarrier();
    return result;
}

fn warp_shuffle_down(value: f32, delta: u32, lane_id: u32) -> f32 {
    warp_scratch[lane_id] = value;
    workgroupBarrier();
    let src_lane = lane_id + delta;
    let result = select(value, warp_scratch[src_lane], src_lane < 32u);
    workgroupBarrier();
    return result;
}

fn warp_ballot(predicate: bool, lane_id: u32) -> u32 {
    var<workgroup> ballot_scratch: array<u32, 32>;
    ballot_scratch[lane_id] = select(0u, 1u, predicate);
    workgroupBarrier();
    var result: u32 = 0u;
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        result = result | (ballot_scratch[i] << i);
    }
    workgroupBarrier();
    return result;
}
```

The WGSL generator will:

1. Detect warp primitive usage during AST traversal
2. Emit the necessary `var<workgroup>` declarations for scratch space
3. Generate calls to the emulation helper functions
4. Insert `workgroupBarrier()` calls to ensure memory consistency
5. Derive `lane_id` from `local_invocation_id.x % 32u`

### CPU/Rust Backend: Thread-Local Simulation

For the CPU backend, warp primitives will be emulated using a thread-local warp context that simulates 32 lanes sequentially:

```rust
pub struct WarpContext {
    lane_values: [f32; 32],
    active_mask: u32,
    warp_size: usize,
}

impl WarpContext {
    pub fn shuffle(&self, value: f32, src_lane: u32) -> f32 {
        self.lane_values[src_lane as usize % self.warp_size]
    }

    pub fn shuffle_xor(&self, value: f32, lane_id: u32, mask: u32) -> f32 {
        let src_lane = lane_id ^ mask;
        self.lane_values[src_lane as usize % self.warp_size]
    }

    pub fn shuffle_down(&self, value: f32, lane_id: u32, delta: u32) -> f32 {
        let src_lane = lane_id + delta;
        if src_lane < self.warp_size as u32 {
            self.lane_values[src_lane as usize]
        } else {
            value
        }
    }

    pub fn ballot(&self, predicate: bool, lane_id: u32) -> u32 {
        // In simulation, collect predicates from all lanes
        let mut result: u32 = 0;
        for i in 0..self.warp_size {
            if self.active_mask & (1 << i) != 0 {
                // Each lane's predicate would be evaluated
                result |= (predicate as u32) << i;
            }
        }
        result
    }

    pub fn vote_all(&self, predicate: bool) -> bool {
        // In simulation, check if all active lanes have predicate true
        predicate // Simplified for single-threaded simulation
    }

    pub fn active_mask(&self) -> u32 {
        self.active_mask
    }
}
```

This context will be:
- Stored in thread-local storage during kernel simulation
- Populated by the kernel launcher when setting up the execution grid
- Accessed by generated code via `cuda_rust_wasm::runtime::warp_*` functions

### Implementation Plan

1. **`src/kernel/warp.rs`**: Implement `WarpContext` with full warp simulation logic
2. **`src/runtime/mod.rs`**: Export warp functions (`warp_shuffle`, `warp_shuffle_xor`, `warp_shuffle_down`, `warp_shuffle_up`, `warp_ballot`, `warp_vote_all`, `warp_vote_any`, `warp_activemask`)
3. **`src/transpiler/wgsl.rs`**: Replace the hardcoded `0` output with shared-memory-based emulation
4. **`src/transpiler/code_generator.rs`**: Verify that generated `cuda_rust_wasm::runtime::warp_*` calls resolve to real implementations
5. **Type support**: Warp operations must support `f32`, `i32`, `u32` value types. The scratch arrays and context will use generic or union-based storage.

### Warp Size Configuration

The emulated warp size will default to 32 (matching NVIDIA) but will be configurable:

- In WGSL, it maps to a subgroup within a workgroup. If the workgroup size is 64, there are 2 logical warps.
- On CPU, the warp size is a simulation parameter.
- The `BackendCapabilities` struct already has a `warp_size: u32` field for this purpose.

## Consequences

### Positive

- **Correctness for warp-dependent kernels.** Reduction kernels, parallel scans, histograms, and sorting networks that use warp shuffle will produce correct results on all backends.
- **No silent failures.** The current behavior (outputting `0` for all warp ops in WGSL) causes silent data corruption. Emulation eliminates this.
- **Leverages existing AST support.** The `WarpOp` enum and `WarpPrimitive` expression variant are already defined; only code generation needs to change.
- **Enables benchmarking.** Emulated warp operations can be profiled to measure the overhead vs. native warp operations, informing optimization decisions.

### Negative

- **Performance overhead vs. native warps.** Emulated warp shuffle through shared memory requires two `workgroupBarrier()` calls per operation (write + read). Native warp shuffle is a single instruction. This adds latency and limits throughput for warp-shuffle-heavy kernels.
- **Shared memory pressure.** Each active warp requires 128 bytes (32 lanes x 4 bytes) of workgroup shared memory for scratch space. Kernels that already use significant shared memory may hit limits. WGSL workgroup memory is typically limited to 16KB.
- **Subgroup extensions not used.** Some WebGPU implementations support the `subgroups` extension (similar to Vulkan subgroups), which provides native warp-like operations. This ADR does not use those extensions for portability, but a future optimization could detect and use them.
- **CPU simulation is single-threaded.** The CPU warp simulation runs all 32 lanes sequentially, providing no parallelism benefit. This is acceptable for correctness but not for performance.

### Risks

- **Workgroup size assumptions.** The emulation assumes workgroup sizes are multiples of 32. Non-standard workgroup sizes may cause incorrect lane mapping.
- **Type limitations.** Initial implementation supports `f32`, `i32`, `u32`. Double-precision (`f64`) warp shuffles are not supported in WGSL (f64 is not a WGSL type).
- **Divergent control flow.** CUDA warp primitives interact with thread divergence. The emulation does not model divergent execution; all lanes are assumed active unless explicitly masked.

## References

- `src/parser/ast.rs` -- `WarpOp` enum and `WarpPrimitive` expression (lines 249-253, 313-322)
- `src/transpiler/wgsl.rs` -- Current hardcoded "0" output for warp primitives (lines 377-389)
- `src/transpiler/code_generator.rs` -- Generated warp function calls (lines 377-432)
- `src/kernel/warp.rs` -- Empty warp module
- `src/runtime/mod.rs` -- Runtime module (no warp exports)
- `src/backend/backend_trait.rs` -- `BackendCapabilities::warp_size` field
- CUDA Warp Shuffle documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
- WGSL Subgroups proposal: https://www.w3.org/TR/WGSL/#subgroup-builtin-functions
