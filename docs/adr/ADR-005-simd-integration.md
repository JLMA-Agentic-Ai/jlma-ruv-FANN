# ADR-005: Wire SIMD Into Forward/Backward Computation Paths

## Status: Accepted

## Context
The `simd/` module has complete AVX2 implementations for matmul, matvec, bias addition,
and activation functions but is completely disconnected from the actual forward pass
(`neuron.rs`) and training (`training/mod.rs`). The `simd` feature flag doesn't even
gate the module (it's gated by `parallel`).

## Decision
1. Fix `simd` feature flag to actually gate `simd/` module
2. Unify the duplicate `ActivationFunction` enum (simd vs activation.rs)
3. Wire SIMD matmul into layer-level forward propagation via `ComputeBackend` trait
4. Add bounds checking (`debug_assert!`) before all unsafe SIMD blocks
5. Replace `std::mem::transmute` with `bytemuck::cast_slice` where applicable

## Consequences
- Actual SIMD acceleration for forward/backward passes
- Single canonical `ActivationFunction` enum
- Safer unsafe code with bounds validation
