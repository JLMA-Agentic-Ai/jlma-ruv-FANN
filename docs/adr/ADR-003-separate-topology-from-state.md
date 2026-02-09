# ADR-003: Separate Network Topology from Runtime State

## Status: Accepted

## Context
`Network::run()` requires `&mut self` because neuron activation values are stored inline.
This prevents concurrent inference and forces full network clones for error calculation
during training (180K+ allocations/epoch on MNIST-sized data).

## Decision
1. Split `Network<T>` into:
   - `NetworkDef<T>` (immutable): topology, weights, biases, activation functions
   - `InferenceBuffer<T>` (mutable): per-layer activation storage
2. `NetworkDef::run(&self, input, &mut InferenceBuffer)` takes `&self`
3. Training algorithms own their `InferenceBuffer` instead of cloning the network
4. Backward-compat: Keep `Network<T>` as a wrapper holding both

## Consequences
- Eliminates network cloning in `calculate_error()` (6 training algorithms)
- Enables `&self` inference = thread-safe concurrent inference
- Pre-allocated buffers eliminate per-call Vec allocations
- Training becomes significantly faster for large networks
