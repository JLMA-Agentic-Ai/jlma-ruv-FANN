# Domain-Driven Design: Bounded Contexts for ruv-fann

## Core Domain: Neural Network Computation

### Bounded Context: Network Definition
- **Aggregates**: `NetworkDef`, `LayerDef`, `ActivationFunction`
- **Value Objects**: `Connection` (from, to, weight), layer topology
- **Invariants**: Layer sizes must match connection indices, weights must be finite
- **Repository**: I/O module (binary, JSON, FANN format serialization)

### Bounded Context: Inference Engine
- **Aggregates**: `InferenceBuffer` (per-layer activation storage)
- **Services**: Forward propagation, activation application
- **Invariants**: Buffer dimensions match network topology
- **Performance**: Zero-alloc inference with pre-allocated buffers

### Bounded Context: Training
- **Aggregates**: `Trainer<T>` (algorithm + state + buffers)
- **Entities**: `TrainingData`, `TrainingConfig`, optimizer state (Adam moments, Rprop step sizes)
- **Services**: Backprop, gradient computation, weight update, error calculation
- **Invariants**: Gradient dimensions match weight dimensions
- **Shared Kernel**: `ErrorFunction` trait, `LearningRateSchedule`, `StopCriteria`

### Bounded Context: Cascade Correlation
- **Aggregates**: `CascadeTrainer`, `CandidateNeuron`
- **Services**: Candidate generation, correlation maximization, network growth
- **Invariants**: Candidate must be trained before installation

### Bounded Context: GPU Compute
- **Aggregates**: `ComputeBackend`, `WebGPUDevice`, `BufferPool`
- **Services**: GPU buffer management, shader compilation, kernel dispatch
- **Anti-Corruption Layer**: `BackendSelector` routes to GPU/SIMD/CPU based on matrix size
- **Invariants**: Buffer sizes match data dimensions

### Bounded Context: Persistence (I/O)
- **Services**: Serialization/deserialization (binary, JSON, FANN format, compressed)
- **Invariants**: Size limits on all reads (ADR-001), validated structure post-load
- **Anti-Corruption Layer**: Format converters between internal representation and file formats

## Context Map

```
Network Definition  <--- uses ---> Inference Engine
       |                                   |
       | shared kernel                     | buffer dims
       v                                   v
    Training --------> GPU Compute <---- Persistence
       |                   |
       v                   v
  Cascade Correlation   BackendSelector
```

## Key DDD Principles Applied
1. **Ubiquitous Language**: `Network`, `Layer`, `Neuron`, `Connection`, `Weight`, `Activation`
2. **Aggregate Boundaries**: NetworkDef is immutable aggregate root; InferenceBuffer is separate
3. **Value Objects**: Connection (immutable weight+indices), ActivationFunction (enum)
4. **Services**: Forward propagation, gradient computation are stateless services operating on aggregates
5. **Anti-Corruption Layers**: BackendSelector protects core from GPU complexity; I/O validators protect from untrusted data
