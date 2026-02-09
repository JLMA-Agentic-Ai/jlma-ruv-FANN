# Aggregate Design: NetworkDef + InferenceBuffer

## Current Problem
```rust
// Current: Network owns both topology AND runtime state
pub struct Network<T> {
    pub layers: Vec<Layer<T>>,    // topology + weights + neuron values (mixed)
    pub connection_rate: f32,
}

// run() requires &mut self because neuron.value is mutated
pub fn run(&mut self, inputs: &[T]) -> Vec<T> { ... }
```

## Target Design (ADR-003)

### NetworkDef (Immutable Aggregate Root)
```rust
pub struct NetworkDef<T: Float> {
    layer_sizes: Vec<usize>,
    // Per-layer weight matrices stored as flat arrays for SIMD-friendly access
    weights: Vec<Vec<T>>,        // weights[layer][from * layer_size + to]
    biases: Vec<Vec<T>>,         // biases[layer][neuron]
    activations: Vec<ActivationFunction>,  // per-layer activation
    connection_rate: f32,
}

impl<T: Float> NetworkDef<T> {
    /// Thread-safe forward pass using external buffer
    pub fn forward(&self, input: &[T], buffer: &mut InferenceBuffer<T>) -> &[T] { ... }

    /// Convenience: allocates buffer internally (backward compat)
    pub fn run(&self, input: &[T]) -> Vec<T> { ... }
}
```

### InferenceBuffer (Mutable Value Object)
```rust
pub struct InferenceBuffer<T: Float> {
    activations: Vec<Vec<T>>,  // pre-allocated per-layer
}

impl<T: Float> InferenceBuffer<T> {
    pub fn new(network: &NetworkDef<T>) -> Self { ... }
}
```

### Training Uses Its Own Buffer
```rust
impl<T: Float> BatchBackprop<T> {
    fn train_epoch(&mut self, network: &mut NetworkDef<T>, data: &TrainingData<T>) {
        // Reuse buffer across samples -- no clone, no alloc
        let mut buffer = &mut self.inference_buffer;
        for (input, target) in data.iter() {
            let output = network.forward(input, buffer);
            self.accumulate_gradients(network, buffer, target);
        }
        self.apply_gradients(network);
    }
}
```

## Migration Path
1. Introduce `NetworkDef` and `InferenceBuffer` as new types
2. `Network<T>` becomes a wrapper: `struct Network<T> { def: NetworkDef<T>, buffer: InferenceBuffer<T> }`
3. `Network::run(&mut self)` delegates to `self.def.forward(&mut self.buffer)`
4. Training algorithms migrate to use `NetworkDef` + owned buffer
5. Eventually deprecate `Network<T>` wrapper
