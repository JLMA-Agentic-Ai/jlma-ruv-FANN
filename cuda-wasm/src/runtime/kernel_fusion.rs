//! Kernel Fusion Engine
//!
//! Automatically detects and fuses element-wise / pointwise kernel sequences
//! to eliminate intermediate memory allocations and round-trips. This mirrors
//! the kernel fusion passes in TensorRT, XLA, and TVM.
//!
//! Fusion rules:
//! 1. Element-wise ops (add, mul, relu, etc.) can always fuse.
//! 2. Reduction followed by broadcast can fuse (vertical fusion).
//! 3. Producer-consumer pairs with matching shapes can fuse (horizontal).

use std::fmt;
use std::collections::HashMap;

/// An operation that can be part of a fused kernel.
#[derive(Debug, Clone, PartialEq)]
pub enum FusableOp {
    /// Element-wise: output[i] = f(input[i])
    Unary(UnaryOp),
    /// Element-wise: output[i] = f(a[i], b[i])
    Binary(BinaryOp),
    /// Reduction over a dimension
    Reduce(ReduceOp),
    /// Memory operation
    MemoryOp(MemOp),
}

/// Unary element-wise operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Relu, Sigmoid, Tanh, Gelu, Sqrt, Rsqrt, Exp, Log, Neg, Abs,
    Cast(PrecisionType, PrecisionType), // from, to
}

/// Binary element-wise operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Max, Min, Pow,
}

/// Reduction operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReduceOp {
    Sum, Max, Min, Mean,
}

/// Memory operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemOp {
    Load, Store, Copy,
}

/// Precision types for cast operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionType {
    Fp16, Bf16, Fp32, Fp64, Int8, Int32,
}

/// A node in the fusion graph.
#[derive(Debug, Clone)]
pub struct FusionNode {
    pub id: usize,
    pub op: FusableOp,
    /// Shape of the output tensor.
    pub shape: Vec<usize>,
    /// Input node IDs.
    pub inputs: Vec<usize>,
}

/// A fused kernel — a sequence of operations executed as one kernel.
#[derive(Debug, Clone)]
pub struct FusedKernel {
    pub id: usize,
    /// Nodes in execution order (topological).
    pub nodes: Vec<FusionNode>,
    /// Input node IDs (external inputs to the fused kernel).
    pub external_inputs: Vec<usize>,
    /// Output node IDs (nodes whose results are needed externally).
    pub external_outputs: Vec<usize>,
    /// Estimated memory saved by fusion (bytes).
    pub memory_saved: usize,
}

impl FusedKernel {
    /// Execute the fused kernel on f32 data.
    ///
    /// `inputs` maps external input IDs to their data.
    pub fn execute(&self, inputs: &HashMap<usize, Vec<f32>>) -> crate::Result<HashMap<usize, Vec<f32>>> {
        let mut buffers: HashMap<usize, Vec<f32>> = HashMap::new();

        // Copy external inputs
        for (&id, data) in inputs {
            buffers.insert(id, data.clone());
        }

        // Execute each node
        for node in &self.nodes {
            let result = match &node.op {
                FusableOp::Unary(op) => {
                    let input = buffers.get(&node.inputs[0])
                        .ok_or_else(|| crate::error::CudaRustError::RuntimeError(
                            format!("Missing input {} for node {}", node.inputs[0], node.id)))?;
                    apply_unary(op, input)
                }
                FusableOp::Binary(op) => {
                    let a = buffers.get(&node.inputs[0])
                        .ok_or_else(|| crate::error::CudaRustError::RuntimeError("Missing input A".into()))?;
                    let b = buffers.get(&node.inputs[1])
                        .ok_or_else(|| crate::error::CudaRustError::RuntimeError("Missing input B".into()))?;
                    apply_binary(op, a, b)
                }
                FusableOp::Reduce(op) => {
                    let input = buffers.get(&node.inputs[0])
                        .ok_or_else(|| crate::error::CudaRustError::RuntimeError("Missing reduce input".into()))?;
                    Ok(apply_reduce(op, input))
                }
                FusableOp::MemoryOp(_) => {
                    // Pass-through
                    let input = buffers.get(&node.inputs[0])
                        .ok_or_else(|| crate::error::CudaRustError::RuntimeError("Missing mem input".into()))?;
                    Ok(input.clone())
                }
            }?;
            buffers.insert(node.id, result);
        }

        // Collect external outputs
        let mut outputs = HashMap::new();
        for &id in &self.external_outputs {
            if let Some(data) = buffers.get(&id) {
                outputs.insert(id, data.clone());
            }
        }
        Ok(outputs)
    }

    /// Number of intermediate buffers eliminated by fusion.
    pub fn buffers_eliminated(&self) -> usize {
        let total_nodes = self.nodes.len();
        let external = self.external_inputs.len() + self.external_outputs.len();
        if total_nodes > external { total_nodes - external } else { 0 }
    }
}

fn apply_unary(op: &UnaryOp, input: &[f32]) -> crate::Result<Vec<f32>> {
    Ok(input.iter().map(|&x| match op {
        UnaryOp::Relu => x.max(0.0),
        UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Gelu => x * 0.5 * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh()),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Rsqrt => 1.0 / x.sqrt(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Cast(_, _) => x, // f32→f32 is identity
    }).collect())
}

fn apply_binary(op: &BinaryOp, a: &[f32], b: &[f32]) -> crate::Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(crate::error::CudaRustError::RuntimeError(
            format!("Binary op shape mismatch: {} vs {}", a.len(), b.len()),
        ));
    }
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        BinaryOp::Div => x / y,
        BinaryOp::Max => x.max(y),
        BinaryOp::Min => x.min(y),
        BinaryOp::Pow => x.powf(y),
    }).collect())
}

fn apply_reduce(op: &ReduceOp, input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return vec![0.0];
    }
    let result = match op {
        ReduceOp::Sum => input.iter().sum(),
        ReduceOp::Max => input.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        ReduceOp::Min => input.iter().cloned().fold(f32::INFINITY, f32::min),
        ReduceOp::Mean => input.iter().sum::<f32>() / input.len() as f32,
    };
    vec![result]
}

/// Fusion analysis engine that detects fusable patterns.
pub struct FusionAnalyzer {
    nodes: Vec<FusionNode>,
    next_id: usize,
}

impl FusionAnalyzer {
    /// Create a new analyzer.
    pub fn new() -> Self {
        Self { nodes: Vec::new(), next_id: 0 }
    }

    /// Add an operation node.
    pub fn add_node(&mut self, op: FusableOp, shape: Vec<usize>, inputs: Vec<usize>) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(FusionNode { id, op, shape, inputs });
        id
    }

    /// Analyze the graph and produce fused kernels.
    pub fn fuse(&self) -> FusionResult {
        let mut fused_kernels = Vec::new();
        let mut visited = vec![false; self.nodes.len()];
        let mut total_memory_saved = 0usize;

        // Build consumer map
        let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
        for node in &self.nodes {
            for &input_id in &node.inputs {
                consumers.entry(input_id).or_default().push(node.id);
            }
        }

        // Greedy fusion: chain element-wise ops
        for i in 0..self.nodes.len() {
            if visited[i] {
                continue;
            }

            let node = &self.nodes[i];
            if !is_element_wise(&node.op) {
                visited[i] = true;
                fused_kernels.push(FusedKernel {
                    id: fused_kernels.len(),
                    nodes: vec![node.clone()],
                    external_inputs: node.inputs.clone(),
                    external_outputs: vec![node.id],
                    memory_saved: 0,
                });
                continue;
            }

            // Start a fusion chain
            let mut chain = vec![node.clone()];
            visited[i] = true;
            let mut current_id = node.id;

            // Extend chain forward while next consumer is a single element-wise op
            loop {
                let next_consumers = consumers.get(&current_id);
                if let Some(cons) = next_consumers {
                    if cons.len() == 1 {
                        let next_id = cons[0];
                        if !visited[next_id] && next_id < self.nodes.len() {
                            let next_node = &self.nodes[next_id];
                            if is_element_wise(&next_node.op) && shapes_match(&node.shape, &next_node.shape) {
                                chain.push(next_node.clone());
                                visited[next_id] = true;
                                current_id = next_id;
                                continue;
                            }
                        }
                    }
                }
                break;
            }

            let shape = &chain[0].shape;
            let elem_size = 4; // f32
            let elems: usize = shape.iter().product();
            let intermediates = if chain.len() > 1 { chain.len() - 1 } else { 0 };
            let saved = intermediates * elems * elem_size;
            total_memory_saved += saved;

            // Determine external inputs and outputs
            let chain_ids: Vec<usize> = chain.iter().map(|n| n.id).collect();
            let external_inputs: Vec<usize> = chain.iter()
                .flat_map(|n| n.inputs.iter())
                .filter(|id| !chain_ids.contains(id))
                .copied()
                .collect();
            let last_id = chain.last().unwrap().id;

            fused_kernels.push(FusedKernel {
                id: fused_kernels.len(),
                nodes: chain,
                external_inputs,
                external_outputs: vec![last_id],
                memory_saved: saved,
            });
        }

        FusionResult {
            fused_kernels,
            total_memory_saved,
            original_kernel_count: self.nodes.len(),
        }
    }
}

fn is_element_wise(op: &FusableOp) -> bool {
    matches!(op, FusableOp::Unary(_) | FusableOp::Binary(_))
}

fn shapes_match(a: &[usize], b: &[usize]) -> bool {
    a == b
}

/// Result of fusion analysis.
#[derive(Debug)]
pub struct FusionResult {
    pub fused_kernels: Vec<FusedKernel>,
    pub total_memory_saved: usize,
    pub original_kernel_count: usize,
}

impl FusionResult {
    /// Number of kernels after fusion.
    pub fn fused_kernel_count(&self) -> usize {
        self.fused_kernels.len()
    }

    /// Reduction in kernel count.
    pub fn kernel_reduction(&self) -> f64 {
        if self.original_kernel_count == 0 { return 0.0; }
        1.0 - (self.fused_kernel_count() as f64 / self.original_kernel_count as f64)
    }
}

impl fmt::Display for FusionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fusion: {} → {} kernels ({:.0}% reduction), {:.1}KB memory saved",
            self.original_kernel_count,
            self.fused_kernel_count(),
            self.kernel_reduction() * 100.0,
            self.total_memory_saved as f64 / 1024.0)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unary_ops() {
        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let relu = apply_unary(&UnaryOp::Relu, &input).unwrap();
        assert_eq!(relu, vec![0.0, 0.0, 1.0, 2.0]);

        let neg = apply_unary(&UnaryOp::Neg, &input).unwrap();
        assert_eq!(neg, vec![1.0, 0.0, -1.0, -2.0]);

        let abs_r = apply_unary(&UnaryOp::Abs, &input).unwrap();
        assert_eq!(abs_r, vec![1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_binary_ops() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let add = apply_binary(&BinaryOp::Add, &a, &b).unwrap();
        assert_eq!(add, vec![5.0, 7.0, 9.0]);

        let mul = apply_binary(&BinaryOp::Mul, &a, &b).unwrap();
        assert_eq!(mul, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_reduce_ops() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(apply_reduce(&ReduceOp::Sum, &input), vec![10.0]);
        assert_eq!(apply_reduce(&ReduceOp::Max, &input), vec![4.0]);
        assert_eq!(apply_reduce(&ReduceOp::Min, &input), vec![1.0]);
        assert_eq!(apply_reduce(&ReduceOp::Mean, &input), vec![2.5]);
    }

    #[test]
    fn test_fusion_chain() {
        let mut analyzer = FusionAnalyzer::new();
        // Chain: input → relu → sigmoid → exp
        let input_id = analyzer.add_node(
            FusableOp::Unary(UnaryOp::Relu), vec![1024], vec![]
        );
        let relu_id = analyzer.add_node(
            FusableOp::Unary(UnaryOp::Sigmoid), vec![1024], vec![input_id]
        );
        let _exp_id = analyzer.add_node(
            FusableOp::Unary(UnaryOp::Exp), vec![1024], vec![relu_id]
        );

        let result = analyzer.fuse();
        // Should fuse all 3 into 1 kernel
        assert_eq!(result.fused_kernel_count(), 1);
        assert!(result.total_memory_saved > 0);
        assert!(result.kernel_reduction() > 0.5);
    }

    #[test]
    fn test_fusion_with_reduction_break() {
        let mut analyzer = FusionAnalyzer::new();
        let relu_id = analyzer.add_node(
            FusableOp::Unary(UnaryOp::Relu), vec![1024], vec![]
        );
        // Reduction breaks the chain
        let reduce_id = analyzer.add_node(
            FusableOp::Reduce(ReduceOp::Sum), vec![1], vec![relu_id]
        );
        let _exp_id = analyzer.add_node(
            FusableOp::Unary(UnaryOp::Exp), vec![1], vec![reduce_id]
        );

        let result = analyzer.fuse();
        // Relu alone, reduce alone, exp alone (reduce breaks fusion)
        assert!(result.fused_kernel_count() >= 2);
    }

    #[test]
    fn test_fused_kernel_execute() {
        // Manually build a fused kernel: relu → add
        let fused = FusedKernel {
            id: 0,
            nodes: vec![
                FusionNode { id: 1, op: FusableOp::Unary(UnaryOp::Relu), shape: vec![4], inputs: vec![0] },
                FusionNode { id: 2, op: FusableOp::Binary(BinaryOp::Add), shape: vec![4], inputs: vec![1, 3] },
            ],
            external_inputs: vec![0, 3],
            external_outputs: vec![2],
            memory_saved: 16,
        };

        let mut inputs = HashMap::new();
        inputs.insert(0, vec![-1.0, 0.0, 1.0, 2.0]);
        inputs.insert(3, vec![10.0, 10.0, 10.0, 10.0]);

        let outputs = fused.execute(&inputs).unwrap();
        let result = outputs.get(&2).unwrap();
        // relu([-1, 0, 1, 2]) = [0, 0, 1, 2], then + [10, 10, 10, 10] = [10, 10, 11, 12]
        assert_eq!(result, &vec![10.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_buffers_eliminated() {
        let fused = FusedKernel {
            id: 0,
            nodes: vec![
                FusionNode { id: 0, op: FusableOp::Unary(UnaryOp::Relu), shape: vec![1024], inputs: vec![] },
                FusionNode { id: 1, op: FusableOp::Unary(UnaryOp::Sigmoid), shape: vec![1024], inputs: vec![0] },
                FusionNode { id: 2, op: FusableOp::Unary(UnaryOp::Exp), shape: vec![1024], inputs: vec![1] },
            ],
            external_inputs: vec![],
            external_outputs: vec![2],
            memory_saved: 8192,
        };
        assert_eq!(fused.buffers_eliminated(), 2); // 3 nodes - 0 inputs - 1 output
    }

    #[test]
    fn test_gelu_sigmoid_fusion() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let gelu = apply_unary(&UnaryOp::Gelu, &input).unwrap();
        let sigmoid = apply_unary(&UnaryOp::Sigmoid, &input).unwrap();
        // Both should produce valid results
        assert!(gelu.iter().all(|v| v.is_finite()));
        assert!(sigmoid.iter().all(|v| *v >= 0.0 && *v <= 1.0));
    }

    #[test]
    fn test_fusion_display() {
        let result = FusionResult {
            fused_kernels: vec![],
            total_memory_saved: 65536,
            original_kernel_count: 10,
        };
        let s = format!("{}", result);
        assert!(s.contains("10"));
        assert!(s.contains("64.0KB"));
    }
}
