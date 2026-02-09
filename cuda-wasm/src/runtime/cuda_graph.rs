//! CUDA Graphs: graph-based kernel execution
//!
//! Provides a dependency graph for kernel launches, enabling the runtime
//! to optimise scheduling by executing independent nodes in parallel and
//! replaying captured workloads without re-recording overhead.

use crate::{Result, runtime_error};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Unique node identifier within a graph
pub type NodeId = usize;

/// Kind of work represented by a graph node
#[derive(Debug, Clone)]
pub enum NodeKind {
    /// GPU kernel launch
    Kernel {
        name: String,
        grid: [u32; 3],
        block: [u32; 3],
    },
    /// Host-to-device or device-to-host memory copy
    Memcpy {
        size: usize,
        kind: MemcpyDirection,
    },
    /// Memory set (fill with a value)
    Memset {
        size: usize,
        value: u8,
    },
    /// Host callback
    HostCallback {
        name: String,
    },
    /// Empty / synchronization-only node
    Empty,
}

/// Memory copy direction for graph edges
#[derive(Debug, Clone, Copy)]
pub enum MemcpyDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

/// A node in a CUDA graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique ID
    pub id: NodeId,
    /// Kind of work
    pub kind: NodeKind,
    /// IDs of nodes that this node depends on
    pub dependencies: Vec<NodeId>,
    /// Execution state
    pub state: NodeState,
}

/// Execution state of a graph node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    Pending,
    Running,
    Completed,
    Failed,
}

/// CUDA Graph: a directed acyclic graph of GPU operations
pub struct CudaGraph {
    /// Graph name
    name: String,
    /// All nodes in the graph, indexed by NodeId
    nodes: HashMap<NodeId, GraphNode>,
    /// Next available node ID
    next_id: NodeId,
    /// Whether the graph has been instantiated (compiled for execution)
    instantiated: bool,
}

impl CudaGraph {
    /// Create a new empty graph
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: HashMap::new(),
            next_id: 0,
            instantiated: false,
        }
    }

    /// Get graph name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Add a kernel node
    pub fn add_kernel_node(
        &mut self,
        name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        dependencies: &[NodeId],
    ) -> Result<NodeId> {
        self.validate_dependencies(dependencies)?;
        let id = self.allocate_id();
        self.nodes.insert(id, GraphNode {
            id,
            kind: NodeKind::Kernel {
                name: name.to_string(),
                grid,
                block,
            },
            dependencies: dependencies.to_vec(),
            state: NodeState::Pending,
        });
        self.instantiated = false;
        Ok(id)
    }

    /// Add a memcpy node
    pub fn add_memcpy_node(
        &mut self,
        size: usize,
        kind: MemcpyDirection,
        dependencies: &[NodeId],
    ) -> Result<NodeId> {
        self.validate_dependencies(dependencies)?;
        let id = self.allocate_id();
        self.nodes.insert(id, GraphNode {
            id,
            kind: NodeKind::Memcpy { size, kind },
            dependencies: dependencies.to_vec(),
            state: NodeState::Pending,
        });
        self.instantiated = false;
        Ok(id)
    }

    /// Add a memset node
    pub fn add_memset_node(
        &mut self,
        size: usize,
        value: u8,
        dependencies: &[NodeId],
    ) -> Result<NodeId> {
        self.validate_dependencies(dependencies)?;
        let id = self.allocate_id();
        self.nodes.insert(id, GraphNode {
            id,
            kind: NodeKind::Memset { size, value },
            dependencies: dependencies.to_vec(),
            state: NodeState::Pending,
        });
        self.instantiated = false;
        Ok(id)
    }

    /// Add a host callback node
    pub fn add_host_node(
        &mut self,
        name: &str,
        dependencies: &[NodeId],
    ) -> Result<NodeId> {
        self.validate_dependencies(dependencies)?;
        let id = self.allocate_id();
        self.nodes.insert(id, GraphNode {
            id,
            kind: NodeKind::HostCallback {
                name: name.to_string(),
            },
            dependencies: dependencies.to_vec(),
            state: NodeState::Pending,
        });
        self.instantiated = false;
        Ok(id)
    }

    /// Add an empty synchronization node
    pub fn add_empty_node(&mut self, dependencies: &[NodeId]) -> Result<NodeId> {
        self.validate_dependencies(dependencies)?;
        let id = self.allocate_id();
        self.nodes.insert(id, GraphNode {
            id,
            kind: NodeKind::Empty,
            dependencies: dependencies.to_vec(),
            state: NodeState::Pending,
        });
        self.instantiated = false;
        Ok(id)
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    /// Get all root nodes (no dependencies)
    pub fn root_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .values()
            .filter(|n| n.dependencies.is_empty())
            .map(|n| n.id)
            .collect()
    }

    /// Get topological ordering of nodes for execution
    pub fn topological_order(&self) -> Result<Vec<NodeId>> {
        let mut visited = HashMap::new();
        let mut order = Vec::new();

        // Sort keys for deterministic iteration order
        let mut keys: Vec<NodeId> = self.nodes.keys().copied().collect();
        keys.sort();

        for id in keys {
            if !visited.contains_key(&id) {
                self.topo_visit(id, &mut visited, &mut order)?;
            }
        }

        // DFS visiting predecessors (deps) produces order where dependencies
        // come before dependents -- already a valid topological order.
        Ok(order)
    }

    /// Check if the graph is a valid DAG (no cycles)
    pub fn validate(&self) -> Result<()> {
        self.topological_order()?;
        Ok(())
    }

    /// Instantiate the graph (compile for execution)
    pub fn instantiate(&mut self) -> Result<GraphExec> {
        self.validate()?;
        self.instantiated = true;

        let order = self.topological_order()?;
        let nodes: Vec<GraphNode> = order
            .iter()
            .map(|id| self.nodes[id].clone())
            .collect();

        Ok(GraphExec {
            graph_name: self.name.clone(),
            nodes,
            execution_count: 0,
            total_execution_time_us: 0,
        })
    }

    /// Whether the graph has been instantiated
    pub fn is_instantiated(&self) -> bool {
        self.instantiated
    }

    // --- Private helpers ---

    fn allocate_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn validate_dependencies(&self, deps: &[NodeId]) -> Result<()> {
        for &dep in deps {
            if !self.nodes.contains_key(&dep) {
                return Err(runtime_error!(
                    "Dependency node {} does not exist in graph",
                    dep
                ));
            }
        }
        Ok(())
    }

    fn topo_visit(
        &self,
        id: NodeId,
        visited: &mut HashMap<NodeId, bool>,
        order: &mut Vec<NodeId>,
    ) -> Result<()> {
        if let Some(&in_progress) = visited.get(&id) {
            if in_progress {
                return Err(runtime_error!("Cycle detected in graph at node {}", id));
            }
            return Ok(());
        }

        visited.insert(id, true); // Mark as in-progress

        if let Some(node) = self.nodes.get(&id) {
            for &dep in &node.dependencies {
                self.topo_visit(dep, visited, order)?;
            }
        }

        visited.insert(id, false); // Mark as completed
        order.push(id);
        Ok(())
    }
}

/// Executable (instantiated) graph
pub struct GraphExec {
    /// Graph name
    graph_name: String,
    /// Nodes in topological order
    nodes: Vec<GraphNode>,
    /// Number of times this graph has been executed
    execution_count: u64,
    /// Total execution time in microseconds
    total_execution_time_us: u64,
}

impl GraphExec {
    /// Execute the graph
    ///
    /// In the CPU emulation backend, nodes are executed sequentially in
    /// topological order. With a real GPU backend, independent nodes could
    /// be dispatched in parallel.
    pub fn launch(&mut self) -> Result<GraphExecResult> {
        let start = Instant::now();
        let mut node_results = Vec::new();

        for node in &self.nodes {
            let node_start = Instant::now();

            // CPU emulation: just record that we "executed" each node
            match &node.kind {
                NodeKind::Kernel { name, grid, block } => {
                    let total_threads =
                        grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2];
                    node_results.push(NodeExecResult {
                        node_id: node.id,
                        name: name.clone(),
                        duration_us: node_start.elapsed().as_micros() as u64,
                        threads_launched: total_threads as u64,
                    });
                }
                NodeKind::Memcpy { size, .. } => {
                    node_results.push(NodeExecResult {
                        node_id: node.id,
                        name: format!("memcpy_{}_bytes", size),
                        duration_us: node_start.elapsed().as_micros() as u64,
                        threads_launched: 0,
                    });
                }
                NodeKind::Memset { size, .. } => {
                    node_results.push(NodeExecResult {
                        node_id: node.id,
                        name: format!("memset_{}_bytes", size),
                        duration_us: node_start.elapsed().as_micros() as u64,
                        threads_launched: 0,
                    });
                }
                NodeKind::HostCallback { name } => {
                    node_results.push(NodeExecResult {
                        node_id: node.id,
                        name: name.clone(),
                        duration_us: node_start.elapsed().as_micros() as u64,
                        threads_launched: 0,
                    });
                }
                NodeKind::Empty => {
                    node_results.push(NodeExecResult {
                        node_id: node.id,
                        name: "sync".to_string(),
                        duration_us: 0,
                        threads_launched: 0,
                    });
                }
            }
        }

        let total_us = start.elapsed().as_micros() as u64;
        self.execution_count += 1;
        self.total_execution_time_us += total_us;

        Ok(GraphExecResult {
            graph_name: self.graph_name.clone(),
            node_results,
            total_duration_us: total_us,
            execution_number: self.execution_count,
        })
    }

    /// Get execution count
    pub fn execution_count(&self) -> u64 {
        self.execution_count
    }

    /// Get average execution time in microseconds
    pub fn avg_execution_time_us(&self) -> u64 {
        if self.execution_count == 0 {
            0
        } else {
            self.total_execution_time_us / self.execution_count
        }
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Result of executing a graph
#[derive(Debug)]
pub struct GraphExecResult {
    pub graph_name: String,
    pub node_results: Vec<NodeExecResult>,
    pub total_duration_us: u64,
    pub execution_number: u64,
}

/// Result of executing a single node
#[derive(Debug)]
pub struct NodeExecResult {
    pub node_id: NodeId,
    pub name: String,
    pub duration_us: u64,
    pub threads_launched: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = CudaGraph::new("test_graph");
        assert_eq!(graph.name(), "test_graph");
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_add_kernel_node() {
        let mut graph = CudaGraph::new("test");
        let id = graph.add_kernel_node("my_kernel", [1, 1, 1], [256, 1, 1], &[]).unwrap();
        assert_eq!(graph.node_count(), 1);
        let node = graph.get_node(id).unwrap();
        assert!(matches!(&node.kind, NodeKind::Kernel { name, .. } if name == "my_kernel"));
    }

    #[test]
    fn test_add_memcpy_node() {
        let mut graph = CudaGraph::new("test");
        let id = graph
            .add_memcpy_node(1024, MemcpyDirection::HostToDevice, &[])
            .unwrap();
        assert_eq!(graph.node_count(), 1);
        let node = graph.get_node(id).unwrap();
        assert!(matches!(&node.kind, NodeKind::Memcpy { size: 1024, .. }));
    }

    #[test]
    fn test_graph_dependencies() {
        let mut graph = CudaGraph::new("pipeline");
        let upload = graph
            .add_memcpy_node(1024, MemcpyDirection::HostToDevice, &[])
            .unwrap();
        let compute = graph
            .add_kernel_node("process", [4, 1, 1], [256, 1, 1], &[upload])
            .unwrap();
        let download = graph
            .add_memcpy_node(1024, MemcpyDirection::DeviceToHost, &[compute])
            .unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.root_nodes(), vec![upload]);

        // Verify topological order
        let order = graph.topological_order().unwrap();
        let upload_pos = order.iter().position(|&x| x == upload).unwrap();
        let compute_pos = order.iter().position(|&x| x == compute).unwrap();
        let download_pos = order.iter().position(|&x| x == download).unwrap();

        assert!(upload_pos < compute_pos);
        assert!(compute_pos < download_pos);
    }

    #[test]
    fn test_invalid_dependency() {
        let mut graph = CudaGraph::new("test");
        let result = graph.add_kernel_node("k", [1, 1, 1], [1, 1, 1], &[999]);
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_instantiate() {
        let mut graph = CudaGraph::new("test");
        graph.add_kernel_node("k1", [1, 1, 1], [256, 1, 1], &[]).unwrap();
        graph.add_kernel_node("k2", [1, 1, 1], [256, 1, 1], &[]).unwrap();

        let exec = graph.instantiate();
        assert!(exec.is_ok());
        assert!(graph.is_instantiated());
    }

    #[test]
    fn test_graph_execute() {
        let mut graph = CudaGraph::new("pipeline");
        let n1 = graph.add_kernel_node("init", [1, 1, 1], [128, 1, 1], &[]).unwrap();
        let n2 = graph.add_kernel_node("compute", [4, 1, 1], [256, 1, 1], &[n1]).unwrap();
        graph.add_kernel_node("finalize", [1, 1, 1], [64, 1, 1], &[n2]).unwrap();

        let mut exec = graph.instantiate().unwrap();
        let result = exec.launch().unwrap();

        assert_eq!(result.graph_name, "pipeline");
        assert_eq!(result.node_results.len(), 3);
        assert_eq!(result.execution_number, 1);
    }

    #[test]
    fn test_graph_replay() {
        let mut graph = CudaGraph::new("replay_test");
        graph.add_kernel_node("k", [1, 1, 1], [32, 1, 1], &[]).unwrap();

        let mut exec = graph.instantiate().unwrap();

        // Execute multiple times (replay)
        for i in 1..=5 {
            let result = exec.launch().unwrap();
            assert_eq!(result.execution_number, i);
        }
        assert_eq!(exec.execution_count(), 5);
    }

    #[test]
    fn test_graph_validate_dag() {
        let mut graph = CudaGraph::new("valid");
        let a = graph.add_kernel_node("a", [1, 1, 1], [1, 1, 1], &[]).unwrap();
        let b = graph.add_kernel_node("b", [1, 1, 1], [1, 1, 1], &[a]).unwrap();
        graph.add_kernel_node("c", [1, 1, 1], [1, 1, 1], &[a, b]).unwrap();

        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_empty_graph_instantiate() {
        let mut graph = CudaGraph::new("empty");
        let mut exec = graph.instantiate().unwrap();
        let result = exec.launch().unwrap();
        assert_eq!(result.node_results.len(), 0);
    }

    #[test]
    fn test_memset_node() {
        let mut graph = CudaGraph::new("memset_test");
        let id = graph.add_memset_node(4096, 0, &[]).unwrap();
        let node = graph.get_node(id).unwrap();
        assert!(matches!(&node.kind, NodeKind::Memset { size: 4096, value: 0 }));
    }

    #[test]
    fn test_host_callback_node() {
        let mut graph = CudaGraph::new("callback_test");
        let id = graph.add_host_node("my_callback", &[]).unwrap();
        let node = graph.get_node(id).unwrap();
        assert!(matches!(&node.kind, NodeKind::HostCallback { name } if name == "my_callback"));
    }

    #[test]
    fn test_diamond_dependency_graph() {
        let mut graph = CudaGraph::new("diamond");
        let root = graph.add_kernel_node("root", [1, 1, 1], [1, 1, 1], &[]).unwrap();
        let left = graph.add_kernel_node("left", [1, 1, 1], [1, 1, 1], &[root]).unwrap();
        let right = graph.add_kernel_node("right", [1, 1, 1], [1, 1, 1], &[root]).unwrap();
        let join = graph.add_kernel_node("join", [1, 1, 1], [1, 1, 1], &[left, right]).unwrap();

        let order = graph.topological_order().unwrap();
        let root_pos = order.iter().position(|&x| x == root).unwrap();
        let left_pos = order.iter().position(|&x| x == left).unwrap();
        let right_pos = order.iter().position(|&x| x == right).unwrap();
        let join_pos = order.iter().position(|&x| x == join).unwrap();

        assert!(root_pos < left_pos);
        assert!(root_pos < right_pos);
        assert!(left_pos < join_pos);
        assert!(right_pos < join_pos);
    }
}
