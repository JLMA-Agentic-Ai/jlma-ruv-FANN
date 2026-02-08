//! vGPU scheduling and GPU partitioning for multi-tenant cuda-wasm workloads
//!
//! Provides scheduling algorithms to assign cuda-wasm workloads to GPU nodes
//! with support for multiple vGPU profiles, scheduling policies, and live
//! migration planning for workload rebalancing.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::CudaRustError;
use super::config::{GpuNode, GpuVendor};

/// vGPU profile defining a GPU partition size and capability.
///
/// Each profile maps to a specific GPU slice with defined memory and compute
/// resources. Naming follows the NVIDIA MIG / AMD partition conventions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VgpuProfile {
    /// NVIDIA A100 1g.5gb - 1/7 GPU, 5 GB memory
    A100_1g5gb,
    /// NVIDIA A100 2g.10gb - 2/7 GPU, 10 GB memory
    A100_2g10gb,
    /// NVIDIA A100 3g.20gb - 3/7 GPU, 20 GB memory
    A100_3g20gb,
    /// NVIDIA A100 4g.20gb - 4/7 GPU, 20 GB memory
    A100_4g20gb,
    /// NVIDIA A100 7g.40gb - Full GPU, 40 GB memory
    A100_7g40gb,
    /// AMD MI250X 1g.16gb - 1/4 GPU, 16 GB memory
    MI250X_1g16gb,
    /// AMD MI250X 2g.32gb - 2/4 GPU, 32 GB memory
    MI250X_2g32gb,
}

impl VgpuProfile {
    /// Memory provided by this vGPU profile in bytes
    pub fn memory_bytes(&self) -> u64 {
        match self {
            VgpuProfile::A100_1g5gb => 5 * 1024 * 1024 * 1024,
            VgpuProfile::A100_2g10gb => 10 * 1024 * 1024 * 1024,
            VgpuProfile::A100_3g20gb => 20 * 1024 * 1024 * 1024,
            VgpuProfile::A100_4g20gb => 20 * 1024 * 1024 * 1024,
            VgpuProfile::A100_7g40gb => 40 * 1024 * 1024 * 1024,
            VgpuProfile::MI250X_1g16gb => 16 * 1024 * 1024 * 1024,
            VgpuProfile::MI250X_2g32gb => 32 * 1024 * 1024 * 1024,
        }
    }

    /// Number of compute units provided by this profile
    pub fn compute_units(&self) -> u32 {
        match self {
            VgpuProfile::A100_1g5gb => 14,
            VgpuProfile::A100_2g10gb => 28,
            VgpuProfile::A100_3g20gb => 42,
            VgpuProfile::A100_4g20gb => 56,
            VgpuProfile::A100_7g40gb => 108,
            VgpuProfile::MI250X_1g16gb => 55,
            VgpuProfile::MI250X_2g32gb => 110,
        }
    }

    /// The GPU vendor this profile applies to
    pub fn vendor(&self) -> GpuVendor {
        match self {
            VgpuProfile::A100_1g5gb
            | VgpuProfile::A100_2g10gb
            | VgpuProfile::A100_3g20gb
            | VgpuProfile::A100_4g20gb
            | VgpuProfile::A100_7g40gb => GpuVendor::Nvidia,
            VgpuProfile::MI250X_1g16gb
            | VgpuProfile::MI250X_2g32gb => GpuVendor::Amd,
        }
    }
}

impl std::fmt::Display for VgpuProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VgpuProfile::A100_1g5gb => write!(f, "A100-1g.5gb"),
            VgpuProfile::A100_2g10gb => write!(f, "A100-2g.10gb"),
            VgpuProfile::A100_3g20gb => write!(f, "A100-3g.20gb"),
            VgpuProfile::A100_4g20gb => write!(f, "A100-4g.20gb"),
            VgpuProfile::A100_7g40gb => write!(f, "A100-7g.40gb"),
            VgpuProfile::MI250X_1g16gb => write!(f, "MI250X-1g.16gb"),
            VgpuProfile::MI250X_2g32gb => write!(f, "MI250X-2g.32gb"),
        }
    }
}

/// Scheduling policy controlling how workloads are placed on GPU nodes
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SchedulingPolicy {
    /// Pack workloads onto fewest nodes to maximize consolidation
    BinPacking,
    /// Spread workloads across nodes for fault tolerance
    Spread,
    /// Prefer nodes where the workload already has GPU affinity
    GpuAffinity,
    /// Optimize for workloads that need maximum GPU memory
    MemoryOptimized,
}

/// A request to schedule a cuda-wasm workload onto GPU resources
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkloadRequest {
    /// Unique workload name
    pub name: String,
    /// Minimum GPU memory required in bytes
    pub min_gpu_memory: u64,
    /// Minimum compute units required
    pub min_compute_units: u32,
    /// Preferred GPU vendor (None means any vendor)
    pub preferred_vendor: Option<GpuVendor>,
    /// Maximum acceptable scheduling latency in milliseconds
    pub max_latency_ms: Option<u64>,
}

impl WorkloadRequest {
    /// Create a new workload request with the given name and memory requirement
    pub fn new(name: impl Into<String>, min_gpu_memory: u64) -> Self {
        Self {
            name: name.into(),
            min_gpu_memory,
            min_compute_units: 0,
            preferred_vendor: None,
            max_latency_ms: None,
        }
    }

    /// Set the minimum compute units
    pub fn with_compute_units(mut self, units: u32) -> Self {
        self.min_compute_units = units;
        self
    }

    /// Set the preferred vendor
    pub fn with_vendor(mut self, vendor: GpuVendor) -> Self {
        self.preferred_vendor = Some(vendor);
        self
    }
}

/// Result of scheduling a workload onto a GPU node
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScheduleResult {
    /// Name of the workload that was scheduled
    pub workload_name: String,
    /// Node ID the workload was assigned to
    pub assigned_node: String,
    /// GPU device ID on the assigned node
    pub assigned_gpu: String,
    /// vGPU profile selected for the workload
    pub vgpu_profile: VgpuProfile,
    /// Estimated performance score (0.0 - 1.0, where 1.0 is optimal)
    pub estimated_performance: f64,
}

/// A plan to migrate a workload from one GPU to another
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MigrationPlan {
    /// Name of the workload to migrate
    pub workload_name: String,
    /// Source node ID
    pub from_node: String,
    /// Source GPU device ID
    pub from_gpu: String,
    /// Destination node ID
    pub to_node: String,
    /// Destination GPU device ID
    pub to_gpu: String,
    /// Reason for the migration
    pub reason: String,
    /// Estimated downtime in milliseconds
    pub estimated_downtime_ms: u64,
}

/// vGPU scheduler for multi-tenant cuda-wasm workloads on Nutanix clusters
pub struct VgpuScheduler {
    /// Scheduling policy to use
    policy: SchedulingPolicy,
    /// Available GPU nodes in the cluster
    nodes: Vec<GpuNode>,
}

impl VgpuScheduler {
    /// Create a new scheduler with the given policy and available nodes
    pub fn new(policy: SchedulingPolicy, nodes: Vec<GpuNode>) -> Self {
        Self { policy, nodes }
    }

    /// Get the current scheduling policy
    pub fn policy(&self) -> &SchedulingPolicy {
        &self.policy
    }

    /// Update the set of available nodes
    pub fn update_nodes(&mut self, nodes: Vec<GpuNode>) {
        self.nodes = nodes;
    }

    /// Select the best vGPU profile for a workload request
    pub fn select_profile(&self, request: &WorkloadRequest) -> Result<VgpuProfile, CudaRustError> {
        let profiles = self.candidate_profiles(request);
        profiles.into_iter().next().ok_or_else(|| {
            CudaRustError::RuntimeError(format!(
                "No suitable vGPU profile for workload '{}' (needs {} bytes memory, {} CUs)",
                request.name, request.min_gpu_memory, request.min_compute_units
            ))
        })
    }

    /// Schedule a batch of workloads onto available GPU nodes
    ///
    /// Applies the configured scheduling policy to assign each workload
    /// to a node and GPU, selecting an appropriate vGPU profile.
    pub fn schedule_workloads(
        &self,
        workloads: &[WorkloadRequest],
    ) -> Result<Vec<ScheduleResult>, CudaRustError> {
        if self.nodes.is_empty() {
            return Err(CudaRustError::RuntimeError(
                "No GPU nodes available for scheduling".to_string(),
            ));
        }

        let mut results = Vec::with_capacity(workloads.len());
        let mut node_load: std::collections::HashMap<String, u32> = std::collections::HashMap::new();

        for workload in workloads {
            let profile = self.select_profile(workload)?;
            let node = self.select_node(workload, &node_load)?;
            let gpu = node
                .available_gpus
                .first()
                .ok_or_else(|| {
                    CudaRustError::RuntimeError(format!(
                        "Node '{}' has no available GPUs",
                        node.host_id
                    ))
                })?;

            let performance = self.estimate_performance(&profile, workload);

            *node_load.entry(node.host_id.clone()).or_insert(0) += 1;

            results.push(ScheduleResult {
                workload_name: workload.name.clone(),
                assigned_node: node.host_id.clone(),
                assigned_gpu: gpu.device_id.clone(),
                vgpu_profile: profile,
                estimated_performance: performance,
            });
        }

        Ok(results)
    }

    /// Plan rebalancing migrations for current workload assignments
    ///
    /// Analyzes the current assignment distribution and proposes migrations
    /// to improve resource utilization or fault tolerance based on the
    /// scheduling policy.
    pub fn rebalance(
        &self,
        current_assignments: &[ScheduleResult],
    ) -> Vec<MigrationPlan> {
        let mut plans = Vec::new();

        if current_assignments.is_empty() || self.nodes.len() < 2 {
            return plans;
        }

        // Count workloads per node
        let mut node_counts: std::collections::HashMap<String, Vec<&ScheduleResult>> =
            std::collections::HashMap::new();
        for assignment in current_assignments {
            node_counts
                .entry(assignment.assigned_node.clone())
                .or_default()
                .push(assignment);
        }

        match &self.policy {
            SchedulingPolicy::Spread => {
                let avg = current_assignments.len() as f64 / self.nodes.len().max(1) as f64;
                let threshold = avg.ceil() as usize;

                for (node_id, assignments) in &node_counts {
                    if assignments.len() > threshold {
                        let excess = assignments.len() - threshold;
                        let target_node = self
                            .nodes
                            .iter()
                            .find(|n| {
                                n.host_id != *node_id
                                    && node_counts
                                        .get(&n.host_id)
                                        .map_or(0, |a| a.len())
                                        < threshold
                            });

                        if let Some(target) = target_node {
                            for assignment in assignments.iter().take(excess) {
                                plans.push(MigrationPlan {
                                    workload_name: assignment.workload_name.clone(),
                                    from_node: node_id.clone(),
                                    from_gpu: assignment.assigned_gpu.clone(),
                                    to_node: target.host_id.clone(),
                                    to_gpu: target
                                        .available_gpus
                                        .first()
                                        .map(|g| g.device_id.clone())
                                        .unwrap_or_default(),
                                    reason: "Spread rebalancing: node overloaded".to_string(),
                                    estimated_downtime_ms: 500,
                                });
                            }
                        }
                    }
                }
            }
            SchedulingPolicy::BinPacking => {
                // Consolidate from lightly-loaded nodes to heavily-loaded ones
                let mut light_nodes: Vec<_> = node_counts
                    .iter()
                    .filter(|(_, a)| a.len() == 1)
                    .collect();
                light_nodes.sort_by_key(|(_, a)| a.len());

                if let Some((heavy_node_id, _)) =
                    node_counts.iter().max_by_key(|(_, a)| a.len())
                {
                    for (light_id, assignments) in &light_nodes {
                        if *light_id == heavy_node_id {
                            continue;
                        }
                        for assignment in *assignments {
                            plans.push(MigrationPlan {
                                workload_name: assignment.workload_name.clone(),
                                from_node: light_id.to_string(),
                                from_gpu: assignment.assigned_gpu.clone(),
                                to_node: heavy_node_id.clone(),
                                to_gpu: String::new(),
                                reason: "BinPacking consolidation".to_string(),
                                estimated_downtime_ms: 300,
                            });
                        }
                    }
                }
            }
            _ => {
                // GpuAffinity and MemoryOptimized do not trigger proactive rebalancing
            }
        }

        plans
    }

    // --- Private helpers ---

    /// Return candidate vGPU profiles sorted by best fit (smallest sufficient profile first)
    fn candidate_profiles(&self, request: &WorkloadRequest) -> Vec<VgpuProfile> {
        let all_profiles = vec![
            VgpuProfile::A100_1g5gb,
            VgpuProfile::A100_2g10gb,
            VgpuProfile::A100_3g20gb,
            VgpuProfile::A100_4g20gb,
            VgpuProfile::A100_7g40gb,
            VgpuProfile::MI250X_1g16gb,
            VgpuProfile::MI250X_2g32gb,
        ];

        let mut candidates: Vec<VgpuProfile> = all_profiles
            .into_iter()
            .filter(|p| {
                p.memory_bytes() >= request.min_gpu_memory
                    && p.compute_units() >= request.min_compute_units
                    && request
                        .preferred_vendor
                        .as_ref()
                        .map_or(true, |v| p.vendor() == *v)
            })
            .collect();

        // Sort by memory ascending (smallest sufficient profile first)
        if self.policy == SchedulingPolicy::MemoryOptimized {
            candidates.sort_by(|a, b| b.memory_bytes().cmp(&a.memory_bytes()));
        } else {
            candidates.sort_by(|a, b| a.memory_bytes().cmp(&b.memory_bytes()));
        }

        candidates
    }

    /// Select the best node for a workload according to the scheduling policy
    fn select_node(
        &self,
        request: &WorkloadRequest,
        node_load: &std::collections::HashMap<String, u32>,
    ) -> Result<GpuNode, CudaRustError> {
        let mut eligible: Vec<&GpuNode> = self
            .nodes
            .iter()
            .filter(|n| {
                !n.available_gpus.is_empty()
                    && request.preferred_vendor.as_ref().map_or(true, |v| {
                        n.available_gpus.iter().any(|g| g.vendor == *v)
                    })
            })
            .collect();

        if eligible.is_empty() {
            return Err(CudaRustError::RuntimeError(format!(
                "No eligible nodes for workload '{}'",
                request.name
            )));
        }

        match &self.policy {
            SchedulingPolicy::BinPacking => {
                eligible.sort_by(|a, b| {
                    let load_a = node_load.get(&a.host_id).copied().unwrap_or(0);
                    let load_b = node_load.get(&b.host_id).copied().unwrap_or(0);
                    load_b.cmp(&load_a)
                });
            }
            SchedulingPolicy::Spread => {
                eligible.sort_by(|a, b| {
                    let load_a = node_load.get(&a.host_id).copied().unwrap_or(0);
                    let load_b = node_load.get(&b.host_id).copied().unwrap_or(0);
                    load_a.cmp(&load_b)
                });
            }
            SchedulingPolicy::MemoryOptimized => {
                eligible.sort_by(|a, b| {
                    b.available_gpu_memory().cmp(&a.available_gpu_memory())
                });
            }
            SchedulingPolicy::GpuAffinity => {
                // Prefer nodes matching the requested vendor
                eligible.sort_by(|a, b| {
                    let a_match = request
                        .preferred_vendor
                        .as_ref()
                        .map_or(0, |v| a.available_gpu_count(v));
                    let b_match = request
                        .preferred_vendor
                        .as_ref()
                        .map_or(0, |v| b.available_gpu_count(v));
                    b_match.cmp(&a_match)
                });
            }
        }

        Ok(eligible[0].clone())
    }

    /// Estimate performance score for a profile/workload combination
    fn estimate_performance(&self, profile: &VgpuProfile, request: &WorkloadRequest) -> f64 {
        let mem_ratio = if request.min_gpu_memory > 0 {
            (profile.memory_bytes() as f64 / request.min_gpu_memory as f64).min(1.0)
        } else {
            1.0
        };

        let compute_ratio = if request.min_compute_units > 0 {
            (profile.compute_units() as f64 / request.min_compute_units as f64).min(1.0)
        } else {
            1.0
        };

        (mem_ratio * 0.6 + compute_ratio * 0.4).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nutanix::config::*;
    use std::collections::HashMap;

    fn make_gpu_node(host_id: &str, vendor: GpuVendor, gpu_count: usize) -> GpuNode {
        let gpus: Vec<GpuInfo> = (0..gpu_count)
            .map(|i| GpuInfo {
                vendor: vendor.clone(),
                model: match &vendor {
                    GpuVendor::Nvidia => GpuModel::NvidiaA100,
                    GpuVendor::Amd => GpuModel::AmdMI250X,
                    _ => GpuModel::Other("Unknown".into()),
                },
                device_id: format!("gpu-{}-{}", host_id, i),
                memory_bytes: 80 * 1024 * 1024 * 1024,
                compute_units: 108,
                assigned: false,
                assigned_vm: None,
                mode: "vgpu".to_string(),
                numa_node: Some(0),
            })
            .collect();

        GpuNode {
            host_id: host_id.to_string(),
            host_name: format!("host-{}", host_id),
            cluster_id: "cluster-1".to_string(),
            cluster_name: "Test Cluster".to_string(),
            ip_address: "10.0.0.1".to_string(),
            available_gpus: gpus.clone(),
            total_gpus: gpus,
            capabilities: HostCapabilities {
                host_id: host_id.to_string(),
                host_name: format!("host-{}", host_id),
                cpu_arch: "x86_64".to_string(),
                cpu_cores: 64,
                ram_bytes: 512 * 1024 * 1024 * 1024,
                has_nvidia: matches!(vendor, GpuVendor::Nvidia),
                has_amd: matches!(vendor, GpuVendor::Amd),
                is_arm: false,
                gpus: vec![],
                hypervisor: "AHV".to_string(),
                aos_version: "6.7".to_string(),
                gpu_passthrough_supported: true,
                vgpu_supported: true,
                metadata: HashMap::new(),
            },
        }
    }

    #[test]
    fn test_vgpu_profile_memory() {
        assert_eq!(VgpuProfile::A100_1g5gb.memory_bytes(), 5 * 1024 * 1024 * 1024);
        assert_eq!(VgpuProfile::A100_7g40gb.memory_bytes(), 40 * 1024 * 1024 * 1024);
        assert_eq!(VgpuProfile::MI250X_2g32gb.memory_bytes(), 32 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_vgpu_profile_vendor() {
        assert_eq!(VgpuProfile::A100_1g5gb.vendor(), GpuVendor::Nvidia);
        assert_eq!(VgpuProfile::A100_7g40gb.vendor(), GpuVendor::Nvidia);
        assert_eq!(VgpuProfile::MI250X_1g16gb.vendor(), GpuVendor::Amd);
        assert_eq!(VgpuProfile::MI250X_2g32gb.vendor(), GpuVendor::Amd);
    }

    #[test]
    fn test_vgpu_profile_display() {
        assert_eq!(VgpuProfile::A100_3g20gb.to_string(), "A100-3g.20gb");
        assert_eq!(VgpuProfile::MI250X_1g16gb.to_string(), "MI250X-1g.16gb");
    }

    #[test]
    fn test_select_profile_nvidia() {
        let nodes = vec![make_gpu_node("n1", GpuVendor::Nvidia, 2)];
        let scheduler = VgpuScheduler::new(SchedulingPolicy::BinPacking, nodes);

        let req = WorkloadRequest::new("test", 8 * 1024 * 1024 * 1024)
            .with_vendor(GpuVendor::Nvidia);
        let profile = scheduler.select_profile(&req).unwrap();
        assert_eq!(profile, VgpuProfile::A100_2g10gb);
    }

    #[test]
    fn test_select_profile_amd() {
        let nodes = vec![make_gpu_node("n1", GpuVendor::Amd, 1)];
        let scheduler = VgpuScheduler::new(SchedulingPolicy::Spread, nodes);

        let req = WorkloadRequest::new("amd-job", 10 * 1024 * 1024 * 1024)
            .with_vendor(GpuVendor::Amd);
        let profile = scheduler.select_profile(&req).unwrap();
        assert_eq!(profile, VgpuProfile::MI250X_1g16gb);
    }

    #[test]
    fn test_schedule_workloads_bin_packing() {
        let nodes = vec![
            make_gpu_node("n1", GpuVendor::Nvidia, 4),
            make_gpu_node("n2", GpuVendor::Nvidia, 2),
        ];
        let scheduler = VgpuScheduler::new(SchedulingPolicy::BinPacking, nodes);

        let workloads = vec![
            WorkloadRequest::new("w1", 5 * 1024 * 1024 * 1024).with_vendor(GpuVendor::Nvidia),
            WorkloadRequest::new("w2", 5 * 1024 * 1024 * 1024).with_vendor(GpuVendor::Nvidia),
        ];

        let results = scheduler.schedule_workloads(&workloads).unwrap();
        assert_eq!(results.len(), 2);
        // BinPacking should prefer same node
        assert_eq!(results[0].assigned_node, results[1].assigned_node);
    }

    #[test]
    fn test_schedule_workloads_spread() {
        let nodes = vec![
            make_gpu_node("n1", GpuVendor::Nvidia, 2),
            make_gpu_node("n2", GpuVendor::Nvidia, 2),
        ];
        let scheduler = VgpuScheduler::new(SchedulingPolicy::Spread, nodes);

        let workloads = vec![
            WorkloadRequest::new("w1", 5 * 1024 * 1024 * 1024).with_vendor(GpuVendor::Nvidia),
            WorkloadRequest::new("w2", 5 * 1024 * 1024 * 1024).with_vendor(GpuVendor::Nvidia),
        ];

        let results = scheduler.schedule_workloads(&workloads).unwrap();
        assert_eq!(results.len(), 2);
        // Spread should use different nodes
        assert_ne!(results[0].assigned_node, results[1].assigned_node);
    }

    #[test]
    fn test_schedule_no_nodes_error() {
        let scheduler = VgpuScheduler::new(SchedulingPolicy::BinPacking, vec![]);
        let workloads = vec![WorkloadRequest::new("w1", 1024)];
        let result = scheduler.schedule_workloads(&workloads);
        assert!(result.is_err());
    }

    #[test]
    fn test_rebalance_spread() {
        let nodes = vec![
            make_gpu_node("n1", GpuVendor::Nvidia, 4),
            make_gpu_node("n2", GpuVendor::Nvidia, 4),
        ];
        let scheduler = VgpuScheduler::new(SchedulingPolicy::Spread, nodes);

        // All workloads on one node
        let assignments = vec![
            ScheduleResult {
                workload_name: "w1".into(),
                assigned_node: "n1".into(),
                assigned_gpu: "gpu-n1-0".into(),
                vgpu_profile: VgpuProfile::A100_1g5gb,
                estimated_performance: 1.0,
            },
            ScheduleResult {
                workload_name: "w2".into(),
                assigned_node: "n1".into(),
                assigned_gpu: "gpu-n1-1".into(),
                vgpu_profile: VgpuProfile::A100_1g5gb,
                estimated_performance: 1.0,
            },
            ScheduleResult {
                workload_name: "w3".into(),
                assigned_node: "n1".into(),
                assigned_gpu: "gpu-n1-2".into(),
                vgpu_profile: VgpuProfile::A100_1g5gb,
                estimated_performance: 1.0,
            },
        ];

        let plans = scheduler.rebalance(&assignments);
        assert!(!plans.is_empty(), "Should propose migrations for imbalanced spread");
        assert!(plans.iter().all(|p| p.to_node == "n2"));
    }

    #[test]
    fn test_rebalance_empty_assignments() {
        let nodes = vec![make_gpu_node("n1", GpuVendor::Nvidia, 2)];
        let scheduler = VgpuScheduler::new(SchedulingPolicy::Spread, nodes);
        let plans = scheduler.rebalance(&[]);
        assert!(plans.is_empty());
    }

    #[test]
    fn test_workload_request_builder() {
        let req = WorkloadRequest::new("my-job", 16 * 1024 * 1024 * 1024)
            .with_compute_units(50)
            .with_vendor(GpuVendor::Nvidia);
        assert_eq!(req.name, "my-job");
        assert_eq!(req.min_compute_units, 50);
        assert_eq!(req.preferred_vendor, Some(GpuVendor::Nvidia));
    }

    #[test]
    fn test_memory_optimized_selects_largest() {
        let nodes = vec![make_gpu_node("n1", GpuVendor::Nvidia, 2)];
        let scheduler = VgpuScheduler::new(SchedulingPolicy::MemoryOptimized, nodes);

        let req = WorkloadRequest::new("big-mem", 5 * 1024 * 1024 * 1024)
            .with_vendor(GpuVendor::Nvidia);
        let profile = scheduler.select_profile(&req).unwrap();
        // MemoryOptimized sorts largest first
        assert_eq!(profile, VgpuProfile::A100_7g40gb);
    }
}
