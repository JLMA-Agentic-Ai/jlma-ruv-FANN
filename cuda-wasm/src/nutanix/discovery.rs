//! GPU resource discovery via Nutanix Prism Central API
//!
//! Provides async methods to query Nutanix Prism Central for GPU-equipped hosts,
//! cluster GPU summaries, and per-host capability detection (NVIDIA, AMD, ARM).

use crate::error::CudaRustError;
use super::config::*;
use std::collections::HashMap;

/// Prism Central v3 API response wrapper for list endpoints
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismListResponse<T> {
    /// The entities returned by the list query
    pub entities: Vec<T>,
    /// API response metadata
    pub metadata: PrismMetadata,
}

/// Prism Central v3 API metadata
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismMetadata {
    /// Total number of matching entities
    pub total_matches: Option<u64>,
    /// Result set length
    pub length: Option<u64>,
    /// Result set offset
    pub offset: Option<u64>,
}

/// Prism Central host entity (v3 API)
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismHostEntity {
    /// Host metadata (contains UUID, etc.)
    pub metadata: PrismEntityMetadata,
    /// Host status information
    pub status: Option<PrismHostStatus>,
    /// Host spec
    pub spec: Option<serde_json::Value>,
}

/// Prism entity metadata
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismEntityMetadata {
    /// Entity UUID
    pub uuid: String,
    /// Entity kind (e.g., "host", "cluster")
    pub kind: String,
}

/// Prism host status block
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismHostStatus {
    /// Host name
    pub name: Option<String>,
    /// Host resources
    pub resources: Option<PrismHostResources>,
    /// Cluster reference
    pub cluster_reference: Option<PrismReference>,
}

/// Prism host resources
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismHostResources {
    /// GPU list
    pub gpu_list: Option<Vec<PrismGpuInfo>>,
    /// Hypervisor info
    pub hypervisor: Option<PrismHypervisor>,
    /// CPU model
    pub cpu_model: Option<String>,
    /// Number of CPU cores
    pub num_cpu_cores: Option<u32>,
    /// Memory size in bytes
    pub memory_capacity_in_bytes: Option<u64>,
    /// Host IP addresses
    pub host_nics_id_list: Option<Vec<String>>,
    /// Controller VM IP
    pub controller_vm: Option<PrismControllerVm>,
}

/// Prism GPU info from host resources
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismGpuInfo {
    /// GPU vendor (e.g., "NVIDIA", "AMD")
    pub vendor: Option<String>,
    /// GPU name/model
    pub name: Option<String>,
    /// GPU device ID
    pub device_id: Option<String>,
    /// GPU mode (passthrough, vGPU)
    pub mode: Option<String>,
    /// Whether the GPU is assignable
    pub assignable: Option<bool>,
    /// Number of vGPU instances possible
    pub num_virtual_display_heads: Option<u32>,
    /// GPU memory in bytes
    pub gpu_memory_in_bytes: Option<u64>,
    /// NUMA node
    pub numa_node: Option<u32>,
    /// Fraction of physical GPU (for vGPU)
    pub fraction: Option<u32>,
    /// VM UUID assigned to (if any)
    pub consumer_reference: Option<PrismReference>,
}

/// Prism hypervisor info
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismHypervisor {
    /// Hypervisor type (AHV, ESXi)
    pub hypervisor_type: Option<String>,
    /// Hypervisor full name
    pub hypervisor_full_name: Option<String>,
}

/// Prism reference (UUID + kind)
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismReference {
    /// Referenced entity UUID
    pub uuid: String,
    /// Referenced entity kind
    pub kind: Option<String>,
    /// Referenced entity name
    pub name: Option<String>,
}

/// Prism controller VM info
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismControllerVm {
    /// Controller VM IP address
    pub ip: Option<String>,
}

/// Prism cluster entity
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismClusterEntity {
    /// Cluster metadata
    pub metadata: PrismEntityMetadata,
    /// Cluster status
    pub status: Option<PrismClusterStatus>,
}

/// Prism cluster status
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismClusterStatus {
    /// Cluster name
    pub name: Option<String>,
    /// Cluster resources
    pub resources: Option<PrismClusterResources>,
}

/// Prism cluster resources
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismClusterResources {
    /// AOS version
    pub config: Option<PrismClusterConfig>,
}

/// Prism cluster config
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismClusterConfig {
    /// Software version
    pub software_map: Option<serde_json::Value>,
    /// Build info
    pub build: Option<PrismBuildInfo>,
}

/// Prism build info
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PrismBuildInfo {
    /// Software version string
    pub version: Option<String>,
}

/// Client for interacting with Nutanix Prism Central API
///
/// Provides async methods for discovering GPU resources across
/// Nutanix clusters, including multi-vendor GPU detection and
/// ARM host identification.
pub struct NutanixClient {
    /// Prism Central connection configuration
    config: NutanixConfig,

    /// HTTP client (when reqwest feature is available)
    #[cfg(feature = "nutanix")]
    client: reqwest::Client,
}

impl NutanixClient {
    /// Create a new NutanixClient with the given configuration
    ///
    /// # Arguments
    /// * `config` - Nutanix Prism Central connection settings
    ///
    /// # Returns
    /// A new NutanixClient instance, or an error if the HTTP client
    /// could not be initialized.
    pub fn new(config: NutanixConfig) -> Result<Self, CudaRustError> {
        #[cfg(feature = "nutanix")]
        {
            let mut builder = reqwest::Client::builder()
                .timeout(config.timeout);

            if !config.verify_ssl {
                builder = builder.danger_accept_invalid_certs(true);
            }

            let client = builder.build().map_err(|e| {
                CudaRustError::RuntimeError(format!("Failed to create HTTP client: {}", e))
            })?;

            Ok(Self { config, client })
        }

        #[cfg(not(feature = "nutanix"))]
        {
            Ok(Self { config })
        }
    }

    /// Get the client configuration
    pub fn config(&self) -> &NutanixConfig {
        &self.config
    }

    /// Discover all GPU-equipped hosts across all clusters managed by Prism Central
    ///
    /// Queries the Prism Central v3 hosts/list endpoint and filters for hosts
    /// that have GPU resources available. When compiled without the `nutanix`
    /// feature, probes the local system for GPU hardware instead.
    ///
    /// # Returns
    /// A vector of `GpuNode` structs representing hosts with GPUs.
    pub async fn discover_gpu_nodes(&self) -> Result<Vec<GpuNode>, CudaRustError> {
        #[cfg(feature = "nutanix")]
        {
            self.discover_gpu_nodes_impl().await
        }

        #[cfg(not(feature = "nutanix"))]
        {
            Ok(self.local_discover_gpu_nodes())
        }
    }

    /// Get an aggregated GPU summary for a specific cluster or all clusters
    ///
    /// # Arguments
    /// * `cluster_id` - Optional cluster UUID to filter by. If None, aggregates all clusters.
    ///
    /// # Returns
    /// A `GpuClusterSummary` with total/available GPU counts and per-vendor breakdowns.
    pub async fn get_cluster_gpu_summary(
        &self,
        cluster_id: Option<&str>,
    ) -> Result<GpuClusterSummary, CudaRustError> {
        let nodes = self.discover_gpu_nodes().await?;

        let filtered_nodes: Vec<GpuNode> = match cluster_id {
            Some(id) => nodes.into_iter().filter(|n| n.cluster_id == id).collect(),
            None => nodes,
        };

        let cluster_name = filtered_nodes
            .first()
            .map(|n| n.cluster_name.clone())
            .unwrap_or_else(|| "All Clusters".to_string());

        let cluster_id_str = cluster_id
            .map(|s| s.to_string())
            .unwrap_or_else(|| "all".to_string());

        let mut gpus_by_vendor: HashMap<String, u32> = HashMap::new();
        let mut gpus_by_model: HashMap<String, u32> = HashMap::new();
        let mut total_gpu_count: u32 = 0;
        let mut available_gpu_count: u32 = 0;
        let mut total_memory: u64 = 0;
        let mut available_memory: u64 = 0;

        for node in &filtered_nodes {
            for gpu in &node.total_gpus {
                total_gpu_count += 1;
                total_memory += gpu.memory_bytes;
                *gpus_by_vendor
                    .entry(gpu.vendor.to_string())
                    .or_insert(0) += 1;
                *gpus_by_model
                    .entry(gpu.model.to_string())
                    .or_insert(0) += 1;
            }
            for gpu in &node.available_gpus {
                available_gpu_count += 1;
                available_memory += gpu.memory_bytes;
            }
        }

        Ok(GpuClusterSummary {
            cluster_id: cluster_id_str,
            cluster_name,
            gpu_host_count: filtered_nodes.len() as u32,
            total_gpu_count,
            available_gpu_count,
            gpus_by_vendor,
            gpus_by_model,
            total_gpu_memory_bytes: total_memory,
            available_gpu_memory_bytes: available_memory,
            nodes: filtered_nodes,
        })
    }

    /// Get detailed capabilities of a specific host
    ///
    /// Queries the Prism Central v3 hosts/{uuid} endpoint to retrieve
    /// full host information including CPU architecture, GPU details,
    /// and hypervisor configuration.
    ///
    /// # Arguments
    /// * `host_id` - The UUID of the host to query
    ///
    /// # Returns
    /// `HostCapabilities` with CPU architecture, GPU inventory, and platform details.
    pub async fn get_host_capabilities(
        &self,
        host_id: &str,
    ) -> Result<HostCapabilities, CudaRustError> {
        #[cfg(feature = "nutanix")]
        {
            self.get_host_capabilities_impl(host_id).await
        }

        #[cfg(not(feature = "nutanix"))]
        {
            Ok(self.local_host_capabilities(host_id))
        }
    }

    /// Find the best nodes for a given workload configuration
    ///
    /// Filters GPU nodes based on vendor preference, GPU count requirements,
    /// and architecture constraints.
    ///
    /// # Arguments
    /// * `vendor` - Preferred GPU vendor
    /// * `gpu_count` - Minimum number of available GPUs required
    /// * `require_arm` - Whether to require ARM architecture hosts
    ///
    /// # Returns
    /// Sorted vector of `GpuNode` entries that match the criteria,
    /// ordered by available GPU count (descending).
    pub async fn find_best_nodes(
        &self,
        vendor: &GpuVendor,
        gpu_count: usize,
        require_arm: bool,
    ) -> Result<Vec<GpuNode>, CudaRustError> {
        let nodes = self.discover_gpu_nodes().await?;

        let mut matching: Vec<GpuNode> = nodes
            .into_iter()
            .filter(|n| {
                n.has_available_gpus(vendor, gpu_count)
                    && (!require_arm || n.capabilities.is_arm)
            })
            .collect();

        // Sort by available GPU count descending (prefer nodes with more GPUs)
        matching.sort_by(|a, b| {
            b.available_gpu_count(vendor)
                .cmp(&a.available_gpu_count(vendor))
        });

        Ok(matching)
    }

    // --- Private implementation methods ---

    /// Actual HTTP-based discovery (only compiled with nutanix feature)
    #[cfg(feature = "nutanix")]
    async fn discover_gpu_nodes_impl(&self) -> Result<Vec<GpuNode>, CudaRustError> {
        let url = self.config.api_url("hosts/list");

        let body = serde_json::json!({
            "kind": "host",
            "length": 500,
            "offset": 0
        });

        let response = self.send_request(&url, &body).await?;
        let list_response: PrismListResponse<PrismHostEntity> =
            serde_json::from_value(response).map_err(|e| {
                CudaRustError::RuntimeError(format!("Failed to parse Prism response: {}", e))
            })?;

        let mut gpu_nodes = Vec::new();

        for entity in list_response.entities {
            if let Some(status) = &entity.status {
                if let Some(resources) = &status.resources {
                    if let Some(gpu_list) = &resources.gpu_list {
                        if !gpu_list.is_empty() {
                            let node = self.prism_host_to_gpu_node(&entity)?;
                            gpu_nodes.push(node);
                        }
                    }
                }
            }
        }

        Ok(gpu_nodes)
    }

    /// Send authenticated request to Prism Central
    #[cfg(feature = "nutanix")]
    async fn send_request(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, CudaRustError> {
        let mut request = self.client.post(url).json(body);

        if !self.config.api_key.is_empty() {
            request = request.bearer_auth(&self.config.api_key);
        } else if let (Some(user), Some(pass)) = (&self.config.username, &self.config.password) {
            request = request.basic_auth(user, Some(pass));
        }

        let response = request.send().await.map_err(|e| {
            CudaRustError::RuntimeError(format!("Prism Central request failed: {}", e))
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(CudaRustError::RuntimeError(format!(
                "Prism Central API error ({}): {}",
                status, body_text
            )));
        }

        response.json().await.map_err(|e| {
            CudaRustError::RuntimeError(format!("Failed to parse Prism response: {}", e))
        })
    }

    /// Convert a Prism host entity to our GpuNode type
    #[cfg(feature = "nutanix")]
    fn prism_host_to_gpu_node(
        &self,
        entity: &PrismHostEntity,
    ) -> Result<GpuNode, CudaRustError> {
        let host_id = entity.metadata.uuid.clone();
        let status = entity.status.as_ref().ok_or_else(|| {
            CudaRustError::RuntimeError("Host entity missing status".to_string())
        })?;
        let resources = status.resources.as_ref().ok_or_else(|| {
            CudaRustError::RuntimeError("Host entity missing resources".to_string())
        })?;

        let host_name = status.name.clone().unwrap_or_else(|| host_id.clone());
        let gpu_list = resources.gpu_list.as_ref().cloned().unwrap_or_default();

        let cluster_ref = status.cluster_reference.as_ref();
        let cluster_id = cluster_ref.map(|r| r.uuid.clone()).unwrap_or_default();
        let cluster_name = cluster_ref
            .and_then(|r| r.name.clone())
            .unwrap_or_default();

        let ip_address = resources
            .controller_vm
            .as_ref()
            .and_then(|vm| vm.ip.clone())
            .unwrap_or_default();

        let cpu_arch = detect_cpu_arch(resources.cpu_model.as_deref());
        let is_arm = cpu_arch == "aarch64";
        let cpu_cores = resources.num_cpu_cores.unwrap_or(0);
        let ram_bytes = resources.memory_capacity_in_bytes.unwrap_or(0);

        let mut all_gpus = Vec::new();
        let mut available_gpus = Vec::new();
        let mut has_nvidia = false;
        let mut has_amd = false;

        for prism_gpu in &gpu_list {
            let gpu = prism_gpu_to_gpu_info(prism_gpu);
            match &gpu.vendor {
                GpuVendor::Nvidia => has_nvidia = true,
                GpuVendor::Amd => has_amd = true,
                _ => {}
            }
            if !gpu.assigned {
                available_gpus.push(gpu.clone());
            }
            all_gpus.push(gpu);
        }

        let hypervisor = resources
            .hypervisor
            .as_ref()
            .and_then(|h| h.hypervisor_type.clone())
            .unwrap_or_else(|| "AHV".to_string());

        let capabilities = HostCapabilities {
            host_id: host_id.clone(),
            host_name: host_name.clone(),
            cpu_arch,
            cpu_cores,
            ram_bytes,
            has_nvidia,
            has_amd,
            is_arm,
            gpus: all_gpus.clone(),
            hypervisor: hypervisor.clone(),
            aos_version: String::new(),
            gpu_passthrough_supported: true,
            vgpu_supported: gpu_list.iter().any(|g| g.mode.as_deref() == Some("VIRTUAL")),
            metadata: HashMap::new(),
        };

        Ok(GpuNode {
            host_id,
            host_name,
            cluster_id,
            cluster_name,
            ip_address,
            available_gpus,
            total_gpus: all_gpus,
            capabilities,
        })
    }

    /// Get host capabilities via API
    #[cfg(feature = "nutanix")]
    async fn get_host_capabilities_impl(
        &self,
        host_id: &str,
    ) -> Result<HostCapabilities, CudaRustError> {
        let url = self.config.api_url(&format!("hosts/{}", host_id));

        let body = serde_json::json!({});
        let response = self.send_request(&url, &body).await?;

        let entity: PrismHostEntity =
            serde_json::from_value(response).map_err(|e| {
                CudaRustError::RuntimeError(format!("Failed to parse host response: {}", e))
            })?;

        let node = self.prism_host_to_gpu_node(&entity)?;
        Ok(node.capabilities)
    }

    // --- Local system probing for non-nutanix builds ---

    /// Discover GPU nodes by probing the local system hardware.
    ///
    /// Checks for NVIDIA GPUs via `/proc/driver/nvidia` and `nvidia-smi`,
    /// and AMD GPUs via sysfs `/sys/class/drm`. Returns an empty vector
    /// if no GPUs are detected.
    #[cfg(not(feature = "nutanix"))]
    fn local_discover_gpu_nodes(&self) -> Vec<GpuNode> {
        let mut nodes = Vec::new();
        let hostname = std::fs::read_to_string("/etc/hostname")
            .unwrap_or_else(|_| "localhost".to_string())
            .trim()
            .to_string();

        let mut gpus = Vec::new();

        // Probe NVIDIA GPUs via /proc/driver/nvidia
        if let Ok(entries) = std::fs::read_dir("/proc/driver/nvidia/gpus") {
            for entry in entries.flatten() {
                if let Ok(info) =
                    std::fs::read_to_string(entry.path().join("information"))
                {
                    let model_name = info
                        .lines()
                        .find(|l| l.contains("Model:"))
                        .map(|l| {
                            l.split(':')
                                .nth(1)
                                .unwrap_or("Unknown")
                                .trim()
                                .to_string()
                        })
                        .unwrap_or_else(|| "NVIDIA GPU".to_string());
                    let device_id =
                        entry.file_name().to_string_lossy().to_string();

                    gpus.push(GpuInfo {
                        vendor: GpuVendor::Nvidia,
                        model: GpuModel::from_name(&model_name),
                        device_id,
                        memory_bytes: 0, // Would need nvidia-smi for accurate value
                        compute_units: 0,
                        assigned: false,
                        assigned_vm: None,
                        mode: "passthrough".to_string(),
                        numa_node: None,
                    });
                }
            }
        }

        // Probe NVIDIA GPUs via nvidia-smi (fallback when /proc is absent)
        if gpus.is_empty() {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args([
                    "--query-gpu=name,memory.total,uuid",
                    "--format=csv,noheader,nounits",
                ])
                .output()
            {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout.lines() {
                        let parts: Vec<&str> = line.split(", ").collect();
                        if parts.len() >= 3 {
                            let name = parts[0].trim();
                            let mem_mb: u64 =
                                parts[1].trim().parse().unwrap_or(0);
                            let uuid = parts[2].trim().to_string();
                            gpus.push(GpuInfo {
                                vendor: GpuVendor::Nvidia,
                                model: GpuModel::from_name(name),
                                device_id: uuid,
                                memory_bytes: mem_mb * 1024 * 1024,
                                compute_units: 0,
                                assigned: false,
                                assigned_vm: None,
                                mode: "passthrough".to_string(),
                                numa_node: None,
                            });
                        }
                    }
                }
            }
        }

        // Probe AMD GPUs via sysfs
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let vendor_path = entry.path().join("device/vendor");
                if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                    if vendor.trim() == "0x1002" {
                        // AMD vendor ID
                        let name = std::fs::read_to_string(
                            entry.path().join("device/product_name"),
                        )
                        .unwrap_or_else(|_| "AMD GPU".to_string());
                        let device_id =
                            entry.file_name().to_string_lossy().to_string();
                        gpus.push(GpuInfo {
                            vendor: GpuVendor::Amd,
                            model: GpuModel::from_name(name.trim()),
                            device_id,
                            memory_bytes: 0,
                            compute_units: 0,
                            assigned: false,
                            assigned_vm: None,
                            mode: "passthrough".to_string(),
                            numa_node: None,
                        });
                    }
                }
            }
        }

        // If no real GPUs found, return empty
        if gpus.is_empty() {
            return nodes;
        }

        let cpu_arch = std::env::consts::ARCH.to_string();
        let is_arm =
            cpu_arch.contains("aarch64") || cpu_arch.contains("arm");
        let has_nvidia =
            gpus.iter().any(|g| g.vendor == GpuVendor::Nvidia);
        let has_amd = gpus.iter().any(|g| g.vendor == GpuVendor::Amd);

        let caps = HostCapabilities {
            host_id: "local-host".to_string(),
            host_name: hostname.clone(),
            cpu_arch,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1),
            ram_bytes: Self::get_system_ram(),
            has_nvidia,
            has_amd,
            is_arm,
            gpus: gpus.clone(),
            hypervisor: "bare-metal".to_string(),
            aos_version: "N/A".to_string(),
            gpu_passthrough_supported: true,
            vgpu_supported: false,
            metadata: HashMap::new(),
        };

        nodes.push(GpuNode {
            host_id: "local-host".to_string(),
            host_name: hostname,
            cluster_id: "local".to_string(),
            cluster_name: "Local System".to_string(),
            ip_address: "127.0.0.1".to_string(),
            available_gpus: gpus.clone(),
            total_gpus: gpus,
            capabilities: caps,
        });

        nodes
    }

    /// Read total system RAM from /proc/meminfo, defaulting to 16 GB.
    #[cfg(not(feature = "nutanix"))]
    fn get_system_ram() -> u64 {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|info| {
                info.lines()
                    .find(|l| l.starts_with("MemTotal"))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|v| v.parse::<u64>().ok())
                            .map(|kb| kb * 1024)
                    })
            })
            .unwrap_or(16 * 1024 * 1024 * 1024) // 16 GB default
    }

    /// Get host capabilities by probing the local system.
    ///
    /// Reuses `local_discover_gpu_nodes` and returns the capabilities
    /// of the first (local) node, or synthesizes a basic capability
    /// set if no GPUs are detected.
    #[cfg(not(feature = "nutanix"))]
    fn local_host_capabilities(&self, host_id: &str) -> HostCapabilities {
        let nodes = self.local_discover_gpu_nodes();
        if let Some(node) = nodes.into_iter().next() {
            // Override the host_id with the requested one
            let mut caps = node.capabilities;
            caps.host_id = host_id.to_string();
            return caps;
        }

        // No GPUs found -- return basic system info
        let hostname = std::fs::read_to_string("/etc/hostname")
            .unwrap_or_else(|_| "localhost".to_string())
            .trim()
            .to_string();
        let cpu_arch = std::env::consts::ARCH.to_string();
        let is_arm =
            cpu_arch.contains("aarch64") || cpu_arch.contains("arm");

        HostCapabilities {
            host_id: host_id.to_string(),
            host_name: hostname,
            cpu_arch,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1),
            ram_bytes: Self::get_system_ram(),
            has_nvidia: false,
            has_amd: false,
            is_arm,
            gpus: Vec::new(),
            hypervisor: "bare-metal".to_string(),
            aos_version: "N/A".to_string(),
            gpu_passthrough_supported: false,
            vgpu_supported: false,
            metadata: HashMap::new(),
        }
    }
}

// --- Helper functions ---

/// Detect CPU architecture from a CPU model string
fn detect_cpu_arch(cpu_model: Option<&str>) -> String {
    match cpu_model {
        Some(model) => {
            let lower = model.to_lowercase();
            if lower.contains("arm") || lower.contains("aarch64") || lower.contains("graviton")
                || lower.contains("ampere") || lower.contains("neoverse")
                || lower.contains("cortex") || lower.contains("apple")
            {
                "aarch64".to_string()
            } else {
                "x86_64".to_string()
            }
        }
        None => "x86_64".to_string(),
    }
}

/// Convert a Prism GPU info entry to our GpuInfo type
#[cfg(feature = "nutanix")]
fn prism_gpu_to_gpu_info(prism_gpu: &PrismGpuInfo) -> GpuInfo {
    let vendor_str = prism_gpu.vendor.as_deref().unwrap_or("Unknown");
    let vendor = match vendor_str.to_uppercase().as_str() {
        "NVIDIA" => GpuVendor::Nvidia,
        "AMD" | "ATI" => GpuVendor::Amd,
        "INTEL" => GpuVendor::Intel,
        _ => GpuVendor::Unknown(vendor_str.to_string()),
    };

    let name = prism_gpu.name.as_deref().unwrap_or("Unknown GPU");
    let model = parse_gpu_model(name, &vendor);

    let assigned = prism_gpu.consumer_reference.is_some();
    let assigned_vm = prism_gpu.consumer_reference.as_ref().map(|r| r.uuid.clone());

    GpuInfo {
        vendor,
        model,
        device_id: prism_gpu.device_id.clone().unwrap_or_default(),
        memory_bytes: prism_gpu.gpu_memory_in_bytes.unwrap_or(0),
        compute_units: 0, // Not directly available from Prism API
        assigned,
        assigned_vm,
        mode: prism_gpu.mode.clone().unwrap_or_else(|| "passthrough".to_string()),
        numa_node: prism_gpu.numa_node,
    }
}

/// Parse a GPU model from its name string
#[cfg(feature = "nutanix")]
fn parse_gpu_model(name: &str, vendor: &GpuVendor) -> GpuModel {
    let upper = name.to_uppercase();

    match vendor {
        GpuVendor::Nvidia => {
            if upper.contains("A100") {
                GpuModel::NvidiaA100
            } else if upper.contains("H100") {
                GpuModel::NvidiaH100
            } else if upper.contains("L40") {
                GpuModel::NvidiaL40S
            } else if upper.contains("T4") {
                GpuModel::NvidiaT4
            } else if upper.contains("V100") {
                GpuModel::NvidiaV100
            } else {
                GpuModel::Other(name.to_string())
            }
        }
        GpuVendor::Amd => {
            if upper.contains("MI250") {
                GpuModel::AmdMI250X
            } else if upper.contains("MI300") {
                GpuModel::AmdMI300X
            } else if upper.contains("MI210") {
                GpuModel::AmdMI210
            } else {
                GpuModel::Other(name.to_string())
            }
        }
        GpuVendor::Intel => {
            if upper.contains("MAX") && upper.contains("1550") {
                GpuModel::IntelMax1550
            } else {
                GpuModel::Other(name.to_string())
            }
        }
        _ => GpuModel::Other(name.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_cpu_arch() {
        assert_eq!(detect_cpu_arch(Some("Intel Xeon Gold 6338")), "x86_64");
        assert_eq!(detect_cpu_arch(Some("AMD EPYC 7763")), "x86_64");
        assert_eq!(detect_cpu_arch(Some("ARM Neoverse N1")), "aarch64");
        assert_eq!(detect_cpu_arch(Some("Ampere Altra Q80-30")), "aarch64");
        assert_eq!(detect_cpu_arch(Some("AWS Graviton3")), "aarch64");
        assert_eq!(detect_cpu_arch(None), "x86_64");
    }

    #[test]
    fn test_client_creation() {
        let config = NutanixConfig::new("https://prism.example.com:9440", "test-key");
        let client = NutanixClient::new(config).unwrap();
        assert_eq!(client.config().base_url, "https://prism.example.com:9440");
    }

    #[tokio::test]
    async fn test_local_discover_gpu_nodes() {
        let config = NutanixConfig::new("https://prism.example.com:9440", "test-key");
        let client = NutanixClient::new(config).unwrap();
        let nodes = client.discover_gpu_nodes().await.unwrap();

        // On CI/environments without GPUs, the result may be empty -- that is correct.
        // On GPU hosts, every node should have a non-empty host name.
        for node in &nodes {
            assert!(!node.host_name.is_empty());
            assert!(!node.host_id.is_empty());
            assert!(!node.total_gpus.is_empty());
        }
    }

    #[tokio::test]
    async fn test_local_cluster_summary() {
        let config = NutanixConfig::new("https://prism.example.com:9440", "test-key");
        let client = NutanixClient::new(config).unwrap();
        let summary = client.get_cluster_gpu_summary(None).await.unwrap();

        // On systems without GPUs both counts will be zero
        assert!(summary.total_gpu_count >= summary.available_gpu_count);
        if summary.total_gpu_count > 0 {
            assert!(!summary.gpus_by_vendor.is_empty());
        }
    }

    #[tokio::test]
    async fn test_local_host_capabilities() {
        let config = NutanixConfig::new("https://prism.example.com:9440", "test-key");
        let client = NutanixClient::new(config).unwrap();
        let caps = client
            .get_host_capabilities("local-host")
            .await
            .unwrap();

        // These should always be populated from real system info
        assert!(!caps.cpu_arch.is_empty());
        assert!(caps.cpu_cores >= 1);
        assert!(caps.ram_bytes > 0);
        assert_eq!(caps.host_id, "local-host");
    }

    #[tokio::test]
    async fn test_find_best_nodes() {
        let config = NutanixConfig::new("https://prism.example.com:9440", "test-key");
        let client = NutanixClient::new(config).unwrap();

        // Results depend on whether real GPUs exist on the host
        let nvidia_nodes = client
            .find_best_nodes(&GpuVendor::Nvidia, 1, false)
            .await
            .unwrap();
        // On hosts without NVIDIA GPUs this will be empty -- that is correct
        for node in &nvidia_nodes {
            assert!(node.has_available_gpus(&GpuVendor::Nvidia, 1));
        }
    }
}
