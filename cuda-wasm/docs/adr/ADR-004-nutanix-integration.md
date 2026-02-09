# ADR-004: Add Nutanix Platform Integration Layer

## Status

**Accepted**

Date: 2025-07-15

## Context

The cuda-rust-wasm project currently supports three deployment models:

1. **Browser-based** -- WASM + WebGPU, running in web applications
2. **Server-side native** -- Direct CUDA/OpenCL execution on machines with GPU drivers
3. **Server-side WASM** -- CPU fallback via WASM runtime (Node.js, Wasmtime, etc.)

There is no support for deploying transpiled kernels on cloud or hyperconverged infrastructure (HCI) platforms. Enterprise GPU computing workloads (AI/ML inference, scientific simulation, video processing) are increasingly deployed on HCI platforms like Nutanix, which provide:

- **GPU passthrough and vGPU** support through AHV (Acropolis Hypervisor) for VMs
- **Nutanix Kubernetes Engine (NKE)** for containerized GPU workloads with device plugins
- **Prism Central API** for programmatic resource discovery and lifecycle management
- **GPU-aware scheduling** that can match workloads to nodes with specific GPU capabilities
- **Infrastructure-as-Code** via Nutanix APIs, Terraform providers, and Ansible modules

Currently, deploying cuda-rust-wasm on a Nutanix cluster requires manual VM provisioning, manual GPU passthrough configuration, manual driver installation, and no awareness of the underlying infrastructure capabilities. There is no programmatic way to:

- Discover which Nutanix cluster nodes have GPUs
- Query GPU models, memory, and compute capability
- Provision VMs or containers with GPU resources
- Deploy and manage transpiled kernel workloads
- Monitor GPU utilization and health

## Decision

We will add a Nutanix platform integration layer that enables automated GPU resource discovery, workload deployment, and lifecycle management on Nutanix HCI clusters.

### Architecture

```
src/platform/
    mod.rs                      -- Platform abstraction trait and registry
    nutanix/
        mod.rs                  -- Nutanix platform module
        prism_client.rs         -- Prism Central REST API client (v3 + v4)
        gpu_discovery.rs        -- GPU resource discovery and capability querying
        vm_templates.rs         -- AHV VM templates with GPU passthrough config
        nke_deployment.rs       -- NKE/Kubernetes deployment manifests and Helm values
        monitoring.rs           -- GPU utilization monitoring via Prism metrics API
        config.rs               -- Nutanix connection configuration and authentication
```

### Prism Central API Client

A typed REST client for the Nutanix Prism Central API (v3 and v4):

```rust
pub struct PrismClient {
    base_url: String,
    auth: PrismAuth,
    http_client: reqwest::Client,
}

pub enum PrismAuth {
    BasicAuth { username: String, password: String },
    ApiKey(String),
    OAuth2 { client_id: String, client_secret: String, token_url: String },
}

impl PrismClient {
    /// List all hosts with their GPU information
    pub async fn list_gpu_hosts(&self) -> Result<Vec<GpuHost>>;

    /// Get detailed GPU information for a specific host
    pub async fn get_host_gpus(&self, host_uuid: &str) -> Result<Vec<GpuDevice>>;

    /// List VMs with GPU passthrough assigned
    pub async fn list_gpu_vms(&self) -> Result<Vec<GpuVm>>;

    /// Create a VM with GPU passthrough
    pub async fn create_gpu_vm(&self, spec: &GpuVmSpec) -> Result<TaskReference>;

    /// Get GPU utilization metrics
    pub async fn get_gpu_metrics(&self, host_uuid: &str) -> Result<GpuMetrics>;
}
```

### GPU Resource Discovery

Automatic discovery of GPU resources across the Nutanix cluster:

```rust
pub struct GpuDevice {
    pub uuid: String,
    pub vendor: GpuVendor,
    pub model: String,
    pub vram_mb: u64,
    pub compute_capability: Option<String>,  // e.g., "8.6" for NVIDIA
    pub mode: GpuMode,                       // Passthrough or vGPU
    pub assigned_vm: Option<String>,         // VM UUID if assigned
    pub host_uuid: String,
    pub numa_node: Option<u32>,
}

pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
}

pub enum GpuMode {
    Passthrough,
    VirtualGpu { profile: String },  // e.g., "grid_t4-16q"
    Unused,
}

pub struct GpuHost {
    pub host_uuid: String,
    pub hostname: String,
    pub gpus: Vec<GpuDevice>,
    pub total_gpu_memory_mb: u64,
    pub available_gpus: usize,
}
```

### AHV VM Templates

Pre-configured VM templates for GPU compute workloads:

```rust
pub struct GpuVmSpec {
    pub name: String,
    pub vcpus: u32,
    pub memory_mb: u64,
    pub disk_gb: u64,
    pub gpu_config: GpuPassthroughConfig,
    pub cloud_init: Option<CloudInitConfig>,
    pub network_uuid: String,
    pub cluster_uuid: String,
}

pub struct GpuPassthroughConfig {
    pub mode: GpuMode,
    pub vendor: GpuVendor,
    pub device_id: Option<u32>,     // Specific PCI device ID
    pub min_vram_mb: Option<u64>,   // Minimum VRAM requirement
}

pub struct CloudInitConfig {
    pub install_nvidia_driver: bool,
    pub driver_version: Option<String>,
    pub install_cuda_toolkit: bool,
    pub cuda_version: Option<String>,
    pub install_cuda_wasm_runtime: bool,
    pub custom_script: Option<String>,
}
```

### NKE Deployment Configuration

Kubernetes deployment manifests for Nutanix Kubernetes Engine:

```rust
pub struct NkeDeployment {
    pub name: String,
    pub namespace: String,
    pub replicas: u32,
    pub gpu_request: GpuResourceRequest,
    pub container_image: String,
    pub environment: HashMap<String, String>,
}

pub struct GpuResourceRequest {
    pub count: u32,                             // Number of GPUs per pod
    pub resource_name: String,                  // e.g., "nvidia.com/gpu"
    pub memory_limit: Option<String>,           // e.g., "16Gi"
    pub node_selector: Option<HashMap<String, String>>,
    pub toleration: Option<String>,             // GPU node taint toleration
}

impl NkeDeployment {
    /// Generate a Kubernetes Deployment YAML
    pub fn to_deployment_yaml(&self) -> String;

    /// Generate a Kubernetes Job YAML for batch GPU workloads
    pub fn to_job_yaml(&self) -> String;

    /// Generate Helm values for the cuda-wasm-runtime chart
    pub fn to_helm_values(&self) -> String;
}
```

### Backend Integration

The Nutanix platform layer integrates with the existing backend system through a new platform-aware backend selector:

```rust
pub trait PlatformProvider: Send + Sync {
    fn name(&self) -> &str;
    async fn discover_gpus(&self) -> Result<Vec<GpuDevice>>;
    async fn provision_compute(&self, spec: &ComputeSpec) -> Result<ComputeInstance>;
    async fn deploy_kernel(&self, instance: &ComputeInstance, kernel: &[u8]) -> Result<()>;
    async fn teardown(&self, instance: &ComputeInstance) -> Result<()>;
}
```

### Configuration

Nutanix connection details will be configured via environment variables and/or a configuration file:

```toml
# cuda-wasm-nutanix.toml
[nutanix]
prism_central_url = "https://prism.example.com:9440"
username = "admin"       # Or use NUTANIX_USERNAME env var
# password via NUTANIX_PASSWORD env var (never in config file)
cluster_uuid = "..."
verify_ssl = true

[nutanix.gpu_defaults]
vendor = "nvidia"
min_vram_mb = 8192
mode = "passthrough"

[nutanix.nke]
kubeconfig_path = "~/.kube/config"
namespace = "cuda-wasm"
gpu_resource_name = "nvidia.com/gpu"
```

### Feature Gating

The Nutanix integration will be behind a feature flag to avoid adding dependencies for users who do not need it:

```toml
[features]
nutanix = ["reqwest", "serde", "serde_json"]

[dependencies]
reqwest = { version = "0.11", features = ["json", "rustls-tls"], optional = true }
```

## Consequences

### Positive

- **Enterprise deployment capability.** Organizations running Nutanix HCI can programmatically deploy cuda-rust-wasm workloads on GPU-equipped nodes.
- **Automated GPU discovery.** The platform layer eliminates manual GPU inventory management by querying Prism Central for available GPU resources.
- **Infrastructure-as-Code.** VM templates and NKE deployment configs enable reproducible, version-controlled GPU infrastructure.
- **Cloud-init automation.** GPU driver installation, CUDA toolkit setup, and cuda-wasm runtime deployment can be fully automated for new VMs.
- **Monitoring integration.** GPU utilization metrics from Prism Central provide visibility into workload performance.
- **Optional dependency.** Feature-gated behind `nutanix`, so non-enterprise users incur no additional dependencies or binary size.

### Negative

- **Dependency on Nutanix APIs.** The Prism Central API is Nutanix-proprietary. API versioning, deprecation, and changes require ongoing maintenance.
- **Authentication complexity.** Prism Central supports multiple authentication methods (basic auth, API key, OAuth2). The integration must handle token refresh, SSL certificate validation, and credential management securely.
- **Network requirements.** The platform layer requires network access to Prism Central and NKE API servers, which may be restricted in production environments.
- **Testing difficulty.** Integration tests require access to a Nutanix cluster with GPU hardware. Mock-based testing can validate API client logic but not end-to-end provisioning.
- **Scope expansion.** Adding infrastructure management to a transpiler project broadens the project's scope significantly.

### Risks

- **API version compatibility.** Prism Central v3 and v4 APIs have different endpoint schemas. The client must handle both versions for broad compatibility.
- **GPU passthrough limitations.** Not all GPU models support passthrough on all hypervisors. The discovery layer must report unsupported configurations.
- **Security posture.** Storing credentials for Prism Central requires careful handling. The integration must support environment variables, secret managers, and credential files with appropriate permissions.
- **vGPU licensing.** NVIDIA vGPU requires licensing. The integration should detect and report vGPU license status but cannot manage licensing directly.

## References

- `src/backend/mod.rs` -- Current backend selection logic
- `src/backend/backend_trait.rs` -- `BackendCapabilities` struct
- `Cargo.toml` -- Feature flag and dependency structure
- Nutanix Prism Central API v3: https://www.nutanix.dev/api-reference/prism-central/v3/
- Nutanix Prism Central API v4: https://www.nutanix.dev/api-reference/prism-central/v4/
- Nutanix Kubernetes Engine: https://www.nutanix.com/products/kubernetes-engine
- NVIDIA GPU Operator for Kubernetes: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/
