# Nutanix Integration for cuda-wasm

This directory contains examples and deployment manifests for running cuda-wasm GPU workloads on Nutanix infrastructure, including Nutanix Kubernetes Engine (NKE) clusters and AHV hypervisor GPU passthrough.

## Overview

The Nutanix integration module (`cuda_rust_wasm::nutanix`) provides:

1. **GPU Discovery** - Query Prism Central API to find GPU-equipped hosts
2. **Deployment Generation** - Generate Kubernetes manifests for NKE deployment
3. **Multi-Vendor Support** - Handle NVIDIA, AMD, and Intel GPUs
4. **Nutanix CSI Storage** - PersistentVolumeClaims using Nutanix Volumes

## Prism Central API Usage

cuda-wasm uses the Nutanix Prism Central v3 API to discover and manage GPU resources.

### Authentication

```rust
use cuda_rust_wasm::nutanix::{NutanixConfig, NutanixClient};

// API key authentication
let config = NutanixConfig::new(
    "https://prism-central.example.com:9440",
    "your-api-key",
);

// Basic authentication
let config = NutanixConfig::with_basic_auth(
    "https://prism-central.example.com:9440",
    "admin",
    "password",
);

let client = NutanixClient::new(config)?;
```

### GPU Discovery

```rust
// Discover all GPU-equipped hosts
let gpu_nodes = client.discover_gpu_nodes().await?;
for node in &gpu_nodes {
    println!("{}: {} GPUs available", node.host_name, node.available_gpus.len());
}

// Get cluster-wide GPU summary
let summary = client.get_cluster_gpu_summary(None).await?;
println!("Total GPUs: {}", summary.total_gpu_count);
println!("Available: {}", summary.available_gpu_count);

// Find nodes matching specific requirements
let best_nodes = client.find_best_nodes(&GpuVendor::Nvidia, 2, false).await?;
```

### Host Capabilities

```rust
let caps = client.get_host_capabilities("host-uuid").await?;
println!("Architecture: {}", caps.cpu_arch);
println!("Has NVIDIA: {}", caps.has_nvidia);
println!("Has AMD: {}", caps.has_amd);
println!("Is ARM: {}", caps.is_arm);
```

## NKE Deployment

### Prerequisites

1. **NKE Cluster** with a GPU-enabled node pool
2. **GPU Operator** (NVIDIA GPU Operator or AMD device plugin) installed
3. **Nutanix CSI Driver** configured with storage class `nutanix-volume`
4. **Node Feature Discovery** (optional, for advanced node selection)

### Quick Deployment

Apply the complete deployment manifest:

```bash
kubectl apply -f examples/nutanix/kubernetes_deployment.yaml
```

This creates:
- `cuda-wasm` namespace
- ConfigMap with runtime settings
- PersistentVolumeClaim for kernel cache (10Gi, Nutanix CSI)
- Deployment with NVIDIA GPU requests
- ClusterIP Service on port 8080
- HorizontalPodAutoscaler (1-8 replicas)

### Programmatic Deployment

Generate manifests from Rust code:

```rust
use cuda_rust_wasm::nutanix::{DeploymentConfig, GpuVendor, deployment::DeploymentGenerator};

let config = DeploymentConfig::new("my-workload", "my-image:latest")
    .with_gpu_vendor(GpuVendor::Nvidia)
    .with_gpus(2)
    .with_hpa(1, 8, 70)
    .with_nke_annotation("nke.nutanix.com/priority", "high");

let generator = DeploymentGenerator::new(config);
let yaml = generator.generate_all();
println!("{}", yaml);
```

## AHV VM GPU Passthrough

Nutanix AHV supports direct GPU passthrough to virtual machines, providing near-bare-metal GPU performance for cuda-wasm workloads.

### Configuration via Prism

1. Navigate to **VMs** in Prism Element
2. Select or create a VM
3. Under **Add New Disk** > **Add GPU**, select:
   - **Mode**: Passthrough (full GPU) or Virtual (vGPU)
   - **GPU**: Select the specific GPU device
4. Power on the VM and install GPU drivers

### GPU Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Passthrough** | Full GPU dedicated to one VM | Maximum performance, CUDA workloads |
| **vGPU (NVIDIA GRID)** | GPU shared across multiple VMs | Multi-tenant, inference workloads |

### Detection in cuda-wasm

```rust
let caps = client.get_host_capabilities("host-uuid").await?;
for gpu in &caps.gpus {
    println!("{} {} - Mode: {}, Assigned: {}",
        gpu.vendor, gpu.model, gpu.mode, gpu.assigned);
}
```

## Multi-Vendor GPU Support

cuda-wasm handles heterogeneous GPU environments common in enterprise deployments.

### NVIDIA GPUs

```rust
let config = DeploymentConfig::new("nvidia-worker", "cuda-wasm:latest")
    .with_gpu_vendor(GpuVendor::Nvidia)
    .with_gpus(1);
// Uses nvidia.com/gpu resource in Kubernetes
```

Supported NVIDIA GPUs:
- Tesla T4 (inference)
- A100 40GB/80GB (training & inference)
- H100 (large-scale training)
- L40S (graphics + compute)
- V100 (legacy workloads)

### AMD GPUs

```rust
let config = DeploymentConfig::new("amd-worker", "cuda-wasm:rocm")
    .with_gpu_vendor(GpuVendor::Amd)
    .with_gpus(1);
// Uses amd.com/gpu resource in Kubernetes
```

Supported AMD GPUs:
- Instinct MI210
- Instinct MI250X
- Instinct MI300X

### Intel GPUs

```rust
let config = DeploymentConfig::new("intel-worker", "cuda-wasm:oneapi")
    .with_gpu_vendor(GpuVendor::Intel)
    .with_gpus(1);
// Uses gpu.intel.com/i915 resource in Kubernetes
```

## Edge Deployment on Nutanix

cuda-wasm can be deployed on Nutanix edge infrastructure for low-latency GPU compute at the edge.

### NC2 (Nutanix Cloud Clusters)

Deploy cuda-wasm on NC2 for hybrid cloud GPU workloads:

```bash
# Same deployment manifest works on NC2
kubectl apply -f examples/nutanix/kubernetes_deployment.yaml
```

### Edge Considerations

- **Bandwidth**: Pre-cache compiled kernels to avoid repeated transpilation
- **Latency**: Use local Nutanix storage for kernel cache
- **Resource constraints**: Adjust replica counts and GPU requests for edge nodes
- **Monitoring**: Use Nutanix Prism for centralized GPU utilization monitoring

## Examples

### deploy_gpu_workload.rs

Full workflow example:

```bash
# With mock data
cargo run --example deploy_gpu_workload

# With real Nutanix connection
NUTANIX_PRISM_URL=https://prism.example.com:9440 \
NUTANIX_API_KEY=your-key \
cargo run --example deploy_gpu_workload --features nutanix
```

### kubernetes_deployment.yaml

Ready-to-use Kubernetes manifest:

```bash
# Deploy to NKE cluster
kubectl apply -f examples/nutanix/kubernetes_deployment.yaml

# Check deployment status
kubectl -n cuda-wasm get pods
kubectl -n cuda-wasm describe deployment cuda-wasm-worker

# View GPU allocation
kubectl -n cuda-wasm describe nodes | grep -A5 "Allocated resources"

# Scale manually
kubectl -n cuda-wasm scale deployment cuda-wasm-worker --replicas=4
```

## Troubleshooting

### GPU Not Detected

1. Verify GPU device plugin is running: `kubectl get pods -n gpu-operator`
2. Check node labels: `kubectl get nodes --show-labels | grep gpu`
3. Verify allocatable GPUs: `kubectl describe node <node> | grep gpu`

### Nutanix CSI Issues

1. Check CSI driver: `kubectl get pods -n ntnx-system`
2. Verify storage class: `kubectl get sc nutanix-volume`
3. Check PVC status: `kubectl -n cuda-wasm get pvc`

### Performance Issues

1. Verify GPU passthrough mode (not vGPU) for maximum performance
2. Check NUMA affinity between GPU and CPU
3. Monitor GPU utilization: `nvidia-smi` or DCGM exporter metrics
4. Ensure kernel cache PVC is using Nutanix Volumes (not Files)
