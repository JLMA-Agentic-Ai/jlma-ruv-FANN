# Deployment Guide: CUDA-WASM on Nutanix + ARM/AMD

## Prerequisites

- Nutanix cluster (AHV) with Prism Central
- NKE (Nutanix Kubernetes Engine) cluster provisioned
- Rust toolchain (for building ruv-cuda-wasm)
- Optional: GPU nodes (NVIDIA, AMD, or Intel)

## Quick Start

### 1. Build the Transpiler

```bash
cd cuda-wasm
cargo build --release
```

For ARM cross-compilation:
```bash
# Target ARM64
cargo build --release --target aarch64-unknown-linux-gnu

# With Nutanix integration
cargo build --release --features nutanix
```

### 2. Transpile Your CUDA Code

```rust
use cuda_wasm::parser::CudaParser;
use cuda_wasm::transpiler::WgslGenerator;

// Parse CUDA source
let ast = CudaParser::parse(cuda_source)?;

// Generate WASM-compatible code
let wasm_code = WgslGenerator::generate(&ast)?;
```

### 3. Discover Nutanix GPU Resources

```rust
use cuda_wasm::nutanix::{NutanixConfig, NutanixClient};

let config = NutanixConfig {
    prism_central_url: "https://prism.example.com:9440".into(),
    username: "admin".into(),
    password: "secret".into(), // Use environment variables in production
    cluster_name: Some("gpu-cluster".into()),
};

let client = NutanixClient::new(config);

// Find all GPU-capable nodes
let gpu_nodes = client.discover_gpu_nodes().await?;
for node in &gpu_nodes {
    println!("Node: {} - GPUs: {:?}", node.name, node.gpus);
}

// Get the best nodes for your workload
let best = client.find_best_nodes(4, Some("nvidia")).await?;
```

### 4. Generate Kubernetes Deployment

```rust
use cuda_wasm::nutanix::DeploymentGenerator;
use cuda_wasm::nutanix::DeploymentConfig;

let deploy_config = DeploymentConfig {
    name: "my-cuda-workload".into(),
    namespace: "gpu-workloads".into(),
    replicas: 3,
    gpu_count: 1,
    memory_limit: "8Gi".into(),
    cpu_limit: "4".into(),
    image: "myregistry/cuda-wasm-app:latest".into(),
};

let generator = DeploymentGenerator::new(deploy_config);
let yaml = generator.generate_full_deployment()?;

// Apply to NKE cluster
std::fs::write("deployment.yaml", &yaml)?;
```

### 5. Deploy to NKE

```bash
# Apply the generated manifests
kubectl apply -f deployment.yaml

# Verify pods are running
kubectl get pods -n gpu-workloads

# Check GPU allocation
kubectl describe nodes | grep -A5 "gpu"
```

## Deployment Topologies

### Single Datacenter (Nutanix AHV)

Best for: Organizations with one location and mixed GPU hardware.

```
Nutanix AHV Cluster
├── Node 1: NVIDIA A100 (training)
├── Node 2: NVIDIA A100 (training)
├── Node 3: AMD MI300X (inference)
├── Node 4: AMD MI300X (inference)
└── Node 5: CPU-only (overflow)

All nodes run the same WASM workload.
NKE schedules based on GPU availability.
```

### Hub and Spoke (Datacenter + Edge)

Best for: Organizations with edge locations using ARM devices.

```
Hub: Nutanix AHV (Datacenter)
├── NVIDIA GPUs for training
├── Model registry
└── Central management

Spoke: Nutanix Edge (Retail/Factory)
├── ARM processors (NEON SIMD)
├── WASM runtime for inference
└── Same container images as hub
```

### Multi-Cloud (NC2)

Best for: Organizations using multiple clouds for cost or compliance.

```
On-Premises: Nutanix AHV
├── Sensitive data stays here
└── Primary training cluster

NC2 on AWS
├── Burst capacity (Graviton ARM)
└── Cost-optimized inference

NC2 on Azure
├── Regional compliance
└── AMD MI300X for specific workloads
```

## ARM-Specific Deployment

### Building for ARM

```bash
# Native ARM build (on ARM host)
cargo build --release

# Cross-compile from x86 to ARM
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu
```

### ARM SIMD Verification

```rust
use cuda_wasm::simd::SimdCapabilities;

let caps = SimdCapabilities::detect();
println!("NEON: {}", caps.has_neon);
println!("SVE: {}", caps.has_sve);
println!("Best SIMD level: {:?}", caps.best_level());
```

### ARM Performance Tuning

| Setting | Recommended Value | Why |
|---------|------------------|-----|
| Thread count | Physical cores | ARM big.LITTLE needs care |
| SIMD width | 128-bit (NEON) | Universal on ARM64 |
| Tile size | 4x4 | Fits NEON register file |
| Memory alignment | 16 bytes | NEON requirement |

## Monitoring and Management

### Health Checks

```bash
# Check WASM runtime status
kubectl exec -it <pod> -- cuda-wasm-health

# Verify SIMD detection
kubectl exec -it <pod> -- cuda-wasm-simd-check

# GPU backend status
kubectl exec -it <pod> -- cuda-wasm-backend-status
```

### Performance Metrics

The runtime exposes metrics compatible with Prometheus:

```
cuda_wasm_kernel_executions_total
cuda_wasm_kernel_duration_seconds
cuda_wasm_backend_type{type="webgpu|wasm|simd|cpu"}
cuda_wasm_simd_level{level="avx2|neon|sse2|scalar"}
cuda_wasm_memory_allocated_bytes
```

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Slow on ARM | NEON not detected | Check `SimdCapabilities::detect()` |
| No GPU backend | Missing drivers | Install GPU drivers or use CPU mode |
| OOM on edge | Buffer too large | Reduce batch size for edge deployment |
| Different results | Floating-point precision | Enable strict IEEE mode |
| Pod won't schedule | GPU resource limit | Check `nvidia.com/gpu` or `amd.com/gpu` in node resources |
