# Executive Summary: CUDA-WASM on Nutanix + ARM/AMD

## The Problem

Organizations running AI and GPU workloads face a hard choice: stay locked into NVIDIA hardware and CUDA, or rewrite everything for new platforms. As ARM servers (AWS Graviton, Ampere Altra, Apple Silicon) and AMD GPUs (MI300X, Instinct) gain ground, this lock-in becomes a real business risk.

**The cost of doing nothing:**
- Vendor lock-in to a single GPU supplier
- Cannot run AI at the edge where ARM dominates
- Legacy CUDA code cannot move to modern cloud-native infrastructure
- Security gaps from running GPU workloads with full hardware access

## The Solution

**ruv-cuda-wasm** is a transpiler that converts CUDA code into WebAssembly (WASM) and WebGPU. This means your existing CUDA kernels run on *any* hardware — NVIDIA, AMD, ARM, or CPU — without rewriting a single line.

When combined with **Nutanix** infrastructure (AHV, NKE, NC2), you get:

- **Write once, run anywhere** — CUDA code works on AMD MI300X, ARM NEON, Intel GPUs, and NVIDIA
- **Edge-ready AI** — Run inference on tiny ARM devices using the same code as your datacenter
- **Sandboxed GPU workloads** — WASM provides memory-safe isolation without the risks of GPU passthrough
- **Zero-rewrite modernization** — Move legacy CUDA code to Kubernetes on Nutanix without changes
- **True cloud mobility** — Deploy the same workload on NC2 (AWS, Azure, GCP) or on-prem

## Key Numbers

| Metric | Value |
|--------|-------|
| CUDA fidelity (core operations) | 95%+ coverage |
| SIMD acceleration (AVX2/NEON) | Near-native CPU performance |
| Supported GPU backends | NVIDIA, AMD (ROCm), Intel, WebGPU |
| Supported CPU architectures | x86_64, ARM64 (NEON), WASM |
| Deployment targets | Nutanix AHV, NKE (Kubernetes), NC2, bare metal |
| Code changes required | Zero — existing CUDA kernels work as-is |

## Who Benefits

- **Infrastructure teams** — Consolidate GPU workloads on Nutanix without hardware lock-in
- **AI/ML engineers** — Run the same model training and inference code everywhere
- **Edge computing teams** — Deploy CUDA-based AI to ARM devices at the edge
- **Security teams** — Sandbox GPU workloads inside WASM instead of granting direct hardware access
- **Finance/procurement** — Choose GPUs from any vendor based on price/performance, not compatibility

## Bottom Line

ruv-cuda-wasm turns CUDA from a hardware requirement into a software capability. Combined with Nutanix, organizations can run GPU workloads on any hardware, in any location, with stronger security — all without rewriting existing code.
