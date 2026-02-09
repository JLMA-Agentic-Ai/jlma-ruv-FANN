# Architecture Overview: CUDA-WASM on Nutanix + ARM/AMD

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CUDA Source Code                             │
│                    (Existing kernels, unchanged)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   ruv-cuda-wasm      │
                    │   Transpiler         │
                    │                     │
                    │  ┌───────────────┐  │
                    │  │ CUDA Parser    │  │  Parses CUDA C++ into AST
                    │  │ (nom-based)    │  │  Full operator precedence
                    │  └───────┬───────┘  │  Warp primitives, atomics
                    │          │          │
                    │  ┌───────▼───────┐  │
                    │  │ Type Converter │  │  CUDA types → Rust/WGSL
                    │  │ Memory Mapper  │  │  Storage class mapping
                    │  │ Builtin Mapper │  │  Math, atomic, warp builtins
                    │  └───────┬───────┘  │
                    │          │          │
                    │  ┌───────▼───────┐  │
                    │  │ Code Generator │  │  Outputs WASM or WebGPU
                    │  └───────────────┘  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼──────┐ ┌──────▼───────┐ ┌──────▼──────┐
     │  WASM Module   │ │ WebGPU Shader│ │ Native Code │
     │  (.wasm)       │ │ (.wgsl)      │ │ (optional)  │
     └────────┬──────┘ └──────┬───────┘ └──────┬──────┘
              │                │                │
     ┌────────▼────────────────▼────────────────▼──────┐
     │              Runtime Layer                        │
     │                                                  │
     │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
     │  │ WASM     │  │ WebGPU   │  │ SIMD         │  │
     │  │ Runtime  │  │ Backend  │  │ Accelerator  │  │
     │  │          │  │          │  │              │  │
     │  │ wasmtime │  │ wgpu     │  │ AVX2/NEON   │  │
     │  │ wasmer   │  │ dawn     │  │ AVX-512/SVE │  │
     │  └──────────┘  └──────────┘  └──────────────┘  │
     │                                                  │
     │  ┌──────────────────────────────────────────┐   │
     │  │ Emulation Layer                           │   │
     │  │  • Warp primitives (shuffle, vote, ballot)│   │
     │  │  • Shared memory (static + dynamic)       │   │
     │  │  • Atomic operations (add, cas, min, max) │   │
     │  │  • Synchronization barriers               │   │
     │  └──────────────────────────────────────────┘   │
     └──────────────────────┬──────────────────────────┘
                            │
     ┌──────────────────────▼──────────────────────────┐
     │              Hardware Layer                       │
     │                                                  │
     │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐ │
     │  │ NVIDIA │ │  AMD   │ │  ARM   │ │ x86 CPU  │ │
     │  │ GPU    │ │  GPU   │ │ NEON   │ │ AVX2     │ │
     │  └────────┘ └────────┘ └────────┘ └──────────┘ │
     └─────────────────────────────────────────────────┘
```

## Nutanix Integration Layer

```
┌──────────────────────────────────────────────────────────────────┐
│                    Nutanix Infrastructure                          │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Prism Central                                               │ │
│  │  • GPU node discovery via v3 API                             │ │
│  │  • Cluster GPU inventory and capabilities                    │ │
│  │  • Workload placement recommendations                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────┐  ┌────────────────────────────────────┐ │
│  │  AHV Hypervisor     │  │  NKE (Kubernetes Engine)           │ │
│  │                     │  │                                    │ │
│  │  • GPU passthrough  │  │  • WASM runtime pods               │ │
│  │  • vGPU sharing     │  │  • GPU device plugin               │ │
│  │  • VM scheduling    │  │  • Node affinity for GPU types     │ │
│  │  • Live migration   │  │  • Horizontal pod autoscaling      │ │
│  └────────────────────┘  └────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  NC2 (Cloud Clusters)                                        │ │
│  │  • Same management plane across AWS, Azure, GCP              │ │
│  │  • Workload migration between on-prem and cloud              │ │
│  │  • Access to cloud-specific GPU instance types               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow: CUDA Kernel Execution

### Step 1: Parse
```
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```
The parser reads CUDA C++ and builds an Abstract Syntax Tree (AST) representing every function, variable, type, and operation.

### Step 2: Transform
The transpiler walks the AST and converts:
- `__global__` → compute shader entry point
- `threadIdx.x` → `global_invocation_id.x` (WebGPU) or loop index (CPU)
- `float*` → `&[f32]` (Rust) or `array<f32>` (WGSL)
- `atomicAdd` → `atomicAdd` (WGSL) or `fetch_add` (Rust)
- `__syncthreads()` → `workgroupBarrier()` (WebGPU) or barrier (CPU)

### Step 3: Execute
The runtime selects the best available backend:

| Priority | Backend | When Used |
|----------|---------|-----------|
| 1 | Native GPU | NVIDIA/AMD GPU detected with drivers |
| 2 | WebGPU | GPU available via WebGPU API |
| 3 | SIMD CPU | No GPU, but AVX2/NEON available |
| 4 | Scalar CPU | Fallback, always works |

## Component Details

### CUDA Parser
- **Technology**: nom parser combinators + logos lexer
- **Coverage**: Functions, types, operators, builtins, warp ops, atomics, control flow
- **Output**: Strongly-typed Rust AST

### SIMD Accelerator
- **x86_64**: AVX2 (256-bit, 8 floats), AVX-512 (512-bit, 16 floats), SSE2 (128-bit, 4 floats)
- **ARM64**: NEON (128-bit, 4 floats), SVE (scalable vector length)
- **WASM**: SIMD128 (128-bit, 4 floats)
- **Operations**: Vector add/mul/scale, dot product, reduce, matrix multiply

### Warp Emulation
Emulates CUDA's 32-thread warp model on non-NVIDIA hardware:
- Shuffle operations (up, down, xor, indexed)
- Vote operations (all, any, ballot)
- Reductions (sum, min, max)
- Uses atomic operations and shared buffers for correctness

### Shared Memory Emulation
- Static shared memory: compile-time sized, type-safe
- Dynamic shared memory: runtime-sized, reinterpretable
- Bank conflict detection for performance analysis

### Nutanix Client
- Discovers GPU nodes via Prism Central REST API
- Generates Kubernetes deployment manifests
- Supports AHV GPU passthrough and NKE scheduling
