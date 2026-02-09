# Ubiquitous Language Glossary for cuda-rust-wasm

This glossary defines the shared vocabulary used across all bounded contexts in the cuda-rust-wasm project. Every term has a single, precise definition that all team members, documentation, code comments, and variable names should use consistently.

Terms are organized by domain area. Cross-references between terms are indicated by **bold** text.

---

## 1. CUDA Execution Model

### Kernel

A function that executes on the GPU, invoked by the host (CPU) and running in parallel across many **Threads**. In CUDA, kernels are declared with the `__global__` qualifier. In the cuda-rust-wasm AST, a kernel is represented by `KernelDef` (in `src/parser/ast.rs`). A kernel is the primary unit of transpilation -- the system transforms CUDA kernels into equivalent Rust functions or **WGSL Shaders**.

**Code reference**: `Item::Kernel(KernelDef)` in `src/parser/ast.rs`

### Thread

A single unit of execution within a **Kernel**. Each thread has a unique index (`threadIdx`) within its **Thread Block** and executes the same kernel code on different data (SIMT model). In WGSL, the equivalent concept is an **Invocation**. In the cuda-rust-wasm runtime, thread indices are accessed via `runtime::thread::index()` returning a **Dim3**.

### Thread Block

A group of **Threads** that execute together on a single Streaming Multiprocessor (SM) and can share data via **Shared Memory** and synchronize via `__syncthreads()`. A thread block has up to 1024 threads organized in a 1D, 2D, or 3D arrangement specified by **Dim3**. In WGSL, the equivalent concept is a **Workgroup**. In the cuda-rust-wasm runtime, represented by the `Block` struct in `src/runtime/grid.rs`.

**Code reference**: `Block` struct in `src/runtime/grid.rs`

### Grid

The complete set of **Thread Blocks** launched by a single **Kernel** invocation. A grid is organized in a 1D, 2D, or 3D arrangement of thread blocks, specified by a **Dim3**. The grid dimensions determine the total number of threads: `gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z`. In the runtime, represented by the `Grid` struct.

**Code reference**: `Grid` struct in `src/runtime/grid.rs`

### Warp

A group of 32 **Threads** within a **Thread Block** that execute in lockstep (SIMT). The warp is the fundamental scheduling unit on NVIDIA GPUs. Threads within a warp can communicate via **Warp Primitives** (shuffle, vote, ballot) without using **Shared Memory**. The warp size is reported by `BackendCapabilities::warp_size` (default 32). On non-NVIDIA backends, warps are emulated (see ADR-003).

**Code reference**: `src/kernel/warp.rs`, `WarpOp` enum in `src/parser/ast.rs`

### Lane

A single **Thread's** position within a **Warp**, identified by a lane index (0-31). Lane IDs are used to address specific threads in warp shuffle operations. For example, `__shfl_sync(mask, val, src_lane)` reads the value from the thread at lane `src_lane` in the same warp.

### Dim3

A three-component unsigned integer vector `(x, y, z)` used to specify the dimensions of **Grids**, **Thread Blocks**, and thread/block indices. In cuda-rust-wasm, `Dim3` is defined in `src/runtime/grid.rs` with constructors for 1D (`one_d`), 2D (`two_d`), and 3D (`new`) configurations. Implements `From<u32>`, `From<(u32, u32)>`, and `From<(u32, u32, u32)>`.

**Code reference**: `Dim3` struct in `src/runtime/grid.rs`

---

## 2. CUDA Memory Model

### Shared Memory

Fast, on-chip memory shared by all **Threads** within a single **Thread Block**. Declared with the `__shared__` qualifier in CUDA. In the AST, represented by `StorageClass::Shared`. In WGSL, mapped to `var<workgroup>`. Shared memory is used for inter-thread communication within a block and as a software-managed cache. Typical limit: 48KB per block (96KB on some architectures).

**Code reference**: `StorageClass::Shared` in `src/parser/ast.rs`, `SharedMemory<T>` in `src/memory/mod.rs`

### Constant Memory

Read-only memory that is cached and broadcast efficiently to all **Threads**. Declared with the `__constant__` qualifier in CUDA. In the AST, represented by `StorageClass::Constant`. In WGSL, mapped to `const` declarations. Limited to 64KB total.

**Code reference**: `StorageClass::Constant` in `src/parser/ast.rs`

### Device Memory

The main GPU memory (VRAM/GDDR/HBM), also called global memory. Accessible by all **Threads** across all **Thread Blocks** but with high latency compared to **Shared Memory**. In cuda-rust-wasm, managed by `DeviceBuffer` in `src/memory/device_memory.rs`. In the AST, pointer parameters to kernels typically point to device memory.

**Code reference**: `DeviceBuffer` in `src/memory/device_memory.rs`

### Unified Memory

A memory management system that provides a single address space accessible from both host (CPU) and device (GPU). The CUDA runtime automatically migrates pages between host and device as needed. In cuda-rust-wasm, abstracted by `UnifiedMemory` in `src/memory/unified_memory.rs`. Reported as available by `BackendCapabilities::supports_unified_memory`.

**Code reference**: `UnifiedMemory` in `src/memory/unified_memory.rs`

### Host Memory

CPU-accessible system memory. In CUDA, host memory may be "pinned" (page-locked) for faster DMA transfers to/from the GPU. In cuda-rust-wasm, managed by `HostBuffer` in `src/memory/host_memory.rs`.

**Code reference**: `HostBuffer` in `src/memory/host_memory.rs`

### Memory Pool

A pre-allocated collection of reusable memory buffers organized by size class (powers of 2). The pool reduces allocation overhead by reusing previously deallocated buffers. In cuda-rust-wasm, implemented by `MemoryPool` in `src/memory/memory_pool.rs`, with a global singleton accessible via `global_pool()`. Tracks cache hit/miss statistics via `PoolStats`.

**Code reference**: `MemoryPool` in `src/memory/memory_pool.rs`

---

## 3. Transpilation Pipeline

### Transpilation Unit

The complete input/output of a single transpilation operation: one CUDA source file parsed into an **AST** and transformed into one or more target-language outputs (Rust code, **WGSL Shader**). Represented by the `Ast` struct containing a `Vec<Item>`.

**Code reference**: `Ast` struct in `src/parser/ast.rs`

### AST Node

A node in the Abstract Syntax Tree produced by the **Parser**. The cuda-rust-wasm AST hierarchy is: `Ast` > `Item` > (`KernelDef` | `FunctionDef` | `GlobalVar` | `TypeDef`) > `Statement` > `Expression`. Every AST node is immutable after construction and serializable via `serde`.

**Code reference**: `src/parser/ast.rs` (parser AST), `src/transpiler/ast.rs` (legacy transpiler AST)

### Code Generator

A component that transforms an **AST** into target-language source code. cuda-rust-wasm has two code generators:
1. **`CodeGenerator`** -- produces Rust source code using `quote` and `proc-macro2` token streams
2. **`WgslGenerator`** -- produces **WGSL** shader source code as a string

**Code reference**: `CodeGenerator` in `src/transpiler/code_generator.rs`, `WgslGenerator` in `src/transpiler/wgsl.rs`

### Backend

An execution target that can compile and run transpiled **Kernels**. cuda-rust-wasm supports multiple backends through the `BackendTrait` interface: WebGPU (browser), CUDA (native NVIDIA), OpenCL (cross-vendor), wgpu-native (Vulkan/Metal/DX12), and WASM/CPU (scalar fallback). Backend selection is automatic based on platform and available hardware.

**Code reference**: `BackendTrait` in `src/backend/backend_trait.rs`, `get_backend()` in `src/backend/mod.rs`

### WGSL Shader

A compute shader written in WebGPU Shading Language (WGSL), the output of transpiling a CUDA **Kernel** for WebGPU execution. A WGSL compute shader is annotated with `@compute @workgroup_size(X, Y, Z)` and uses `@builtin(global_invocation_id)` for thread indexing. Generated by `WgslGenerator`.

**Code reference**: `WgslGenerator` in `src/transpiler/wgsl.rs`

### Type Mapping

The process of converting CUDA C++ types to equivalent types in the target language. Key mappings include:
- `int` / `unsigned int` to `i32` / `u32` (both Rust and WGSL)
- `float` to `f32` (both Rust and WGSL)
- `double` to `f64` (Rust only; WGSL does not support `f64`)
- `float*` (pointer) to `&mut f32` (Rust) or `array<f32>` in storage buffer (WGSL)
- `float4` to `[f32; 4]` (Rust) or `vec4<f32>` (WGSL)

**Code reference**: `type_to_wgsl()` in `src/transpiler/wgsl.rs`, `generate_type()` in `src/transpiler/code_generator.rs`

### Built-in Function

A CUDA function that has no user-visible definition but is provided by the CUDA runtime or compiler. Examples: `__syncthreads()`, `atomicAdd()`, `__shfl_sync()`, `rsqrtf()`, `__ldg()`. The transpiler maps these to target-language equivalents (e.g., `__syncthreads()` to `workgroupBarrier()` in WGSL).

**Code reference**: `src/transpiler/builtin_functions.rs`

---

## 4. SIMD and Vectorization

### SIMD Lane

A single data element position within a **Vector Register**. For a 256-bit AVX2 register holding `f32` values, there are 8 SIMD lanes (256 / 32 = 8). SIMD operations process all lanes simultaneously.

### Vector Register

A hardware register that holds multiple data elements and operates on all of them in a single instruction. Widths vary by architecture:
- **NEON** (ARM): 128-bit (4 x `f32`)
- **SSE4.2** (x86): 128-bit (4 x `f32`)
- **AVX2** (x86): 256-bit (8 x `f32`)
- **AVX-512** (x86): 512-bit (16 x `f32`)
- **WASM SIMD128**: 128-bit (4 x `f32`)

### Intrinsic

A compiler-provided function that maps directly to a specific CPU instruction. In Rust, intrinsics are accessed via `std::arch` (e.g., `std::arch::x86_64::_mm256_add_ps` for AVX2 float addition). Intrinsics require `unsafe` blocks because they bypass Rust's type system for direct hardware access.

### Feature Detection

Runtime determination of which **SIMD** instruction sets are available on the current CPU. On x86, performed via `std::is_x86_feature_detected!("avx2")`. On ARM and WASM, determined at compile time via target features. The SIMD dispatch layer uses feature detection to select the fastest available code path.

---

## 5. Nutanix Platform

### Nutanix Cluster

A group of physical server nodes running the Nutanix hyperconverged infrastructure software stack. Each node contributes compute, storage, and optionally GPU resources. The cluster is managed as a single entity through **Prism Central**.

### Prism Central

The centralized management plane for one or more **Nutanix Clusters**. Provides a REST API (v3 and v4) for resource discovery, VM lifecycle management, monitoring, and configuration. In cuda-rust-wasm, accessed via the `PrismClient` for GPU resource operations.

### AHV (Acropolis Hypervisor)

Nutanix's native Type-1 hypervisor. AHV manages virtual machines, including **GPU Passthrough** assignment. AHV is the virtualization layer that allocates physical GPU devices to specific VMs.

### NKE (Nutanix Kubernetes Engine)

Nutanix's managed Kubernetes service running on **AHV** VMs. NKE clusters can schedule GPU workloads using the NVIDIA GPU Operator and device plugin, exposing GPUs as Kubernetes resources (`nvidia.com/gpu`). In cuda-rust-wasm, NKE is a target for containerized kernel execution.

### NKE Pod

A Kubernetes pod running on an **NKE** cluster with GPU resources allocated. A pod running a cuda-rust-wasm workload would request GPU resources in its resource spec and have access to the GPU via the NVIDIA device plugin.

### GPU Passthrough

A hardware virtualization feature where a physical GPU is directly assigned to a virtual machine, bypassing the hypervisor for GPU operations. This gives the VM near-native GPU performance. In **AHV**, GPU passthrough is configured per-VM and managed through **Prism Central**. Alternative: vGPU (virtual GPU), which shares a physical GPU among multiple VMs.

---

## 6. WebGPU / WGSL Equivalents

### Workgroup

The WGSL/WebGPU equivalent of a CUDA **Thread Block**. A workgroup is a group of **Invocations** that execute together, can share **Workgroup Memory** (equivalent to CUDA **Shared Memory**), and synchronize via `workgroupBarrier()`. The workgroup size is specified by the `@workgroup_size(X, Y, Z)` attribute on compute shaders.

### Invocation

The WGSL/WebGPU equivalent of a CUDA **Thread**. A single execution instance of a compute shader within a **Workgroup**. Each invocation has a unique `local_invocation_id` (within its workgroup) and `global_invocation_id` (across the entire dispatch).

### Dispatch

The WGSL/WebGPU equivalent of a CUDA kernel launch. A dispatch specifies the number of **Workgroups** in each dimension (equivalent to CUDA's **Grid** dimensions). Called via `computePass.dispatchWorkgroups(x, y, z)`.

### Binding

A connection between a GPU resource (buffer, texture, sampler) and a shader variable. In WGSL, bindings are declared with `@group(G) @binding(B)` attributes. In cuda-rust-wasm, each pointer parameter to a CUDA **Kernel** becomes a storage buffer binding in the generated WGSL.

### Storage Buffer

A GPU buffer used for read/write data in compute shaders. In WGSL, declared as `var<storage, read_write> name: array<T>`. This is the WGSL equivalent of a CUDA device memory pointer passed to a **Kernel**.

### Workgroup Memory

The WGSL equivalent of CUDA **Shared Memory**. Declared with `var<workgroup>` address space. Accessible by all **Invocations** within a **Workgroup**. Synchronized via `workgroupBarrier()`.

### Workgroup Barrier

The WGSL equivalent of CUDA `__syncthreads()`. Called as `workgroupBarrier()` in WGSL. Ensures all **Invocations** in a **Workgroup** have reached the barrier before any continue, and that all writes to **Workgroup Memory** are visible.

**Code reference**: `Statement::SyncThreads` maps to `workgroupBarrier();` in `src/transpiler/wgsl.rs`

---

## 7. Warp Primitive Operations

### Warp Shuffle (`__shfl_sync`)

Reads a value from a specific **Lane** within the same **Warp**. Each thread specifies a source lane, and the operation returns the value held by the thread at that lane. Used for arbitrary data exchange within a warp. In WGSL, emulated via **Workgroup Memory** and `workgroupBarrier()` (see ADR-003).

**AST representation**: `WarpOp::Shuffle`

### Warp Shuffle XOR (`__shfl_xor_sync`)

Reads a value from the **Lane** computed as `current_lane XOR mask`. Commonly used for butterfly reduction patterns (e.g., parallel sum within a warp). For a 32-thread warp, masks of 16, 8, 4, 2, 1 perform a complete reduction in 5 steps.

**AST representation**: `WarpOp::ShuffleXor`

### Warp Shuffle Up (`__shfl_up_sync`)

Reads a value from the **Lane** that is `delta` positions lower (i.e., `current_lane - delta`). Used for inclusive/exclusive prefix scans within a warp.

**AST representation**: `WarpOp::ShuffleUp`

### Warp Shuffle Down (`__shfl_down_sync`)

Reads a value from the **Lane** that is `delta` positions higher (i.e., `current_lane + delta`). Used for reductions where lower lanes accumulate results from higher lanes.

**AST representation**: `WarpOp::ShuffleDown`

### Warp Vote (`__all_sync`, `__any_sync`)

Collective predicate evaluation across all active **Lanes** in a **Warp**. `__all_sync` returns true if the predicate is true for all active lanes. `__any_sync` returns true if the predicate is true for any active lane.

**AST representation**: `WarpOp::Vote`

### Warp Ballot (`__ballot_sync`)

Returns a 32-bit mask where bit `i` is set if the predicate is true for **Lane** `i`. Used for stream compaction, population counting, and divergence detection.

**AST representation**: `WarpOp::Ballot`

### Active Mask (`__activemask`)

Returns a 32-bit mask indicating which **Lanes** in the current **Warp** are actively executing (not diverged). Used to construct the `mask` parameter for other warp primitives.

**AST representation**: `WarpOp::ActiveMask`

---

## 8. Atomic Operations

### Atomic Add (`atomicAdd`)

Atomically reads a value from a memory address, adds a value to it, writes the result back, and returns the original value. Thread-safe across all **Threads** in a **Grid**. In WGSL, maps directly to `atomicAdd()` for `i32`/`u32`; requires CAS-loop emulation for `f32`.

### Atomic Compare-And-Swap (`atomicCAS`)

Atomically compares the value at a memory address with an expected value, and if they match, replaces it with a new value. Returns the original value. The fundamental building block for lock-free algorithms. In WGSL, maps to `atomicCompareExchangeWeak()`.

### Atomic Exchange (`atomicExch`)

Atomically replaces the value at a memory address with a new value and returns the original value. In WGSL, maps to `atomicExchange()`.

### Atomic Min/Max (`atomicMin`, `atomicMax`)

Atomically computes the minimum or maximum of the current value at an address and a provided value, storing the result. Returns the original value. In WGSL, maps directly to `atomicMin()` / `atomicMax()`.

### Atomic Bitwise (`atomicAnd`, `atomicOr`, `atomicXor`)

Atomically performs a bitwise AND, OR, or XOR between the value at an address and a provided value. Returns the original value. In WGSL, maps directly to `atomicAnd()` / `atomicOr()` / `atomicXor()`.

---

## 9. Backend and Runtime

### Backend Capabilities

A data structure describing the features and limits of a specific **Backend**. Includes maximum thread counts, block dimensions, shared memory size, warp size, and supported API features. Used by the transpiler and runtime to make code generation and launch decisions.

**Code reference**: `BackendCapabilities` struct in `src/backend/backend_trait.rs`

### Memcpy Kind

The direction of a memory copy operation between host and device. Four kinds: `HostToDevice`, `DeviceToHost`, `DeviceToDevice`, `HostToHost`. Each maps to different underlying buffer operations depending on the **Backend**.

**Code reference**: `MemcpyKind` enum in `src/backend/backend_trait.rs`

### Stream

An ordered sequence of GPU operations (kernel launches, memory copies) that execute sequentially relative to each other but may execute concurrently with operations on other streams. Used for overlapping computation and data transfer.

**Code reference**: `Stream` struct in `src/runtime/stream.rs`

### Event

A synchronization marker in a **Stream** that can be recorded and waited upon. Used to measure elapsed time between GPU operations and to synchronize operations across different streams.

**Code reference**: `Event` struct in `src/runtime/event.rs`

### Launch Configuration

The parameters for a **Kernel** launch: **Grid** dimensions (number of **Thread Blocks**), **Block** dimensions (threads per block), shared memory size, and optional **Stream**. In CUDA, specified with `<<<grid, block, shared_mem, stream>>>` syntax.

**Code reference**: `LaunchConfig` in `src/runtime/kernel.rs`

---

## 10. Project-Specific Terms

### cuda-rust-wasm

The overall project name. A CUDA-to-Rust/WGSL transpiler with WebGPU and WebAssembly support. Published as the `cuda-rust-wasm` crate.

### CudaRust

The main entry-point struct that combines a **Parser** and **Transpiler** into a single transpilation API. Provides `transpile()` and `to_webgpu()` methods.

**Code reference**: `CudaRust` struct in `src/lib.rs`

### Neural Integration

The bridge module connecting cuda-rust-wasm to the ruv-FANN neural network framework. Provides GPU-accelerated neural network operations (matrix multiplication, activation functions, convolution) using transpiled CUDA kernels.

**Code reference**: `src/neural_integration/` module

### Profile Metrics

Performance measurements collected during system operation, including execution duration (total, average, min, max), memory allocation/deallocation tracking, and custom metrics. Collected by the `GlobalProfiler` and exportable as CSV.

**Code reference**: `ProfileMetrics` in `src/profiling/mod.rs`

---

## Usage Guidelines

1. **In code**: Use these exact terms for type names, variable names, function names, and module names. For example, use `KernelDef` (not `GpuFunction`), `Dim3` (not `Vec3u`), `Workgroup` (not `ThreadGroup`).

2. **In documentation**: Always capitalize domain terms as defined here. Provide cross-references to this glossary for first occurrences in a document.

3. **In conversations**: Use the domain terms precisely. "Kernel" means a GPU function, not an OS kernel. "Thread" means a GPU execution unit, not a CPU thread.

4. **When adding new terms**: Add them to this glossary before using them in code or documentation. Include the code reference, the definition, and the relationships to existing terms.

5. **CUDA vs. WGSL terminology**: When both ecosystems have terms for the same concept, prefer the CUDA term as the canonical form and note the WGSL equivalent. For example, say "Thread Block (Workgroup in WGSL)" in documentation.
