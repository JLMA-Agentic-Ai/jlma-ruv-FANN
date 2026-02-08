# Domain-Driven Design: Domain Model for cuda-rust-wasm

## Overview

This document defines the domain model for the cuda-rust-wasm project -- a CUDA-to-WebAssembly/WebGPU transpiler written in Rust. The domain model follows Domain-Driven Design principles to establish clear boundaries, aggregates, and relationships between the project's subsystems.

The system's core purpose is to transform CUDA GPU compute programs into portable representations (Rust, WGSL) that can execute on diverse hardware backends (WebGPU, Vulkan, Metal, CPU).

---

## 1. Bounded Contexts

The system is decomposed into eight bounded contexts, each with its own ubiquitous language, invariants, and internal models.

### 1.1 Parser Context

**Responsibility**: Lexical and syntactic analysis of CUDA C++ source code into a typed abstract syntax tree.

**Key Concepts**: Token, Lexeme, Grammar Rule, AST Node, Parse Error, Source Location

**Internal Modules**:
- `parser::lexer` -- Tokenization (logos)
- `parser::cuda_parser` -- Recursive descent parsing (nom)
- `parser::ast` -- AST type definitions
- `parser::kernel_extractor` -- Kernel-specific extraction utilities
- `parser::ptx_parser` -- PTX intermediate representation parsing

**Invariants**:
- A valid parse always produces a well-typed `Ast` with at least one `Item`
- All AST nodes preserve source location information for error reporting
- The parser must reject syntactically invalid CUDA code with descriptive errors

---

### 1.2 Transpiler Context

**Responsibility**: Transformation of CUDA AST into target-language representations (Rust source code, WGSL shader source).

**Key Concepts**: Transpilation Unit, Code Generator, Type Mapping, Memory Mapping, Built-in Function, Output Artifact

**Internal Modules**:
- `transpiler::code_generator` -- Rust code generation (quote/proc-macro2)
- `transpiler::wgsl` -- WGSL shader generation
- `transpiler::type_converter` -- CUDA-to-target type mapping
- `transpiler::memory_mapper` -- Memory space mapping (shared, global, constant)
- `transpiler::builtin_functions` -- CUDA built-in to target function mapping
- `transpiler::kernel_translator` -- Kernel-specific translation logic

**Invariants**:
- Transpilation preserves the computational semantics of the source kernel
- Type conversions are lossless where possible; lossy conversions emit warnings
- Generated code is syntactically valid in the target language

---

### 1.3 Runtime Context

**Responsibility**: Execution environment for transpiled kernels, providing CUDA-compatible abstractions for thread indexing, synchronization, and kernel launch.

**Key Concepts**: Device, Stream, Event, Launch Configuration, Thread Context, Grid, Block, Dim3

**Internal Modules**:
- `runtime::device` -- Device abstraction and selection
- `runtime::kernel` -- Kernel launch mechanics
- `runtime::stream` -- Asynchronous execution streams
- `runtime::event` -- Timing and synchronization events
- `runtime::grid` -- Grid/Block/Dim3 types
- `runtime::thread` / `runtime::block` -- Thread/block index access

**Invariants**:
- Thread indices are always within valid ranges for the launch configuration
- Kernel launch validates block dimensions against hardware limits
- Stream operations maintain FIFO ordering guarantees

---

### 1.4 Memory Context

**Responsibility**: Memory allocation, deallocation, pooling, and transfer between host and device address spaces.

**Key Concepts**: Device Buffer, Host Buffer, Unified Memory, Memory Pool, Allocation, Memcpy

**Internal Modules**:
- `memory::device_memory` -- GPU-side buffer management
- `memory::host_memory` -- CPU-side pinned buffer management
- `memory::unified_memory` -- Unified/managed memory abstraction
- `memory::memory_pool` -- Power-of-2 memory pool with pre-allocation

**Invariants**:
- Every allocation has a corresponding deallocation path (no resource leaks)
- Memory pool allocations are rounded to power-of-2 sizes
- Pool statistics accurately reflect allocation/deallocation history
- Kernel memory alignment guarantees are maintained

---

### 1.5 Backend Context

**Responsibility**: Hardware abstraction for executing compiled kernels on specific GPU APIs or CPU fallbacks.

**Key Concepts**: Backend, Capabilities, Kernel Compilation, Kernel Launch, Memory Operations, Synchronization

**Internal Modules**:
- `backend::backend_trait` -- Common interface (`BackendTrait`)
- `backend::webgpu` -- Browser WebGPU backend
- `backend::webgpu_optimized` -- Optimized browser WebGPU
- `backend::native_gpu` -- CUDA/OpenCL native backend
- `backend::wasm_runtime` -- CPU/WASM fallback backend
- `backend::wgpu_native` -- (Proposed) Native GPU via wgpu

**Invariants**:
- Backend selection follows a deterministic priority chain
- `BackendCapabilities` accurately reports hardware limits
- All backends implement the same `BackendTrait` interface

---

### 1.6 SIMD Context

**Responsibility**: Vectorized execution of kernel operations on CPU, using platform-specific SIMD instruction sets.

**Key Concepts**: SIMD Capability, Vector Register, Intrinsic, Feature Detection, Lane, Dispatch

**Internal Modules** (proposed by ADR-002):
- `simd::portable` -- Portable SIMD abstraction
- `simd::x86` -- AVX2/AVX-512 intrinsics
- `simd::arm` -- NEON intrinsics
- `simd::wasm` -- WASM SIMD128 intrinsics
- `simd::operations` -- SIMD-accelerated kernel primitives

**Invariants**:
- SIMD operations produce bit-identical results to scalar equivalents for integer operations
- Floating-point SIMD results match scalar results within IEEE 754 rounding tolerance
- Feature detection correctly identifies available instruction sets at runtime

---

### 1.7 Nutanix Platform Context

**Responsibility**: Integration with Nutanix HCI for GPU resource discovery, VM provisioning, and Kubernetes deployment.

**Key Concepts**: Prism Central, Cluster, Host, GPU Device, VM Template, NKE Deployment, GPU Passthrough

**Internal Modules** (proposed by ADR-004):
- `platform::nutanix::prism_client` -- REST API client
- `platform::nutanix::gpu_discovery` -- GPU resource enumeration
- `platform::nutanix::vm_templates` -- AHV VM specification
- `platform::nutanix::nke_deployment` -- Kubernetes manifest generation
- `platform::nutanix::monitoring` -- GPU metrics collection

**Invariants**:
- API calls are authenticated and use TLS
- GPU discovery reflects the current cluster state (no stale caching)
- VM specifications validate against cluster resource availability before provisioning

---

### 1.8 Profiling Context

**Responsibility**: Performance measurement, metrics collection, and reporting for all system operations.

**Key Concepts**: Profile, Timer, Metric, Counter, Duration, Memory Footprint, CSV Export

**Internal Modules**:
- `profiling::kernel_profiler` -- Kernel execution timing
- `profiling::memory_profiler` -- Memory allocation tracking
- `profiling::runtime_profiler` -- Runtime operation profiling
- `profiling::performance_monitor` -- Counters and reporting

**Invariants**:
- Profiling can be enabled/disabled at runtime without affecting correctness
- Duration measurements use monotonic clocks
- Memory tracking accounts for all allocations and deallocations
- CSV export produces valid, parseable output

---

## 2. Aggregates

### 2.1 KernelDef (Aggregate Root)

The central aggregate in the system. A `KernelDef` represents a complete CUDA kernel and is the primary unit of transpilation.

```
KernelDef (Aggregate Root)
    |
    +-- name: String
    +-- params: Vec<Parameter>
    |       +-- name: String
    |       +-- ty: Type (Value Object)
    |       +-- qualifiers: Vec<ParamQualifier>
    |
    +-- body: Block
    |       +-- statements: Vec<Statement>
    |               +-- VarDecl { name, ty, init, storage }
    |               +-- If { condition, then_branch, else_branch }
    |               +-- For { init, condition, update, body }
    |               +-- While { condition, body }
    |               +-- Expr(Expression)
    |               +-- SyncThreads
    |               +-- Return, Break, Continue
    |
    +-- attributes: Vec<KernelAttribute>
            +-- LaunchBounds { max_threads, min_blocks }
            +-- MaxRegisters(u32)
```

**Invariants**:
- A kernel always has a non-empty name
- All parameter names within a kernel are unique
- The body is a valid sequence of statements
- Launch bounds, if specified, are positive integers

**Lifecycle**:
1. Created by the Parser from CUDA source
2. Consumed by the Transpiler to produce target code
3. Never modified after creation (immutable value)

---

### 2.2 TranspilationUnit (Aggregate Root)

Represents a complete CUDA source file being transpiled, containing multiple kernels, device functions, global variables, and type definitions.

```
TranspilationUnit (Aggregate Root)
    |
    +-- Ast
    |   +-- items: Vec<Item>
    |       +-- Kernel(KernelDef)
    |       +-- DeviceFunction(FunctionDef)
    |       +-- HostFunction(FunctionDef)
    |       +-- GlobalVar(GlobalVar)
    |       +-- TypeDef(TypeDef)
    |       +-- Include(String)
    |
    +-- source_path: Option<PathBuf>
    +-- target_backend: BackendType
    +-- transpilation_options: TranspilationOptions
```

**Invariants**:
- A transpilation unit must contain at least one kernel or function
- All type references within the unit must resolve to defined types
- Include directives are recorded but not recursively resolved

---

### 2.3 ExecutionContext (Aggregate Root)

Represents the runtime state for executing a transpiled kernel, including device selection, stream management, and launch configuration.

```
ExecutionContext (Aggregate Root)
    |
    +-- Runtime
    |   +-- device: Arc<Device>
    |   +-- default_stream: Stream
    |
    +-- LaunchConfig
    |   +-- grid: Grid (contains Dim3)
    |   +-- block: Block (contains Dim3)
    |   +-- shared_memory_size: usize
    |   +-- stream: Option<Stream>
    |
    +-- Backend (via BackendTrait)
        +-- capabilities: BackendCapabilities
```

**Invariants**:
- A device is always selected before kernel launch
- Launch configuration validates against device capabilities
- Streams provide ordered execution within a context

---

### 2.4 MemoryPool (Aggregate Root)

Manages a pool of reusable memory allocations organized by size class.

```
MemoryPool (Aggregate Root)
    |
    +-- pools: HashMap<usize, Vec<Vec<u8>>>
    |       Key: power-of-2 size class
    |       Value: available buffers of that size
    |
    +-- config: PoolConfig
    |   +-- max_pool_size: usize
    |   +-- min_pooled_size: usize
    |   +-- max_pooled_size: usize
    |   +-- prealloc_count: usize
    |
    +-- stats: PoolStats
        +-- total_allocations: u64
        +-- cache_hits: u64
        +-- cache_misses: u64
        +-- peak_memory_usage: usize
```

**Invariants**:
- Pool sizes are always powers of 2
- Cache hit ratio is calculated as `hits / total_allocations`
- Pre-allocated buffers are initialized to zero
- Pool size per class does not exceed `max_pool_size / class_size`

---

## 3. Value Objects

Value objects are immutable types defined by their attributes rather than identity.

### 3.1 Dim3

A three-dimensional index or size, analogous to CUDA's `dim3`.

```rust
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
```

**Properties**: Immutable, equality by value, supports conversion from `u32`, `(u32, u32)`, `(u32, u32, u32)`.

### 3.2 Type

A CUDA type representation, used throughout the AST and code generators.

```rust
pub enum Type {
    Void, Bool,
    Int(IntType),           // I8, I16, I32, I64, U8, U16, U32, U64
    Float(FloatType),       // F16, F32, F64
    Pointer(Box<Type>),
    Array(Box<Type>, Option<usize>),
    Vector(VectorType),     // float4, int2, etc.
    Named(String),
    Texture(TextureType),
}
```

**Properties**: Recursive, supports arbitrary nesting (pointer to array of float4), equality by structural comparison.

### 3.3 Expression

An AST expression node representing a computation.

```rust
pub enum Expression {
    Literal(Literal), Var(String),
    Binary { op, left, right },
    Unary { op, expr },
    Call { name, args },
    Index { array, index },
    Member { object, field },
    Cast { ty, expr },
    ThreadIdx(Dimension), BlockIdx(Dimension),
    BlockDim(Dimension), GridDim(Dimension),
    WarpPrimitive { op, args },
    AtomicOperation { op, address, value, compare, ordering },
}
```

**Properties**: Tree-structured, immutable after construction, supports visitor pattern traversal.

### 3.4 BackendCapabilities

Describes the features and limits of a specific GPU backend.

```rust
pub struct BackendCapabilities {
    pub name: String,
    pub supports_cuda: bool,
    pub supports_opencl: bool,
    pub supports_vulkan: bool,
    pub supports_webgpu: bool,
    pub max_threads: u32,
    pub max_threads_per_block: u32,
    pub max_blocks_per_grid: u32,
    pub max_shared_memory: usize,
    pub supports_dynamic_parallelism: bool,
    pub supports_unified_memory: bool,
    pub max_grid_dim: [u32; 3],
    pub max_block_dim: [u32; 3],
    pub warp_size: u32,
}
```

**Properties**: Immutable snapshot of hardware capabilities, used for validation and code generation decisions.

---

## 4. Domain Events

Domain events represent significant occurrences within the system that other bounded contexts may need to react to.

### 4.1 KernelParsed

**Trigger**: Parser successfully constructs a `KernelDef` from source code.

**Payload**:
```rust
pub struct KernelParsed {
    pub kernel_name: String,
    pub param_count: usize,
    pub statement_count: usize,
    pub uses_shared_memory: bool,
    pub uses_warp_primitives: bool,
    pub uses_atomics: bool,
    pub source_location: SourceLocation,
    pub parse_duration: Duration,
}
```

**Consumers**: Profiling Context (parse timing), Transpiler Context (initiates transpilation)

---

### 4.2 TranspilationComplete

**Trigger**: Transpiler produces target-language output from an AST.

**Payload**:
```rust
pub struct TranspilationComplete {
    pub kernel_name: String,
    pub target: TranspilationTarget, // Rust, WGSL
    pub output_size_bytes: usize,
    pub warnings: Vec<TranspilationWarning>,
    pub duration: Duration,
    pub optimizations_applied: Vec<String>,
}
```

**Consumers**: Profiling Context (transpilation metrics), Backend Context (compiled artifact), Caching (store result)

---

### 4.3 KernelLaunched

**Trigger**: A transpiled kernel is dispatched for execution on a backend.

**Payload**:
```rust
pub struct KernelLaunched {
    pub kernel_name: String,
    pub backend: String,
    pub grid_dim: Dim3,
    pub block_dim: Dim3,
    pub shared_memory_bytes: usize,
    pub argument_count: usize,
    pub launch_timestamp: Instant,
}
```

**Consumers**: Profiling Context (launch tracking), Monitoring (utilization)

---

### 4.4 MemoryAllocated

**Trigger**: Device or host memory is allocated.

**Payload**:
```rust
pub struct MemoryAllocated {
    pub size_bytes: usize,
    pub pool_hit: bool,
    pub pool_size_class: Option<usize>,
    pub allocation_type: AllocationType, // Device, Host, Unified, Shared
    pub alignment: usize,
    pub timestamp: Instant,
}
```

**Consumers**: Profiling Context (memory tracking), Memory Pool (statistics)

---

### 4.5 BackendSelected

**Trigger**: The backend selection logic chooses a compute backend.

**Payload**:
```rust
pub struct BackendSelected {
    pub backend_name: String,
    pub capabilities: BackendCapabilities,
    pub selection_reason: String, // "CUDA available", "wgpu Vulkan fallback", "CPU fallback"
    pub alternatives_considered: Vec<String>,
}
```

**Consumers**: Profiling Context (backend selection logging)

---

## 5. Domain Services

### 5.1 TranspilationService

**Responsibility**: Orchestrates the end-to-end transpilation pipeline from CUDA source to target output.

```rust
pub struct TranspilationService {
    parser: CudaParser,
    transpiler: Transpiler,
}

impl TranspilationService {
    /// Parse and transpile CUDA source to Rust
    pub fn transpile_to_rust(&self, source: &str) -> Result<String>;

    /// Parse and transpile CUDA source to WGSL
    pub fn transpile_to_wgsl(&self, source: &str) -> Result<String>;

    /// Parse CUDA source to AST only (for inspection)
    pub fn parse_only(&self, source: &str) -> Result<Ast>;

    /// Transpile with options (optimization level, target features)
    pub fn transpile_with_options(&self, source: &str, options: &TranspilationOptions) -> Result<TranspilationResult>;
}
```

**Current Implementation**: `CudaRust` struct in `src/lib.rs` and `CudaTranspiler` in `src/transpiler/mod.rs`.

---

### 5.2 BackendSelectionService

**Responsibility**: Selects the optimal compute backend based on platform, available hardware, and user preferences.

```rust
pub struct BackendSelectionService;

impl BackendSelectionService {
    /// Select the best available backend
    pub fn select_backend() -> Box<dyn BackendTrait>;

    /// Select a specific backend by name
    pub fn select_by_name(name: &str) -> Result<Box<dyn BackendTrait>>;

    /// List all available backends with their capabilities
    pub fn available_backends() -> Vec<BackendInfo>;
}
```

**Current Implementation**: `get_backend()` function in `src/backend/mod.rs`.

---

### 5.3 OptimizationService

**Responsibility**: Applies optimization passes to the AST or generated code.

```rust
pub struct OptimizationService;

impl OptimizationService {
    /// Apply dead code elimination
    pub fn eliminate_dead_code(ast: &mut Ast);

    /// Optimize memory access patterns
    pub fn optimize_memory_access(ast: &mut Ast);

    /// Fuse sequential operations where possible
    pub fn fuse_operations(ast: &mut Ast);

    /// Select optimal workgroup size for target backend
    pub fn optimize_workgroup_size(kernel: &KernelDef, capabilities: &BackendCapabilities) -> Dim3;
}
```

**Current Implementation**: Not yet implemented. Optimization is a future concern.

---

## 6. Repositories

### 6.1 KernelRepository

**Responsibility**: Storage and retrieval of parsed kernel definitions.

```rust
pub trait KernelRepository {
    /// Store a parsed kernel
    fn store(&mut self, kernel: KernelDef);

    /// Retrieve a kernel by name
    fn get(&self, name: &str) -> Option<&KernelDef>;

    /// List all stored kernel names
    fn list(&self) -> Vec<&str>;

    /// Remove a kernel by name
    fn remove(&mut self, name: &str) -> Option<KernelDef>;
}
```

**Implementation**: In-memory `HashMap<String, KernelDef>`. No persistent storage currently required.

---

### 6.2 CompiledKernelCache

**Responsibility**: Caching compiled kernel artifacts to avoid redundant transpilation and compilation.

```rust
pub trait CompiledKernelCache {
    /// Cache a compiled kernel artifact
    fn store(&mut self, key: &CacheKey, artifact: CompiledArtifact);

    /// Retrieve a cached artifact
    fn get(&self, key: &CacheKey) -> Option<&CompiledArtifact>;

    /// Invalidate cache entries for a specific kernel
    fn invalidate(&mut self, kernel_name: &str);

    /// Clear all cached artifacts
    fn clear(&mut self);
}

pub struct CacheKey {
    pub kernel_name: String,
    pub source_hash: u64,
    pub target_backend: String,
    pub optimization_level: u8,
}

pub struct CompiledArtifact {
    pub compiled_bytes: Vec<u8>,
    pub target: String,
    pub compile_time: Duration,
    pub created_at: Instant,
}
```

**Implementation**: In-memory LRU cache. Could be extended to disk-based caching for large projects.

---

## 7. Anti-Corruption Layers (ACLs)

ACLs protect bounded contexts from external model contamination by translating between external and internal representations.

### 7.1 CUDAToCoreACL

**Purpose**: Translates CUDA-specific concepts into the core domain model.

**Translations**:

| CUDA Concept | Core Domain Concept |
|--------------|-------------------|
| `__global__ void kernel(...)` | `KernelDef` with `Item::Kernel` |
| `__device__ T func(...)` | `FunctionDef` with `Item::DeviceFunction` |
| `__shared__ T var` | `StorageClass::Shared` |
| `threadIdx.x` | `Expression::ThreadIdx(Dimension::X)` |
| `__syncthreads()` | `Statement::SyncThreads` |
| `atomicAdd(&x, v)` | `Expression::AtomicOperation { op: AtomicOp::Add, ... }` |
| `__shfl_sync(mask, val, lane)` | `Expression::WarpPrimitive { op: WarpOp::Shuffle, ... }` |
| `<<<grid, block>>>` | `LaunchConfig { grid, block }` |

**Location**: Implemented within the Parser Context (`src/parser/cuda_parser.rs`).

---

### 7.2 WGSLToCoreACL

**Purpose**: Translates core domain model concepts into WGSL-specific representations.

**Translations**:

| Core Domain Concept | WGSL Representation |
|--------------------|--------------------|
| `KernelDef` | `@compute @workgroup_size(X,Y,Z) fn name(...)` |
| `StorageClass::Shared` | `var<workgroup>` |
| `StorageClass::Global` (pointer param) | `@group(0) @binding(N) var<storage, read_write>` |
| `Statement::SyncThreads` | `workgroupBarrier()` |
| `Expression::ThreadIdx(X)` | `local_invocation_id.x` (via alias) |
| `Expression::BlockIdx(X)` | `workgroup_id.x` (via alias) |
| `Type::Int(I32)` | `i32` |
| `Type::Float(F64)` | Error: f64 not supported in WGSL |
| `AtomicOp::Add` on `i32` | `atomicAdd(...)` |
| `AtomicOp::Add` on `f32` | CAS-loop emulation with `bitcast` |

**Location**: Implemented within the Transpiler Context (`src/transpiler/wgsl.rs`).

---

### 7.3 NutanixAPIACL

**Purpose**: Translates Nutanix Prism Central API responses into the platform domain model.

**Translations**:

| Nutanix API Response | Platform Domain Concept |
|---------------------|------------------------|
| `hosts/list` response | `Vec<GpuHost>` |
| `host.gpu_list[]` | `Vec<GpuDevice>` with vendor, model, VRAM |
| `vms/list` with `gpu_list` filter | `Vec<GpuVm>` |
| VM create task response | `TaskReference` with UUID and status |
| `host_stats.gpu_usage_ppm` | `GpuMetrics` with utilization percentage |
| Cluster configuration | `ClusterCapabilities` with GPU inventory |

**Location**: Implemented within the Nutanix Platform Context (`src/platform/nutanix/prism_client.rs`).

---

## 8. Context Map

The context map shows relationships between bounded contexts using DDD relationship patterns.

```
                              +-----------------+
                              |    Profiling    |
                              |    Context      |
                              +--------+--------+
                                       |
                            Published Language (Events)
                                       |
         +-----------------------------+-----------------------------+
         |                             |                             |
         v                             v                             v
+--------+--------+          +--------+--------+          +--------+--------+
|     Parser      |          |   Transpiler    |          |    Runtime      |
|     Context     +--------->+    Context      +--------->+    Context      |
|                 | Customer |                 | Customer |                 |
+--------+--------+ Supplier +--------+--------+ Supplier +--------+--------+
         |                             |                             |
         |                    +--------+--------+                    |
         |                    |                 |                    |
         |              +-----v------+   +------v-----+             |
         |              |   WGSL     |   |   Rust     |             |
         |              |   ACL      |   |   ACL      |             |
         |              +-----+------+   +------+-----+             |
         |                    |                 |                    |
    CUDA ACL                  v                 v              Backend
    (Conformist)      +-------+-------+  +------+------+    Selection
         |            |    Backend    |  |    SIMD     |       |
         |            |    Context    |  |   Context   |       |
         |            +-------+-------+  +------+------+       |
         |                    |                 |              |
         |                    |                 v              |
         |                    |          CPU Fallback          |
         |                    |                                |
         |              +-----v----------+                     |
         |              |   Nutanix      |                     |
         |              |   Platform     |                     |
         |              |   Context      |                     |
         |              +----+-----------+                     |
         |                   |                                 |
         |              Nutanix API ACL                        |
         |              (Anti-Corruption)                      |
         |                                                     |
         +-----------------------------------------------------+
                    Shared Kernel: AST Types
```

### Relationship Patterns

| Upstream | Downstream | Pattern | Description |
|----------|-----------|---------|-------------|
| Parser | Transpiler | **Customer-Supplier** | Transpiler depends on Parser's AST output. Parser defines the AST schema; Transpiler consumes it. |
| Transpiler | Backend | **Customer-Supplier** | Backend consumes compiled/generated code from Transpiler. |
| Transpiler | Runtime | **Customer-Supplier** | Runtime provides the execution abstractions that generated code calls into. |
| CUDA Source | Parser | **Conformist** | Parser conforms to CUDA's grammar. CUDA is an external standard; the parser adapts. |
| Core Domain | WGSL | **Anti-Corruption Layer** | WGSL generator translates core AST concepts to WGSL-specific representations. |
| Core Domain | Nutanix API | **Anti-Corruption Layer** | Nutanix client translates API responses to domain types. |
| Profiling | All Contexts | **Published Language** | Profiling defines domain events that all contexts can emit. |
| Parser + Transpiler | All | **Shared Kernel** | AST types (`src/parser/ast.rs`) are shared across Parser, Transpiler, and Code Generators. |

---

## 9. Module Dependency Diagram

```
lib.rs (CudaRust)
  |
  +-- parser/
  |     +-- cuda_parser.rs    (CudaParser)
  |     +-- ast.rs            (Ast, KernelDef, Expression, Statement, Type, ...)
  |     +-- lexer.rs          (Token types and lexer)
  |     +-- kernel_extractor.rs
  |     +-- ptx_parser.rs
  |
  +-- transpiler/
  |     +-- mod.rs            (Transpiler, CudaTranspiler)
  |     +-- code_generator.rs (CodeGenerator -- Rust output)
  |     +-- wgsl.rs           (WgslGenerator -- WGSL output)
  |     +-- type_converter.rs
  |     +-- memory_mapper.rs
  |     +-- builtin_functions.rs
  |     +-- kernel_translator.rs
  |     +-- ast.rs            (Legacy AST -- to be deprecated)
  |
  +-- runtime/
  |     +-- mod.rs            (Runtime)
  |     +-- device.rs         (Device, BackendType)
  |     +-- kernel.rs         (launch_kernel, LaunchConfig)
  |     +-- stream.rs         (Stream)
  |     +-- event.rs          (Event)
  |     +-- grid.rs           (Grid, Block, Dim3)
  |
  +-- memory/
  |     +-- device_memory.rs  (DeviceBuffer)
  |     +-- host_memory.rs    (HostBuffer)
  |     +-- unified_memory.rs (UnifiedMemory)
  |     +-- memory_pool.rs    (MemoryPool, KernelMemoryManager)
  |
  +-- backend/
  |     +-- backend_trait.rs  (BackendTrait, BackendCapabilities, MemcpyKind)
  |     +-- webgpu.rs         (WebGPUBackend)
  |     +-- webgpu_optimized.rs
  |     +-- native_gpu.rs     (NativeGPUBackend)
  |     +-- wasm_runtime.rs   (WasmRuntime)
  |
  +-- kernel/
  |     +-- grid.rs
  |     +-- thread.rs
  |     +-- shared_memory.rs
  |     +-- warp.rs
  |
  +-- profiling/
  |     +-- kernel_profiler.rs
  |     +-- memory_profiler.rs
  |     +-- runtime_profiler.rs
  |     +-- performance_monitor.rs
  |
  +-- neural_integration/     (ruv-FANN bridge)
  +-- utils/
  +-- error.rs                (CudaRustError, Result)
```

---

## 10. Glossary Cross-Reference

For the complete ubiquitous language glossary referenced throughout this document, see [ubiquitous-language.md](./ubiquitous-language.md).
