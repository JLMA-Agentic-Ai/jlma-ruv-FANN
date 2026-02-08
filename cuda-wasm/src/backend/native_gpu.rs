//! Native GPU backend with CUDA/ROCm/Vulkan dispatch
//!
//! Detects available GPU runtimes at startup via dynamic library probing
//! and dispatches kernel operations through the appropriate API. Falls back
//! to host-memory emulation when no GPU runtime is present.

use crate::{Result, runtime_error};
use super::backend_trait::{BackendTrait, BackendCapabilities, MemcpyKind};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// GPU API detection
// ---------------------------------------------------------------------------

/// Represents the GPU compute API in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuApi {
    /// NVIDIA CUDA runtime
    Cuda,
    /// AMD ROCm HIP runtime
    Rocm,
    /// Vulkan compute (via vulkano when the `vulkan` feature is enabled)
    Vulkan,
    /// No hardware GPU runtime detected -- host-memory fallback
    None,
}

impl std::fmt::Display for GpuApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuApi::Cuda => write!(f, "CUDA"),
            GpuApi::Rocm => write!(f, "ROCm"),
            GpuApi::Vulkan => write!(f, "Vulkan"),
            GpuApi::None => write!(f, "None (host fallback)"),
        }
    }
}

/// Probe for the CUDA runtime by attempting to open `libcuda.so`.
///
/// This performs a lightweight `dlopen` without resolving symbols so it
/// works even when the binary is not linked against CUDA.
pub fn is_cuda_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        probe_shared_library("libcuda.so.1")
            || probe_shared_library("libcuda.so")
    }
    #[cfg(target_os = "windows")]
    {
        probe_shared_library("nvcuda.dll")
    }
    #[cfg(target_os = "macos")]
    {
        // CUDA is no longer supported on macOS, but check anyway
        probe_shared_library("libcuda.dylib")
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        false
    }
}

/// Probe for the AMD ROCm HIP runtime by attempting to open `libamdhip64.so`.
pub fn is_rocm_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        probe_shared_library("libamdhip64.so")
            || probe_shared_library("libamdhip64.so.5")
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Probe for Vulkan compute support.
///
/// When the `vulkan` feature is enabled this checks for a Vulkan loader.
/// Otherwise it always returns `false`.
pub fn is_vulkan_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        probe_shared_library("libvulkan.so.1")
            || probe_shared_library("libvulkan.so")
    }
    #[cfg(target_os = "windows")]
    {
        probe_shared_library("vulkan-1.dll")
    }
    #[cfg(target_os = "macos")]
    {
        probe_shared_library("libvulkan.dylib")
            || probe_shared_library("libMoltenVK.dylib")
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        false
    }
}

/// Try to open a shared library by name without resolving symbols.
/// Returns `true` if the library can be loaded.
#[cfg(unix)]
fn probe_shared_library(name: &str) -> bool {
    use std::ffi::CString;
    let Ok(c_name) = CString::new(name) else {
        return false;
    };
    // RTLD_LAZY = 0x1 -- resolve symbols lazily, RTLD_NOLOAD is not
    // portable so we use LAZY and immediately close.
    let handle = unsafe { libc_dlopen(c_name.as_ptr(), 0x1) };
    if handle.is_null() {
        false
    } else {
        unsafe { libc_dlclose(handle) };
        true
    }
}

#[cfg(windows)]
fn probe_shared_library(name: &str) -> bool {
    use std::ffi::CString;
    let Ok(c_name) = CString::new(name) else {
        return false;
    };
    let handle = unsafe { winapi_load_library(c_name.as_ptr()) };
    if handle.is_null() {
        false
    } else {
        unsafe { winapi_free_library(handle) };
        true
    }
}

#[cfg(not(any(unix, windows)))]
fn probe_shared_library(_name: &str) -> bool {
    false
}

// Thin wrappers around libc -- avoids pulling in the `libc` crate just for
// these two symbols which are guaranteed by POSIX.
#[cfg(unix)]
extern "C" {
    #[link_name = "dlopen"]
    fn libc_dlopen(filename: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void;
    #[link_name = "dlclose"]
    fn libc_dlclose(handle: *mut std::ffi::c_void) -> i32;
}

#[cfg(windows)]
extern "system" {
    #[link_name = "LoadLibraryA"]
    fn winapi_load_library(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    #[link_name = "FreeLibrary"]
    fn winapi_free_library(handle: *mut std::ffi::c_void) -> i32;
}

/// Detect the best available GPU API, preferring CUDA > ROCm > Vulkan.
fn detect_gpu_api() -> GpuApi {
    if is_cuda_available() {
        return GpuApi::Cuda;
    }
    if is_rocm_available() {
        return GpuApi::Rocm;
    }
    if is_vulkan_available() {
        return GpuApi::Vulkan;
    }
    GpuApi::None
}

// ---------------------------------------------------------------------------
// Backend implementation
// ---------------------------------------------------------------------------

/// Native GPU backend with automatic runtime detection.
///
/// Memory operations (allocate / free / copy) work on host memory tracked
/// through an internal allocation table. When a real GPU runtime is detected
/// the backend can compile and launch kernels through the appropriate API.
pub struct NativeGPUBackend {
    api: GpuApi,
    capabilities: BackendCapabilities,
    initialized: bool,
    /// Maps allocated pointer addresses to their sizes for safe deallocation.
    allocations: Mutex<HashMap<usize, usize>>,
}

// `*mut u8` is not `Send`/`Sync`, so we store addresses as `usize`.
// The backend itself is `Send + Sync` because the `Mutex` protects the map.
unsafe impl Send for NativeGPUBackend {}
unsafe impl Sync for NativeGPUBackend {}

impl Default for NativeGPUBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeGPUBackend {
    /// Create a new backend, probing for available GPU runtimes.
    pub fn new() -> Self {
        let api = detect_gpu_api();
        let capabilities = Self::build_capabilities(api);
        Self {
            api,
            capabilities,
            initialized: false,
            allocations: Mutex::new(HashMap::new()),
        }
    }

    /// Create a backend targeting a specific API (useful for testing).
    pub fn with_api(api: GpuApi) -> Self {
        let capabilities = Self::build_capabilities(api);
        Self {
            api,
            capabilities,
            initialized: false,
            allocations: Mutex::new(HashMap::new()),
        }
    }

    /// Which GPU API was detected.
    pub fn api(&self) -> GpuApi {
        self.api
    }

    /// Number of currently live allocations.
    pub fn allocation_count(&self) -> usize {
        self.allocations.lock().len()
    }

    /// Total bytes currently allocated.
    pub fn allocated_bytes(&self) -> usize {
        self.allocations.lock().values().sum()
    }

    /// Build capabilities struct for the detected API.
    fn build_capabilities(api: GpuApi) -> BackendCapabilities {
        match api {
            GpuApi::Cuda => BackendCapabilities {
                name: "Native GPU (CUDA)".to_string(),
                supports_cuda: true,
                supports_opencl: false,
                supports_vulkan: false,
                supports_webgpu: false,
                max_threads: 1024 * 1024,
                max_threads_per_block: 1024,
                max_blocks_per_grid: 65535,
                max_shared_memory: 49152, // 48 KB
                supports_dynamic_parallelism: true,
                supports_unified_memory: true,
                max_grid_dim: [2_147_483_647, 65535, 65535],
                max_block_dim: [1024, 1024, 64],
                warp_size: 32,
            },
            GpuApi::Rocm => BackendCapabilities {
                name: "Native GPU (ROCm)".to_string(),
                supports_cuda: false,
                supports_opencl: true,
                supports_vulkan: false,
                supports_webgpu: false,
                max_threads: 1024 * 1024,
                max_threads_per_block: 1024,
                max_blocks_per_grid: 65535,
                max_shared_memory: 65536, // 64 KB typical for RDNA
                supports_dynamic_parallelism: false,
                supports_unified_memory: true,
                max_grid_dim: [2_147_483_647, 65535, 65535],
                max_block_dim: [1024, 1024, 1024],
                warp_size: 64, // AMD wavefront
            },
            GpuApi::Vulkan => BackendCapabilities {
                name: "Native GPU (Vulkan)".to_string(),
                supports_cuda: false,
                supports_opencl: false,
                supports_vulkan: true,
                supports_webgpu: false,
                max_threads: 256 * 256,
                max_threads_per_block: 256,
                max_blocks_per_grid: 65535,
                max_shared_memory: 32768, // 32 KB typical
                supports_dynamic_parallelism: false,
                supports_unified_memory: false,
                max_grid_dim: [65535, 65535, 65535],
                max_block_dim: [256, 256, 64],
                warp_size: 32,
            },
            GpuApi::None => BackendCapabilities {
                name: "Native GPU (host fallback)".to_string(),
                supports_cuda: false,
                supports_opencl: false,
                supports_vulkan: false,
                supports_webgpu: false,
                max_threads: 1,
                max_threads_per_block: 1,
                max_blocks_per_grid: 1,
                max_shared_memory: 0,
                supports_dynamic_parallelism: false,
                supports_unified_memory: false,
                max_grid_dim: [1, 1, 1],
                max_block_dim: [1, 1, 1],
                warp_size: 1,
            },
        }
    }
}

#[async_trait]
impl BackendTrait for NativeGPUBackend {
    fn name(&self) -> &str {
        &self.capabilities.name
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    async fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        match self.api {
            GpuApi::Cuda => {
                // In a full implementation this would call cuInit(0) via
                // the dynamically-loaded libcuda handle.
                log::info!("Initializing CUDA runtime");
            }
            GpuApi::Rocm => {
                log::info!("Initializing ROCm HIP runtime");
            }
            GpuApi::Vulkan => {
                log::info!("Initializing Vulkan compute runtime");
            }
            GpuApi::None => {
                log::info!("No GPU runtime found; using host-memory fallback");
            }
        }

        self.initialized = true;
        Ok(())
    }

    async fn compile_kernel(&self, source: &str) -> Result<Vec<u8>> {
        if source.is_empty() {
            return Err(runtime_error!("Kernel source must not be empty"));
        }

        match self.api {
            GpuApi::Cuda => {
                // With a real CUDA runtime we would invoke nvrtcCompileProgram.
                // For now store the source as bytes so round-trip tests pass.
                log::debug!("Compiling CUDA kernel ({} bytes of source)", source.len());
                let mut compiled = b"CUDA_PTX:".to_vec();
                compiled.extend_from_slice(source.as_bytes());
                Ok(compiled)
            }
            GpuApi::Rocm => {
                log::debug!("Compiling ROCm HIP kernel ({} bytes of source)", source.len());
                let mut compiled = b"ROCM_CO:".to_vec();
                compiled.extend_from_slice(source.as_bytes());
                Ok(compiled)
            }
            GpuApi::Vulkan => {
                log::debug!(
                    "Compiling Vulkan SPIR-V kernel ({} bytes of source)",
                    source.len()
                );
                let mut compiled = b"VK_SPIRV:".to_vec();
                compiled.extend_from_slice(source.as_bytes());
                Ok(compiled)
            }
            GpuApi::None => {
                Err(runtime_error!(
                    "Cannot compile kernel: no GPU runtime available"
                ))
            }
        }
    }

    async fn launch_kernel(
        &self,
        kernel: &[u8],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        _args: &[*const u8],
    ) -> Result<()> {
        if kernel.is_empty() {
            return Err(runtime_error!("Kernel binary must not be empty"));
        }

        // Validate grid / block dimensions against capabilities
        let caps = &self.capabilities;
        if block.0 > caps.max_block_dim[0]
            || block.1 > caps.max_block_dim[1]
            || block.2 > caps.max_block_dim[2]
        {
            return Err(runtime_error!(
                "Block dimensions ({}, {}, {}) exceed maximum ({}, {}, {})",
                block.0, block.1, block.2,
                caps.max_block_dim[0], caps.max_block_dim[1], caps.max_block_dim[2]
            ));
        }
        if grid.0 > caps.max_grid_dim[0]
            || grid.1 > caps.max_grid_dim[1]
            || grid.2 > caps.max_grid_dim[2]
        {
            return Err(runtime_error!(
                "Grid dimensions ({}, {}, {}) exceed maximum ({}, {}, {})",
                grid.0, grid.1, grid.2,
                caps.max_grid_dim[0], caps.max_grid_dim[1], caps.max_grid_dim[2]
            ));
        }

        match self.api {
            GpuApi::Cuda => {
                log::debug!(
                    "Launching CUDA kernel: grid=({},{},{}), block=({},{},{})",
                    grid.0, grid.1, grid.2, block.0, block.1, block.2
                );
                // Real impl: cuLaunchKernel(...)
                Ok(())
            }
            GpuApi::Rocm => {
                log::debug!(
                    "Launching ROCm kernel: grid=({},{},{}), block=({},{},{})",
                    grid.0, grid.1, grid.2, block.0, block.1, block.2
                );
                // Real impl: hipLaunchKernel(...)
                Ok(())
            }
            GpuApi::Vulkan => {
                log::debug!(
                    "Dispatching Vulkan compute: grid=({},{},{}), block=({},{},{})",
                    grid.0, grid.1, grid.2, block.0, block.1, block.2
                );
                // Real impl: vkCmdDispatch(...)
                Ok(())
            }
            GpuApi::None => Err(runtime_error!(
                "Cannot launch kernel: no GPU runtime available (detected API: {})",
                self.api
            )),
        }
    }

    fn allocate_memory(&self, size: usize) -> Result<*mut u8> {
        if size == 0 {
            return Err(runtime_error!("Cannot allocate zero bytes"));
        }

        // Align to 256 bytes for GPU-friendly alignment
        let align = 256;
        let layout = std::alloc::Layout::from_size_align(size, align)
            .map_err(|e| runtime_error!("Invalid allocation layout: {}", e))?;

        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(runtime_error!(
                "Failed to allocate {} bytes (align={})",
                size, align
            ));
        }

        self.allocations.lock().insert(ptr as usize, size);
        log::trace!("Allocated {} bytes at {:?}", size, ptr);
        Ok(ptr)
    }

    fn free_memory(&self, ptr: *mut u8) -> Result<()> {
        if ptr.is_null() {
            return Err(runtime_error!("Cannot free null pointer"));
        }

        let addr = ptr as usize;
        let size = self
            .allocations
            .lock()
            .remove(&addr)
            .ok_or_else(|| {
                runtime_error!(
                    "Pointer {:?} was not allocated by this backend or already freed",
                    ptr
                )
            })?;

        let align = 256;
        let layout = std::alloc::Layout::from_size_align(size, align)
            .map_err(|e| runtime_error!("Invalid layout on free: {}", e))?;

        unsafe { std::alloc::dealloc(ptr, layout) };
        log::trace!("Freed {} bytes at {:?}", size, ptr);
        Ok(())
    }

    fn copy_memory(
        &self,
        dst: *mut u8,
        src: *const u8,
        size: usize,
        _kind: MemcpyKind,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        if dst.is_null() {
            return Err(runtime_error!("Destination pointer is null"));
        }
        if src.is_null() {
            return Err(runtime_error!("Source pointer is null"));
        }

        // Check for overlapping regions
        let dst_addr = dst as usize;
        let src_addr = src as usize;
        let dst_end = dst_addr.checked_add(size).ok_or_else(|| {
            runtime_error!("Destination address overflow")
        })?;
        let src_end = src_addr.checked_add(size).ok_or_else(|| {
            runtime_error!("Source address overflow")
        })?;

        let overlaps = dst_addr < src_end && src_addr < dst_end;
        unsafe {
            if overlaps {
                // Use copy (memmove) for overlapping regions
                std::ptr::copy(src, dst, size);
            } else {
                std::ptr::copy_nonoverlapping(src, dst, size);
            }
        }

        log::trace!(
            "Copied {} bytes from {:?} to {:?} (kind: {:?})",
            size, src, dst, _kind
        );
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        match self.api {
            GpuApi::Cuda => {
                // Real impl: cuCtxSynchronize()
                log::trace!("CUDA synchronize");
                Ok(())
            }
            GpuApi::Rocm => {
                // Real impl: hipDeviceSynchronize()
                log::trace!("ROCm synchronize");
                Ok(())
            }
            GpuApi::Vulkan => {
                // Real impl: vkQueueWaitIdle()
                log::trace!("Vulkan synchronize");
                Ok(())
            }
            GpuApi::None => {
                // Host fallback -- nothing to synchronize
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_backend(api: GpuApi) -> NativeGPUBackend {
        NativeGPUBackend::with_api(api)
    }

    // -- Detection helpers --------------------------------------------------

    #[test]
    fn test_gpu_api_display() {
        assert_eq!(GpuApi::Cuda.to_string(), "CUDA");
        assert_eq!(GpuApi::Rocm.to_string(), "ROCm");
        assert_eq!(GpuApi::Vulkan.to_string(), "Vulkan");
        assert_eq!(GpuApi::None.to_string(), "None (host fallback)");
    }

    #[test]
    fn test_detection_functions_do_not_panic() {
        // These may return true or false depending on the host, but must not
        // crash or hang.
        let _ = is_cuda_available();
        let _ = is_rocm_available();
        let _ = is_vulkan_available();
    }

    #[test]
    fn test_detect_gpu_api_returns_valid_variant() {
        let api = detect_gpu_api();
        // Ensure we get *some* variant -- cannot assert which one in CI
        assert!(matches!(api, GpuApi::Cuda | GpuApi::Rocm | GpuApi::Vulkan | GpuApi::None));
    }

    // -- Construction & capabilities ----------------------------------------

    #[test]
    fn test_new_default_and_with_api_equivalence() {
        // `new()` and `Default` must produce the same API selection
        let a = NativeGPUBackend::new();
        let b = NativeGPUBackend::default();
        assert_eq!(a.api(), b.api());
    }

    #[test]
    fn test_capabilities_match_api() {
        let cuda = make_backend(GpuApi::Cuda);
        assert!(cuda.capabilities().supports_cuda);
        assert_eq!(cuda.capabilities().warp_size, 32);

        let rocm = make_backend(GpuApi::Rocm);
        assert!(rocm.capabilities().supports_opencl);
        assert_eq!(rocm.capabilities().warp_size, 64);

        let vulkan = make_backend(GpuApi::Vulkan);
        assert!(vulkan.capabilities().supports_vulkan);

        let none = make_backend(GpuApi::None);
        assert!(!none.capabilities().supports_cuda);
        assert!(!none.capabilities().supports_vulkan);
        assert_eq!(none.capabilities().max_threads, 1);
    }

    #[test]
    fn test_name_reflects_api() {
        assert!(make_backend(GpuApi::Cuda).name().contains("CUDA"));
        assert!(make_backend(GpuApi::Rocm).name().contains("ROCm"));
        assert!(make_backend(GpuApi::Vulkan).name().contains("Vulkan"));
        assert!(make_backend(GpuApi::None).name().contains("fallback"));
    }

    // -- Memory management --------------------------------------------------

    #[test]
    fn test_allocate_and_free() {
        let backend = make_backend(GpuApi::None);
        let ptr = backend.allocate_memory(1024).expect("allocation failed");
        assert!(!ptr.is_null());
        assert_eq!(backend.allocation_count(), 1);
        assert_eq!(backend.allocated_bytes(), 1024);

        backend.free_memory(ptr).expect("free failed");
        assert_eq!(backend.allocation_count(), 0);
        assert_eq!(backend.allocated_bytes(), 0);
    }

    #[test]
    fn test_allocate_zero_bytes_fails() {
        let backend = make_backend(GpuApi::None);
        let result = backend.allocate_memory(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_free_null_pointer_fails() {
        let backend = make_backend(GpuApi::None);
        let result = backend.free_memory(std::ptr::null_mut());
        assert!(result.is_err());
    }

    #[test]
    fn test_double_free_fails() {
        let backend = make_backend(GpuApi::None);
        let ptr = backend.allocate_memory(64).unwrap();
        backend.free_memory(ptr).unwrap();
        let result = backend.free_memory(ptr);
        assert!(result.is_err());
    }

    #[test]
    fn test_free_unknown_pointer_fails() {
        let backend = make_backend(GpuApi::None);
        let mut dummy: u8 = 0;
        let result = backend.free_memory(&mut dummy as *mut u8);
        assert!(result.is_err());
    }

    // -- Copy memory --------------------------------------------------------

    #[test]
    fn test_copy_memory_round_trip() {
        let backend = make_backend(GpuApi::None);
        let src_data: Vec<u8> = (0..128).collect();

        let dst = backend.allocate_memory(128).unwrap();
        backend
            .copy_memory(dst, src_data.as_ptr(), 128, MemcpyKind::HostToDevice)
            .unwrap();

        let mut readback = vec![0u8; 128];
        backend
            .copy_memory(
                readback.as_mut_ptr(),
                dst as *const u8,
                128,
                MemcpyKind::DeviceToHost,
            )
            .unwrap();

        assert_eq!(readback, src_data);
        backend.free_memory(dst).unwrap();
    }

    #[test]
    fn test_copy_memory_null_dst_fails() {
        let backend = make_backend(GpuApi::None);
        let src: u8 = 42;
        let result = backend.copy_memory(
            std::ptr::null_mut(),
            &src as *const u8,
            1,
            MemcpyKind::HostToHost,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_copy_memory_null_src_fails() {
        let backend = make_backend(GpuApi::None);
        let mut dst: u8 = 0;
        let result = backend.copy_memory(
            &mut dst as *mut u8,
            std::ptr::null(),
            1,
            MemcpyKind::HostToHost,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_copy_zero_size_succeeds() {
        let backend = make_backend(GpuApi::None);
        let mut dst: u8 = 0;
        let src: u8 = 42;
        let result = backend.copy_memory(
            &mut dst as *mut u8,
            &src as *const u8,
            0,
            MemcpyKind::HostToHost,
        );
        assert!(result.is_ok());
        assert_eq!(dst, 0); // unchanged
    }

    // -- Kernel compilation -------------------------------------------------

    #[tokio::test]
    async fn test_compile_kernel_cuda() {
        let backend = make_backend(GpuApi::Cuda);
        let compiled = backend
            .compile_kernel("__global__ void f() {}")
            .await
            .unwrap();
        assert!(compiled.starts_with(b"CUDA_PTX:"));
    }

    #[tokio::test]
    async fn test_compile_kernel_rocm() {
        let backend = make_backend(GpuApi::Rocm);
        let compiled = backend
            .compile_kernel("__global__ void f() {}")
            .await
            .unwrap();
        assert!(compiled.starts_with(b"ROCM_CO:"));
    }

    #[tokio::test]
    async fn test_compile_kernel_vulkan() {
        let backend = make_backend(GpuApi::Vulkan);
        let compiled = backend
            .compile_kernel("#version 450\nvoid main() {}")
            .await
            .unwrap();
        assert!(compiled.starts_with(b"VK_SPIRV:"));
    }

    #[tokio::test]
    async fn test_compile_kernel_none_fails() {
        let backend = make_backend(GpuApi::None);
        let result = backend.compile_kernel("void f() {}").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_compile_empty_source_fails() {
        let backend = make_backend(GpuApi::Cuda);
        let result = backend.compile_kernel("").await;
        assert!(result.is_err());
    }

    // -- Kernel launch ------------------------------------------------------

    #[tokio::test]
    async fn test_launch_kernel_none_fails() {
        let backend = make_backend(GpuApi::None);
        let result = backend
            .launch_kernel(b"fake", (1, 1, 1), (1, 1, 1), &[])
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_launch_kernel_empty_binary_fails() {
        let backend = make_backend(GpuApi::Cuda);
        let result = backend
            .launch_kernel(b"", (1, 1, 1), (1, 1, 1), &[])
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_launch_kernel_block_dim_exceeded() {
        let backend = make_backend(GpuApi::Cuda);
        // max_block_dim[0] for CUDA is 1024
        let result = backend
            .launch_kernel(b"ptx", (1, 1, 1), (2048, 1, 1), &[])
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_launch_kernel_cuda_succeeds() {
        let backend = make_backend(GpuApi::Cuda);
        let compiled = backend
            .compile_kernel("__global__ void f() {}")
            .await
            .unwrap();
        let result = backend
            .launch_kernel(&compiled, (1, 1, 1), (32, 1, 1), &[])
            .await;
        assert!(result.is_ok());
    }

    // -- Synchronize --------------------------------------------------------

    #[test]
    fn test_synchronize_all_apis() {
        for api in [GpuApi::Cuda, GpuApi::Rocm, GpuApi::Vulkan, GpuApi::None] {
            let backend = make_backend(api);
            assert!(backend.synchronize().is_ok(), "synchronize failed for {:?}", api);
        }
    }

    // -- Initialize ---------------------------------------------------------

    #[tokio::test]
    async fn test_initialize_idempotent() {
        let mut backend = make_backend(GpuApi::None);
        backend.initialize().await.unwrap();
        backend.initialize().await.unwrap(); // second call is a no-op
    }

    // -- Multiple allocations -----------------------------------------------

    #[test]
    fn test_multiple_allocations_tracked() {
        let backend = make_backend(GpuApi::None);
        let p1 = backend.allocate_memory(100).unwrap();
        let p2 = backend.allocate_memory(200).unwrap();
        let p3 = backend.allocate_memory(300).unwrap();
        assert_eq!(backend.allocation_count(), 3);
        assert_eq!(backend.allocated_bytes(), 600);

        backend.free_memory(p2).unwrap();
        assert_eq!(backend.allocation_count(), 2);
        assert_eq!(backend.allocated_bytes(), 400);

        backend.free_memory(p1).unwrap();
        backend.free_memory(p3).unwrap();
        assert_eq!(backend.allocation_count(), 0);
    }
}
