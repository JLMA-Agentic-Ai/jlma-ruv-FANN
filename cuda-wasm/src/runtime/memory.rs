//! Memory management for runtime kernel execution
//!
//! Provides allocate/copy/free for host-side memory used by the runtime
//! kernel executor. For GPU device memory, use the backend trait directly.

use crate::Result;

/// Allocate memory for kernel data.
/// Uses the system allocator since runtime kernels execute on the CPU.
pub fn allocate(size: usize) -> Result<*mut u8> {
    if size == 0 {
        return Ok(std::ptr::null_mut());
    }
    let layout = std::alloc::Layout::from_size_align(size, std::mem::align_of::<f64>())
        .map_err(|e| crate::runtime_error!("Invalid allocation layout: {}", e))?;
    // SAFETY: layout is valid and non-zero sized
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err(crate::runtime_error!("Memory allocation failed for {} bytes", size));
    }
    Ok(ptr)
}

/// Copy memory between buffers.
pub fn copy(dst: *mut u8, src: *const u8, size: usize) -> Result<()> {
    if size == 0 {
        return Ok(());
    }
    if dst.is_null() || src.is_null() {
        return Err(crate::runtime_error!("Null pointer in memory copy"));
    }
    // SAFETY: caller guarantees valid, non-overlapping regions of `size` bytes
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, size);
    }
    Ok(())
}

/// Free previously allocated memory.
pub fn free(ptr: *mut u8) -> Result<()> {
    if ptr.is_null() {
        return Ok(()); // freeing null is a no-op, matching C/CUDA behavior
    }
    // NOTE: We cannot free without knowing the layout. Track allocations
    // for a proper implementation. For now, leak prevention relies on
    // higher-level RAII wrappers (DeviceBuffer, HostBuffer, MemoryPool).
    // This is intentionally a no-op to avoid UB from mismatched layouts.
    Ok(())
}
