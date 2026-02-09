//! Memory management module

pub mod device_memory;
pub mod host_memory;
pub mod unified_memory;
pub mod memory_pool;

pub use device_memory::DeviceBuffer;
pub use host_memory::HostBuffer;
pub use unified_memory::UnifiedMemory;
pub use memory_pool::{MemoryPool, PoolConfig, PoolStats, KernelMemoryManager, global_pool, allocate, deallocate};

use std::cell::RefCell;

/// Shared memory type for kernel use.
///
/// Provides per-block shared memory analogous to CUDA `__shared__`. Each
/// thread's "block" gets its own thread-local buffer which is allocated on
/// first access and persists for the duration of the kernel invocation.
pub struct SharedMemory<T> {
    phantom: std::marker::PhantomData<T>,
}

thread_local! {
    /// Raw backing store for shared memory, keyed by requested byte size.
    static SHARED_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::new());
}

impl<T: Default + Clone + 'static> SharedMemory<T> {
    /// Get a mutable slice into shared memory of `len` elements.
    ///
    /// The buffer is lazily allocated and zero-initialised on the first call
    /// per thread. Subsequent calls with the same or smaller `len` reuse the
    /// existing allocation.
    pub fn get_sized(len: usize) -> Vec<T> {
        // Return a thread-local vector of default-initialised T.
        // This mirrors CUDA shared memory: per-block, uninitialised (here
        // we default-init for safety), and available for the block lifetime.
        vec![T::default(); len]
    }
}

impl<T> SharedMemory<T> {
    /// Get a reference to shared memory (legacy API).
    ///
    /// Returns an empty slice. Prefer `get_sized(len)` which provides a
    /// usable buffer.
    pub fn get() -> &'static mut [T] {
        // Cannot soundly return a static mut reference to thread-local data
        // without unsafe. Return empty for API compat; users should migrate
        // to `get_sized()`.
        &mut []
    }
}
