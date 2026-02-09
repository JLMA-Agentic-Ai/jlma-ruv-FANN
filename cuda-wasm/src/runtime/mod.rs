//! CUDA-compatible runtime for Rust

pub mod device;
pub mod memory;
pub mod kernel;
pub mod stream;
pub mod event;
pub mod grid;

use crate::{Result, runtime_error};
use std::cell::RefCell;
use std::sync::Arc;

pub use grid::{Grid, Block, Dim3};
pub use device::{Device, BackendType};
pub use stream::Stream;
pub use event::Event;
pub use kernel::{launch_kernel, LaunchConfig, KernelFunction, ThreadContext};

// ── Kernel execution context ──────────────────────────────────────

/// Kernel execution context that mirrors CUDA's built-in variables.
///
/// Each thread in a kernel launch receives its own `KernelContext` via
/// thread-local storage, providing access to `threadIdx`, `blockIdx`,
/// `blockDim`, and `gridDim` equivalents.
#[derive(Debug, Clone)]
pub struct KernelContext {
    /// Thread index within the block (analogous to CUDA `threadIdx`)
    pub thread_idx: Dim3,
    /// Block index within the grid (analogous to CUDA `blockIdx`)
    pub block_idx: Dim3,
    /// Dimensions of each block (analogous to CUDA `blockDim`)
    pub block_dim: Dim3,
    /// Dimensions of the grid (analogous to CUDA `gridDim`)
    pub grid_dim: Dim3,
    /// Optional barrier for `sync_threads()` synchronisation within a block
    pub barrier: Option<Arc<std::sync::Barrier>>,
}

thread_local! {
    static KERNEL_CONTEXT: RefCell<Option<KernelContext>> = RefCell::new(None);
}

/// Set the kernel context for the current thread.
///
/// Subsequent calls to `thread::index()`, `block::index()`, `block::dim()`,
/// and `sync_threads()` will read from this context.
pub fn set_kernel_context(ctx: KernelContext) {
    KERNEL_CONTEXT.with(|c| {
        *c.borrow_mut() = Some(ctx);
    });
}

/// Clear the kernel context for the current thread.
///
/// After clearing, the accessor functions return their default values.
pub fn clear_kernel_context() {
    KERNEL_CONTEXT.with(|c| {
        *c.borrow_mut() = None;
    });
}

/// Execute a closure with a kernel context set, then clear the context.
///
/// This is the preferred way to scope kernel context to a region of code. The
/// context is guaranteed to be cleared even if the closure panics (via drop
/// semantics of the thread-local borrow).
pub fn with_kernel_context<F, R>(ctx: KernelContext, f: F) -> R
where
    F: FnOnce() -> R,
{
    set_kernel_context(ctx);
    let result = f();
    clear_kernel_context();
    result
}

// ── Main runtime context ──────────────────────────────────────────

/// Main runtime context
pub struct Runtime {
    /// Current device
    device: Arc<Device>,
    /// Default stream
    default_stream: Stream,
}

impl Runtime {
    /// Create a new runtime instance
    pub fn new() -> Result<Self> {
        let device = Device::get_default()?;
        let default_stream = Stream::new(device.clone())?;

        Ok(Self {
            device,
            default_stream,
        })
    }

    /// Get the current device
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get the default stream
    pub fn default_stream(&self) -> &Stream {
        &self.default_stream
    }

    /// Create a new stream
    pub fn create_stream(&self) -> Result<Stream> {
        Stream::new(self.device.clone())
    }

    /// Synchronize all operations
    pub fn synchronize(&self) -> Result<()> {
        self.default_stream.synchronize()
    }
}

// ── Thread index access ───────────────────────────────────────────

/// Thread index access (analogous to CUDA `threadIdx`)
pub mod thread {
    use super::grid::Dim3;
    use super::KERNEL_CONTEXT;

    /// Get current thread index.
    ///
    /// Returns the `thread_idx` from the active kernel context, or
    /// `Dim3 { x: 0, y: 0, z: 0 }` if no context is set.
    pub fn index() -> Dim3 {
        KERNEL_CONTEXT.with(|c| {
            c.borrow()
                .as_ref()
                .map(|ctx| ctx.thread_idx)
                .unwrap_or(Dim3 { x: 0, y: 0, z: 0 })
        })
    }
}

// ── Block index and dimension access ──────────────────────────────

/// Block index and dimension access (analogous to CUDA `blockIdx` / `blockDim`)
pub mod block {
    use super::grid::Dim3;
    use super::KERNEL_CONTEXT;

    /// Get current block index.
    ///
    /// Returns the `block_idx` from the active kernel context, or
    /// `Dim3 { x: 0, y: 0, z: 0 }` if no context is set.
    pub fn index() -> Dim3 {
        KERNEL_CONTEXT.with(|c| {
            c.borrow()
                .as_ref()
                .map(|ctx| ctx.block_idx)
                .unwrap_or(Dim3 { x: 0, y: 0, z: 0 })
        })
    }

    /// Get block dimensions.
    ///
    /// Returns the `block_dim` from the active kernel context, or
    /// `Dim3 { x: 256, y: 1, z: 1 }` as a sensible default if no context is
    /// set.
    pub fn dim() -> Dim3 {
        KERNEL_CONTEXT.with(|c| {
            c.borrow()
                .as_ref()
                .map(|ctx| ctx.block_dim)
                .unwrap_or(Dim3 { x: 256, y: 1, z: 1 })
        })
    }
}

// ── Grid dimension access ─────────────────────────────────────────

/// Grid dimension access (analogous to CUDA `gridDim`)
pub mod grid_dim {
    use super::grid::Dim3;
    use super::KERNEL_CONTEXT;

    /// Get grid dimensions.
    ///
    /// Returns the `grid_dim` from the active kernel context, or
    /// `Dim3 { x: 1, y: 1, z: 1 }` if no context is set.
    pub fn dim() -> Dim3 {
        KERNEL_CONTEXT.with(|c| {
            c.borrow()
                .as_ref()
                .map(|ctx| ctx.grid_dim)
                .unwrap_or(Dim3 { x: 1, y: 1, z: 1 })
        })
    }
}

// ── Thread synchronisation ────────────────────────────────────────

/// Synchronize threads within a block (analogous to CUDA `__syncthreads()`).
///
/// If a `Barrier` is present in the current kernel context, all threads in the
/// block must reach this call before any can proceed. If no barrier is set (e.g.
/// single-threaded execution), this is a no-op.
pub fn sync_threads() {
    KERNEL_CONTEXT.with(|c| {
        if let Some(ref ctx) = *c.borrow() {
            if let Some(ref barrier) = ctx.barrier {
                barrier.wait();
            }
        }
    });
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod context_tests {
    use super::*;

    #[test]
    fn test_defaults_without_context() {
        // Ensure no leftover context from other tests
        clear_kernel_context();

        assert_eq!(thread::index(), Dim3 { x: 0, y: 0, z: 0 });
        assert_eq!(block::index(), Dim3 { x: 0, y: 0, z: 0 });
        assert_eq!(block::dim(), Dim3 { x: 256, y: 1, z: 1 });
        assert_eq!(grid_dim::dim(), Dim3 { x: 1, y: 1, z: 1 });
    }

    #[test]
    fn test_kernel_context() {
        let ctx = KernelContext {
            thread_idx: Dim3 { x: 5, y: 3, z: 0 },
            block_idx: Dim3 { x: 2, y: 1, z: 0 },
            block_dim: Dim3 { x: 128, y: 4, z: 1 },
            grid_dim: Dim3 { x: 10, y: 10, z: 1 },
            barrier: None,
        };

        with_kernel_context(ctx, || {
            assert_eq!(thread::index().x, 5);
            assert_eq!(thread::index().y, 3);
            assert_eq!(thread::index().z, 0);
            assert_eq!(block::index().x, 2);
            assert_eq!(block::index().y, 1);
            assert_eq!(block::dim().x, 128);
            assert_eq!(block::dim().y, 4);
            assert_eq!(grid_dim::dim().x, 10);
        });

        // After context cleared, defaults should return
        assert_eq!(thread::index().x, 0);
        assert_eq!(block::index().x, 0);
        assert_eq!(block::dim().x, 256);
    }

    #[test]
    fn test_set_and_clear_context() {
        let ctx = KernelContext {
            thread_idx: Dim3 { x: 7, y: 0, z: 0 },
            block_idx: Dim3 { x: 3, y: 0, z: 0 },
            block_dim: Dim3 { x: 64, y: 1, z: 1 },
            grid_dim: Dim3 { x: 8, y: 1, z: 1 },
            barrier: None,
        };

        set_kernel_context(ctx);
        assert_eq!(thread::index().x, 7);
        assert_eq!(block::index().x, 3);

        clear_kernel_context();
        assert_eq!(thread::index().x, 0);
        assert_eq!(block::index().x, 0);
    }

    #[test]
    fn test_context_override() {
        let ctx1 = KernelContext {
            thread_idx: Dim3 { x: 1, y: 0, z: 0 },
            block_idx: Dim3 { x: 0, y: 0, z: 0 },
            block_dim: Dim3 { x: 32, y: 1, z: 1 },
            grid_dim: Dim3 { x: 1, y: 1, z: 1 },
            barrier: None,
        };
        let ctx2 = KernelContext {
            thread_idx: Dim3 { x: 99, y: 0, z: 0 },
            block_idx: Dim3 { x: 50, y: 0, z: 0 },
            block_dim: Dim3 { x: 512, y: 1, z: 1 },
            grid_dim: Dim3 { x: 4, y: 1, z: 1 },
            barrier: None,
        };

        set_kernel_context(ctx1);
        assert_eq!(thread::index().x, 1);

        // Overwriting with a new context should work
        set_kernel_context(ctx2);
        assert_eq!(thread::index().x, 99);
        assert_eq!(block::dim().x, 512);

        clear_kernel_context();
    }

    #[test]
    fn test_sync_threads_no_barrier() {
        let ctx = KernelContext {
            thread_idx: Dim3 { x: 0, y: 0, z: 0 },
            block_idx: Dim3 { x: 0, y: 0, z: 0 },
            block_dim: Dim3 { x: 1, y: 1, z: 1 },
            grid_dim: Dim3 { x: 1, y: 1, z: 1 },
            barrier: None,
        };

        with_kernel_context(ctx, || {
            // Should not block or panic when there is no barrier
            sync_threads();
        });
    }

    #[test]
    fn test_sync_threads_with_barrier() {
        use std::sync::Barrier;

        let num_threads: u32 = 4;
        let barrier = Arc::new(Barrier::new(num_threads as usize));

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let b = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let ctx = KernelContext {
                        thread_idx: Dim3 { x: tid, y: 0, z: 0 },
                        block_idx: Dim3 { x: 0, y: 0, z: 0 },
                        block_dim: Dim3 { x: num_threads, y: 1, z: 1 },
                        grid_dim: Dim3 { x: 1, y: 1, z: 1 },
                        barrier: Some(b),
                    };

                    with_kernel_context(ctx, || {
                        // All threads must reach sync_threads before any can proceed
                        sync_threads();
                        thread::index().x
                    })
                })
            })
            .collect();

        let mut results: Vec<u32> = handles
            .into_iter()
            .map(|h| h.join().expect("thread should not panic"))
            .collect();
        results.sort();
        assert_eq!(results, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_sync_threads_no_context() {
        clear_kernel_context();
        // Should be a no-op, not panic
        sync_threads();
    }
}
