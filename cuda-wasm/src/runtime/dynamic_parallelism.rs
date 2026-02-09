//! Dynamic parallelism support for child kernel launches
//!
//! Allows kernels to launch child kernels, emulating CUDA's dynamic
//! parallelism feature. In the CPU emulation backend, child kernels
//! are executed synchronously or queued for deferred execution.

use crate::{Result, runtime_error};
use crate::runtime::grid::{Grid, Block, Dim3};
use crate::runtime::kernel::ThreadContext;
use std::sync::{Arc, Mutex};

/// A child kernel that can be launched from within a parent kernel
pub trait ChildKernel: Send + Sync {
    /// Execute the child kernel for a single thread
    fn execute(&self, ctx: ThreadContext);

    /// Get kernel name
    fn name(&self) -> &str;
}

/// Child kernel launch record
#[derive(Debug, Clone)]
pub struct ChildLaunch {
    /// Kernel name
    pub kernel_name: String,
    /// Grid dimensions
    pub grid: Dim3,
    /// Block dimensions
    pub block: Dim3,
    /// Shared memory size
    pub shared_mem_bytes: usize,
    /// Whether execution is complete
    pub completed: bool,
}

/// Dynamic parallelism context for managing child kernel launches
pub struct DynamicParallelismContext {
    /// Maximum nesting depth for child launches
    max_depth: u32,
    /// Current nesting depth
    current_depth: u32,
    /// Record of child launches
    launch_history: Arc<Mutex<Vec<ChildLaunch>>>,
    /// Maximum number of concurrent child kernels
    max_pending: usize,
}

impl DynamicParallelismContext {
    /// Create a new dynamic parallelism context
    pub fn new() -> Self {
        Self {
            max_depth: 24, // CUDA default max depth
            current_depth: 0,
            launch_history: Arc::new(Mutex::new(Vec::new())),
            max_pending: 2048,
        }
    }

    /// Create with custom nesting depth limit
    pub fn with_max_depth(mut self, depth: u32) -> Self {
        self.max_depth = depth;
        self
    }

    /// Create with custom max pending limit
    pub fn with_max_pending(mut self, max: usize) -> Self {
        self.max_pending = max;
        self
    }

    /// Launch a child kernel (synchronous execution in CPU backend)
    pub fn launch_child<K: ChildKernel>(
        &mut self,
        kernel: &K,
        grid: Grid,
        block: Block,
        shared_mem_bytes: usize,
    ) -> Result<()> {
        // Check nesting depth
        if self.current_depth >= self.max_depth {
            return Err(runtime_error!(
                "Maximum kernel nesting depth {} exceeded",
                self.max_depth
            ));
        }

        // Check pending limit
        {
            let history = self.launch_history.lock().unwrap();
            let pending = history.iter().filter(|l| !l.completed).count();
            if pending >= self.max_pending {
                return Err(runtime_error!(
                    "Maximum pending child kernels {} exceeded",
                    self.max_pending
                ));
            }
        }

        // Validate block config
        block.validate()?;

        // Record the launch
        let launch_record = ChildLaunch {
            kernel_name: kernel.name().to_string(),
            grid: grid.dim,
            block: block.dim,
            shared_mem_bytes,
            completed: false,
        };

        {
            let mut history = self.launch_history.lock().unwrap();
            history.push(launch_record);
        }

        // Execute child kernel (CPU emulation: synchronous)
        self.current_depth += 1;

        let total_blocks = grid.num_blocks();
        let threads_per_block = block.num_threads();

        for block_id in 0..total_blocks {
            let block_idx = Dim3 {
                x: block_id % grid.dim.x,
                y: (block_id / grid.dim.x) % grid.dim.y,
                z: block_id / (grid.dim.x * grid.dim.y),
            };

            for thread_id in 0..threads_per_block {
                let thread_idx = Dim3 {
                    x: thread_id % block.dim.x,
                    y: (thread_id / block.dim.x) % block.dim.y,
                    z: thread_id / (block.dim.x * block.dim.y),
                };

                let ctx = ThreadContext {
                    thread_idx,
                    block_idx,
                    block_dim: block.dim,
                    grid_dim: grid.dim,
                };

                kernel.execute(ctx);
            }
        }

        self.current_depth -= 1;

        // Mark as completed
        {
            let mut history = self.launch_history.lock().unwrap();
            if let Some(last) = history.last_mut() {
                last.completed = true;
            }
        }

        Ok(())
    }

    /// Synchronize all pending child kernels (no-op in synchronous mode)
    pub fn device_synchronize(&self) -> Result<()> {
        // In CPU emulation, all launches are synchronous, so this is a no-op
        Ok(())
    }

    /// Get the number of completed child launches
    pub fn completed_launches(&self) -> usize {
        self.launch_history
            .lock()
            .unwrap()
            .iter()
            .filter(|l| l.completed)
            .count()
    }

    /// Get launch history
    pub fn launch_history(&self) -> Vec<ChildLaunch> {
        self.launch_history.lock().unwrap().clone()
    }

    /// Get current nesting depth
    pub fn current_depth(&self) -> u32 {
        self.current_depth
    }

    /// Get maximum nesting depth
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    /// Reset the context
    pub fn reset(&mut self) {
        self.current_depth = 0;
        self.launch_history.lock().unwrap().clear();
    }
}

impl Default for DynamicParallelismContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AddOneKernel {
        data: Arc<Mutex<Vec<f32>>>,
    }

    impl ChildKernel for AddOneKernel {
        fn execute(&self, ctx: ThreadContext) {
            let tid = ctx.global_thread_id();
            let mut data = self.data.lock().unwrap();
            if tid < data.len() {
                data[tid] += 1.0;
            }
        }

        fn name(&self) -> &str {
            "add_one"
        }
    }

    #[test]
    fn test_dynamic_parallelism_basic() {
        let mut dp = DynamicParallelismContext::new();
        let data = Arc::new(Mutex::new(vec![0.0f32; 16]));
        let kernel = AddOneKernel { data: data.clone() };

        dp.launch_child(&kernel, Grid::new(1u32), Block::new(16u32), 0)
            .unwrap();

        let result = data.lock().unwrap();
        assert!(result.iter().all(|&v| v == 1.0));
        assert_eq!(dp.completed_launches(), 1);
    }

    #[test]
    fn test_dynamic_parallelism_multiple_launches() {
        let mut dp = DynamicParallelismContext::new();
        let data = Arc::new(Mutex::new(vec![0.0f32; 8]));
        let kernel = AddOneKernel { data: data.clone() };

        for _ in 0..3 {
            dp.launch_child(&kernel, Grid::new(1u32), Block::new(8u32), 0)
                .unwrap();
        }

        let result = data.lock().unwrap();
        assert!(result.iter().all(|&v| v == 3.0));
        assert_eq!(dp.completed_launches(), 3);
    }

    #[test]
    fn test_dynamic_parallelism_max_depth() {
        let mut dp = DynamicParallelismContext::new().with_max_depth(0);
        let data = Arc::new(Mutex::new(vec![0.0f32; 4]));
        let kernel = AddOneKernel { data };

        let result = dp.launch_child(&kernel, Grid::new(1u32), Block::new(4u32), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dynamic_parallelism_device_sync() {
        let dp = DynamicParallelismContext::new();
        assert!(dp.device_synchronize().is_ok());
    }

    #[test]
    fn test_dynamic_parallelism_reset() {
        let mut dp = DynamicParallelismContext::new();
        let data = Arc::new(Mutex::new(vec![0.0f32; 4]));
        let kernel = AddOneKernel { data };

        dp.launch_child(&kernel, Grid::new(1u32), Block::new(4u32), 0)
            .unwrap();
        assert_eq!(dp.completed_launches(), 1);

        dp.reset();
        assert_eq!(dp.completed_launches(), 0);
        assert_eq!(dp.current_depth(), 0);
    }

    struct AddOne2DKernel {
        data: Arc<Mutex<Vec<f32>>>,
        width: usize,
    }

    impl ChildKernel for AddOne2DKernel {
        fn execute(&self, ctx: ThreadContext) {
            let (x, y) = ctx.global_thread_id_2d();
            let idx = y * self.width + x;
            let mut data = self.data.lock().unwrap();
            if idx < data.len() {
                data[idx] += 1.0;
            }
        }

        fn name(&self) -> &str {
            "add_one_2d"
        }
    }

    #[test]
    fn test_dynamic_parallelism_2d_grid() {
        let mut dp = DynamicParallelismContext::new();
        // 2x2 grid, 4x4 block = 8x8 = 64 threads
        let width = 2 * 4; // grid.x * block.x = 8
        let height = 2 * 4; // grid.y * block.y = 8
        let data = Arc::new(Mutex::new(vec![0.0f32; width * height]));
        let kernel = AddOne2DKernel { data: data.clone(), width };

        dp.launch_child(
            &kernel,
            Grid::new((2u32, 2u32)),
            Block::new((4u32, 4u32)),
            0,
        )
        .unwrap();

        let result = data.lock().unwrap();
        assert!(result.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_launch_history() {
        let mut dp = DynamicParallelismContext::new();
        let data = Arc::new(Mutex::new(vec![0.0f32; 4]));
        let kernel = AddOneKernel { data };

        dp.launch_child(&kernel, Grid::new(1u32), Block::new(4u32), 0)
            .unwrap();

        let history = dp.launch_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].kernel_name, "add_one");
        assert!(history[0].completed);
    }
}
