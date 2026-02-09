//! Cooperative groups for cross-block synchronization
//!
//! Provides a software emulation of CUDA cooperative groups, enabling
//! flexible thread grouping and synchronization patterns beyond the
//! traditional block-level `__syncthreads()`.

use crate::{Result, runtime_error};
use std::sync::{Arc, Barrier, Mutex};

/// Thread group abstraction for cooperative kernel execution
#[derive(Debug, Clone)]
pub struct CooperativeGroup {
    /// Number of threads in this group
    size: u32,
    /// Rank of this thread within the group
    rank: u32,
    /// Synchronization barrier shared among group members
    barrier: Arc<Barrier>,
}

impl CooperativeGroup {
    /// Create a new cooperative group
    pub fn new(size: u32, rank: u32) -> Result<Self> {
        if rank >= size {
            return Err(runtime_error!(
                "Thread rank {} exceeds group size {}",
                rank, size
            ));
        }
        Ok(Self {
            size,
            rank,
            barrier: Arc::new(Barrier::new(size as usize)),
        })
    }

    /// Create a group with a shared barrier
    pub fn with_barrier(size: u32, rank: u32, barrier: Arc<Barrier>) -> Result<Self> {
        if rank >= size {
            return Err(runtime_error!(
                "Thread rank {} exceeds group size {}",
                rank, size
            ));
        }
        Ok(Self { size, rank, barrier })
    }

    /// Get the number of threads in this group
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Get this thread's rank within the group
    pub fn thread_rank(&self) -> u32 {
        self.rank
    }

    /// Synchronize all threads in this group
    pub fn sync(&self) {
        self.barrier.wait();
    }

    /// Check if this thread is the leader (rank 0)
    pub fn is_leader(&self) -> bool {
        self.rank == 0
    }
}

/// Thread block group (all threads in a block)
pub struct ThreadBlockGroup {
    inner: CooperativeGroup,
    block_idx: [u32; 3],
    block_dim: [u32; 3],
}

impl ThreadBlockGroup {
    /// Create a thread block group
    pub fn new(block_dim: [u32; 3], thread_idx: [u32; 3], barrier: Arc<Barrier>) -> Result<Self> {
        let size = block_dim[0] * block_dim[1] * block_dim[2];
        let rank = thread_idx[2] * block_dim[0] * block_dim[1]
            + thread_idx[1] * block_dim[0]
            + thread_idx[0];
        let inner = CooperativeGroup::with_barrier(size, rank, barrier)?;
        Ok(Self {
            inner,
            block_idx: [0, 0, 0],
            block_dim,
        })
    }

    /// Set the block index
    pub fn with_block_idx(mut self, idx: [u32; 3]) -> Self {
        self.block_idx = idx;
        self
    }

    /// Synchronize the thread block
    pub fn sync(&self) {
        self.inner.sync();
    }

    /// Get block dimensions
    pub fn dim_threads(&self) -> [u32; 3] {
        self.block_dim
    }

    /// Get group size (total threads in block)
    pub fn size(&self) -> u32 {
        self.inner.size()
    }

    /// Get thread rank within block
    pub fn thread_rank(&self) -> u32 {
        self.inner.thread_rank()
    }

    /// Get block index
    pub fn block_index(&self) -> [u32; 3] {
        self.block_idx
    }
}

/// Grid group (all threads across all blocks)
pub struct GridGroup {
    /// Total number of threads across all blocks
    total_threads: u32,
    /// Global rank of this thread
    global_rank: u32,
    /// Grid dimensions
    grid_dim: [u32; 3],
    /// Block dimensions
    block_dim: [u32; 3],
    /// Optional grid-level barrier for cooperative launch
    barrier: Option<Arc<Barrier>>,
}

impl GridGroup {
    /// Create a grid group
    pub fn new(
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        block_idx: [u32; 3],
        thread_idx: [u32; 3],
    ) -> Self {
        let threads_per_block = block_dim[0] * block_dim[1] * block_dim[2];
        let total_blocks = grid_dim[0] * grid_dim[1] * grid_dim[2];
        let total_threads = total_blocks * threads_per_block;

        let block_linear = block_idx[2] * grid_dim[0] * grid_dim[1]
            + block_idx[1] * grid_dim[0]
            + block_idx[0];
        let thread_linear = thread_idx[2] * block_dim[0] * block_dim[1]
            + thread_idx[1] * block_dim[0]
            + thread_idx[0];
        let global_rank = block_linear * threads_per_block + thread_linear;

        Self {
            total_threads,
            global_rank,
            grid_dim,
            block_dim,
            barrier: None,
        }
    }

    /// Create with a grid-level barrier for cooperative launch synchronization
    pub fn with_barrier(mut self, barrier: Arc<Barrier>) -> Self {
        self.barrier = Some(barrier);
        self
    }

    /// Get total number of threads in the grid
    pub fn size(&self) -> u32 {
        self.total_threads
    }

    /// Get this thread's global rank
    pub fn thread_rank(&self) -> u32 {
        self.global_rank
    }

    /// Get grid dimensions
    pub fn dim_blocks(&self) -> [u32; 3] {
        self.grid_dim
    }

    /// Get block dimensions
    pub fn dim_threads(&self) -> [u32; 3] {
        self.block_dim
    }

    /// Check if this thread is the leader
    pub fn is_leader(&self) -> bool {
        self.global_rank == 0
    }

    /// Synchronize all threads in the grid (cooperative launch only)
    pub fn sync(&self) -> Result<()> {
        match &self.barrier {
            Some(b) => {
                b.wait();
                Ok(())
            }
            None => Err(runtime_error!(
                "Grid sync requires cooperative launch with a shared barrier"
            )),
        }
    }
}

/// Tiled partition: a subdivision of a thread group
pub struct TiledPartition {
    /// Tile size (must be power of 2, max 32 for warp-level)
    tile_size: u32,
    /// Rank within the tile
    rank: u32,
    /// Barrier for the tile
    barrier: Arc<Barrier>,
    /// Shared data buffer for shuffle operations
    shared_data: Arc<Mutex<Vec<f32>>>,
}

impl TiledPartition {
    /// Create a tiled partition
    pub fn new(tile_size: u32, rank: u32) -> Result<Self> {
        if !tile_size.is_power_of_two() || tile_size > 32 {
            return Err(runtime_error!(
                "Tile size must be a power of 2 and <= 32, got {}",
                tile_size
            ));
        }
        if rank >= tile_size {
            return Err(runtime_error!(
                "Rank {} exceeds tile size {}",
                rank, tile_size
            ));
        }
        Ok(Self {
            tile_size,
            rank,
            barrier: Arc::new(Barrier::new(tile_size as usize)),
            shared_data: Arc::new(Mutex::new(vec![0.0; tile_size as usize])),
        })
    }

    /// Create with shared state
    pub fn with_shared(
        tile_size: u32,
        rank: u32,
        barrier: Arc<Barrier>,
        shared_data: Arc<Mutex<Vec<f32>>>,
    ) -> Result<Self> {
        if rank >= tile_size {
            return Err(runtime_error!(
                "Rank {} exceeds tile size {}",
                rank, tile_size
            ));
        }
        Ok(Self {
            tile_size,
            rank,
            barrier,
            shared_data,
        })
    }

    /// Get tile size
    pub fn size(&self) -> u32 {
        self.tile_size
    }

    /// Get thread rank within tile
    pub fn thread_rank(&self) -> u32 {
        self.rank
    }

    /// Synchronize threads within the tile
    pub fn sync(&self) {
        self.barrier.wait();
    }

    /// Shuffle: get value from thread with given rank
    pub fn shfl(&self, value: f32, src_rank: u32) -> f32 {
        {
            let mut data = self.shared_data.lock().unwrap();
            data[self.rank as usize] = value;
        }
        self.sync();
        let result = {
            let data = self.shared_data.lock().unwrap();
            let idx = (src_rank % self.tile_size) as usize;
            data[idx]
        };
        self.sync();
        result
    }

    /// Shuffle down: get value from thread rank + delta
    pub fn shfl_down(&self, value: f32, delta: u32) -> f32 {
        let src = self.rank + delta;
        if src >= self.tile_size {
            value // Return own value if source is out of range
        } else {
            self.shfl(value, src)
        }
    }

    /// Shuffle up: get value from thread rank - delta
    pub fn shfl_up(&self, value: f32, delta: u32) -> f32 {
        if self.rank < delta {
            value
        } else {
            self.shfl(value, self.rank - delta)
        }
    }

    /// Shuffle XOR: get value from thread rank ^ mask
    pub fn shfl_xor(&self, value: f32, mask: u32) -> f32 {
        self.shfl(value, self.rank ^ mask)
    }
}

/// Create a cooperative group for the current thread block
pub fn this_thread_block(
    block_dim: [u32; 3],
    thread_idx: [u32; 3],
    barrier: Arc<Barrier>,
) -> Result<ThreadBlockGroup> {
    ThreadBlockGroup::new(block_dim, thread_idx, barrier)
}

/// Create a grid group for cooperative kernel launch
pub fn this_grid(
    grid_dim: [u32; 3],
    block_dim: [u32; 3],
    block_idx: [u32; 3],
    thread_idx: [u32; 3],
) -> GridGroup {
    GridGroup::new(grid_dim, block_dim, block_idx, thread_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooperative_group_creation() {
        let group = CooperativeGroup::new(32, 0).unwrap();
        assert_eq!(group.size(), 32);
        assert_eq!(group.thread_rank(), 0);
        assert!(group.is_leader());
    }

    #[test]
    fn test_cooperative_group_invalid_rank() {
        let result = CooperativeGroup::new(32, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_thread_block_group() {
        let barrier = Arc::new(Barrier::new(1));
        let group = ThreadBlockGroup::new([4, 4, 1], [2, 1, 0], barrier).unwrap();
        assert_eq!(group.size(), 16);
        assert_eq!(group.thread_rank(), 1 * 4 + 2); // y * dim_x + x = 6
        assert_eq!(group.dim_threads(), [4, 4, 1]);
    }

    #[test]
    fn test_grid_group() {
        let gg = GridGroup::new([2, 2, 1], [4, 4, 1], [1, 0, 0], [2, 1, 0]);
        assert_eq!(gg.size(), 4 * 16); // 4 blocks * 16 threads
        assert_eq!(gg.dim_blocks(), [2, 2, 1]);
        assert_eq!(gg.dim_threads(), [4, 4, 1]);
        // Block 1 (linear), thread 6 (linear) = 1*16 + 6 = 22
        assert_eq!(gg.thread_rank(), 22);
    }

    #[test]
    fn test_grid_group_leader() {
        let gg = GridGroup::new([1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]);
        assert!(gg.is_leader());

        let gg2 = GridGroup::new([2, 1, 1], [4, 1, 1], [1, 0, 0], [2, 0, 0]);
        assert!(!gg2.is_leader());
    }

    #[test]
    fn test_grid_sync_without_barrier() {
        let gg = GridGroup::new([1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]);
        assert!(gg.sync().is_err());
    }

    #[test]
    fn test_grid_sync_with_barrier() {
        let barrier = Arc::new(Barrier::new(1));
        let gg = GridGroup::new([1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0])
            .with_barrier(barrier);
        assert!(gg.sync().is_ok());
    }

    #[test]
    fn test_tiled_partition_creation() {
        let tile = TiledPartition::new(4, 0).unwrap();
        assert_eq!(tile.size(), 4);
        assert_eq!(tile.thread_rank(), 0);
    }

    #[test]
    fn test_tiled_partition_invalid_size() {
        // Not a power of two
        assert!(TiledPartition::new(3, 0).is_err());
        // Too large
        assert!(TiledPartition::new(64, 0).is_err());
    }

    #[test]
    fn test_cooperative_group_sync() {
        // Single-thread sync should not deadlock
        let group = CooperativeGroup::new(1, 0).unwrap();
        group.sync();
    }

    #[test]
    fn test_multi_thread_cooperative_sync() {
        let barrier = Arc::new(Barrier::new(4));
        let handles: Vec<_> = (0..4)
            .map(|rank| {
                let b = barrier.clone();
                std::thread::spawn(move || {
                    let group = CooperativeGroup::with_barrier(4, rank, b).unwrap();
                    group.sync();
                    group.thread_rank()
                })
            })
            .collect();

        let mut ranks: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        ranks.sort();
        assert_eq!(ranks, vec![0, 1, 2, 3]);
    }
}
