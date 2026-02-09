//! Occupancy Calculator
//!
//! Predicts GPU occupancy (active warps / max warps) for a given kernel
//! configuration, mirroring CUDA's `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
//!
//! Occupancy is limited by three resources:
//! 1. Registers per thread
//! 2. Shared memory per block
//! 3. Threads per block (warp granularity)

use std::fmt;

/// GPU architecture specification.
#[derive(Debug, Clone)]
pub struct GpuArchSpec {
    /// Compute capability name (e.g., "sm_90", "gfx1100").
    pub name: String,
    /// Max threads per SM/CU.
    pub max_threads_per_sm: u32,
    /// Max blocks (thread groups) per SM.
    pub max_blocks_per_sm: u32,
    /// Max warps per SM.
    pub max_warps_per_sm: u32,
    /// Warp size (32 for NVIDIA, 64 for AMD RDNA, 32 for AMD CDNA).
    pub warp_size: u32,
    /// Total registers per SM.
    pub registers_per_sm: u32,
    /// Register allocation granularity (registers are allocated in chunks).
    pub register_alloc_granularity: u32,
    /// Shared memory per SM (bytes).
    pub shared_memory_per_sm: u32,
    /// Shared memory allocation granularity (bytes).
    pub shared_memory_alloc_granularity: u32,
    /// Number of SMs/CUs on the device.
    pub sm_count: u32,
}

impl GpuArchSpec {
    /// NVIDIA Hopper (SM 9.0) — H100.
    pub fn hopper() -> Self {
        Self {
            name: "sm_90".into(),
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
            max_warps_per_sm: 64,
            warp_size: 32,
            registers_per_sm: 65536,
            register_alloc_granularity: 256,
            shared_memory_per_sm: 228 * 1024,
            shared_memory_alloc_granularity: 256,
            sm_count: 132,
        }
    }

    /// NVIDIA Ada Lovelace (SM 8.9) — RTX 4090.
    pub fn ada_lovelace() -> Self {
        Self {
            name: "sm_89".into(),
            max_threads_per_sm: 1536,
            max_blocks_per_sm: 24,
            max_warps_per_sm: 48,
            warp_size: 32,
            registers_per_sm: 65536,
            register_alloc_granularity: 256,
            shared_memory_per_sm: 100 * 1024,
            shared_memory_alloc_granularity: 256,
            sm_count: 128,
        }
    }

    /// NVIDIA Ampere (SM 8.0) — A100.
    pub fn ampere() -> Self {
        Self {
            name: "sm_80".into(),
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
            max_warps_per_sm: 64,
            warp_size: 32,
            registers_per_sm: 65536,
            register_alloc_granularity: 256,
            shared_memory_per_sm: 164 * 1024,
            shared_memory_alloc_granularity: 128,
            sm_count: 108,
        }
    }

    /// AMD CDNA3 — MI300X.
    pub fn cdna3() -> Self {
        Self {
            name: "gfx942".into(),
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
            max_warps_per_sm: 32,
            warp_size: 64, // AMD wavefront
            registers_per_sm: 65536,
            register_alloc_granularity: 256,
            shared_memory_per_sm: 64 * 1024,
            shared_memory_alloc_granularity: 256,
            sm_count: 304,
        }
    }

    /// Generic spec for WebGPU/software simulation.
    pub fn generic() -> Self {
        Self {
            name: "generic".into(),
            max_threads_per_sm: 1024,
            max_blocks_per_sm: 16,
            max_warps_per_sm: 32,
            warp_size: 32,
            registers_per_sm: 32768,
            register_alloc_granularity: 256,
            shared_memory_per_sm: 48 * 1024,
            shared_memory_alloc_granularity: 256,
            sm_count: 1,
        }
    }
}

/// Kernel resource requirements.
#[derive(Debug, Clone)]
pub struct KernelResources {
    /// Threads per block (block size).
    pub threads_per_block: u32,
    /// Registers used per thread.
    pub registers_per_thread: u32,
    /// Static shared memory per block (bytes).
    pub shared_memory_static: u32,
    /// Dynamic shared memory per block (bytes).
    pub shared_memory_dynamic: u32,
}

impl KernelResources {
    /// Create with basic info.
    pub fn new(threads_per_block: u32, registers_per_thread: u32, shared_memory: u32) -> Self {
        Self {
            threads_per_block,
            registers_per_thread,
            shared_memory_static: shared_memory,
            shared_memory_dynamic: 0,
        }
    }

    /// Total shared memory per block.
    pub fn total_shared_memory(&self) -> u32 {
        self.shared_memory_static + self.shared_memory_dynamic
    }
}

/// Occupancy calculation result.
#[derive(Debug, Clone)]
pub struct OccupancyResult {
    /// Active blocks per SM.
    pub active_blocks_per_sm: u32,
    /// Active warps per SM.
    pub active_warps_per_sm: u32,
    /// Max warps per SM (hardware limit).
    pub max_warps_per_sm: u32,
    /// Occupancy as a fraction (0.0 to 1.0).
    pub occupancy: f64,
    /// Which resource is the bottleneck.
    pub limiting_factor: LimitingFactor,
    /// Blocks limited by thread count.
    pub blocks_limited_by_threads: u32,
    /// Blocks limited by registers.
    pub blocks_limited_by_registers: u32,
    /// Blocks limited by shared memory.
    pub blocks_limited_by_smem: u32,
    /// Blocks limited by max-blocks-per-SM.
    pub blocks_limited_by_max_blocks: u32,
}

/// Resource that limits occupancy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LimitingFactor {
    Threads,
    Registers,
    SharedMemory,
    MaxBlocksPerSm,
}

/// Calculate occupancy for a kernel on a given GPU architecture.
pub fn calculate_occupancy(arch: &GpuArchSpec, kernel: &KernelResources) -> OccupancyResult {
    let warp_size = arch.warp_size;

    // Warps per block (round up)
    let warps_per_block = (kernel.threads_per_block + warp_size - 1) / warp_size;

    // 1. Thread limit
    let blocks_by_threads = if warps_per_block > 0 {
        arch.max_warps_per_sm / warps_per_block
    } else {
        0
    };

    // 2. Register limit
    let regs_per_warp = kernel.registers_per_thread * warp_size;
    let regs_per_warp_aligned = round_up(regs_per_warp, arch.register_alloc_granularity);
    let regs_per_block = regs_per_warp_aligned * warps_per_block;
    let blocks_by_registers = if regs_per_block > 0 {
        arch.registers_per_sm / regs_per_block
    } else {
        arch.max_blocks_per_sm
    };

    // 3. Shared memory limit
    let smem_per_block = kernel.total_shared_memory();
    let smem_aligned = round_up(smem_per_block, arch.shared_memory_alloc_granularity);
    let blocks_by_smem = if smem_aligned > 0 {
        arch.shared_memory_per_sm / smem_aligned
    } else {
        arch.max_blocks_per_sm
    };

    // 4. Max blocks per SM limit
    let blocks_by_max = arch.max_blocks_per_sm;

    // Take the minimum
    let active_blocks = blocks_by_threads
        .min(blocks_by_registers)
        .min(blocks_by_smem)
        .min(blocks_by_max);

    let active_warps = active_blocks * warps_per_block;
    let occupancy = active_warps as f64 / arch.max_warps_per_sm as f64;

    let limiting_factor = if active_blocks == blocks_by_threads {
        LimitingFactor::Threads
    } else if active_blocks == blocks_by_registers {
        LimitingFactor::Registers
    } else if active_blocks == blocks_by_smem {
        LimitingFactor::SharedMemory
    } else {
        LimitingFactor::MaxBlocksPerSm
    };

    OccupancyResult {
        active_blocks_per_sm: active_blocks,
        active_warps_per_sm: active_warps,
        max_warps_per_sm: arch.max_warps_per_sm,
        occupancy,
        limiting_factor,
        blocks_limited_by_threads: blocks_by_threads,
        blocks_limited_by_registers: blocks_by_registers,
        blocks_limited_by_smem: blocks_by_smem,
        blocks_limited_by_max_blocks: blocks_by_max,
    }
}

/// Suggest optimal block size for maximum occupancy.
pub fn suggest_block_size(arch: &GpuArchSpec, registers_per_thread: u32, shared_memory: u32) -> BlockSizeSuggestion {
    let mut best_occupancy = 0.0;
    let mut best_block_size = arch.warp_size;
    let mut results = Vec::new();

    // Try block sizes from 1 warp to max
    let max_block = arch.max_threads_per_sm.min(1024);
    let mut block_size = arch.warp_size;

    while block_size <= max_block {
        let kernel = KernelResources::new(block_size, registers_per_thread, shared_memory);
        let result = calculate_occupancy(arch, &kernel);

        results.push((block_size, result.occupancy));

        if result.occupancy > best_occupancy {
            best_occupancy = result.occupancy;
            best_block_size = block_size;
        }
        block_size += arch.warp_size;
    }

    BlockSizeSuggestion {
        optimal_block_size: best_block_size,
        max_occupancy: best_occupancy,
        all_results: results,
    }
}

/// Block size suggestion result.
#[derive(Debug)]
pub struct BlockSizeSuggestion {
    pub optimal_block_size: u32,
    pub max_occupancy: f64,
    pub all_results: Vec<(u32, f64)>,
}

fn round_up(value: u32, granularity: u32) -> u32 {
    if granularity == 0 { return value; }
    ((value + granularity - 1) / granularity) * granularity
}

impl fmt::Display for OccupancyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Occupancy: {:.1}% ({}/{} warps, {} blocks/SM, limited by {:?})",
            self.occupancy * 100.0,
            self.active_warps_per_sm,
            self.max_warps_per_sm,
            self.active_blocks_per_sm,
            self.limiting_factor)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_occupancy_basic() {
        let arch = GpuArchSpec::ampere();
        let kernel = KernelResources::new(256, 32, 0);
        let result = calculate_occupancy(&arch, &kernel);

        assert!(result.occupancy > 0.0);
        assert!(result.occupancy <= 1.0);
        assert!(result.active_blocks_per_sm > 0);
    }

    #[test]
    fn test_occupancy_register_limited() {
        let arch = GpuArchSpec::ampere();
        // High register usage: 128 regs/thread, 256 threads
        let kernel = KernelResources::new(256, 128, 0);
        let result = calculate_occupancy(&arch, &kernel);

        assert!(result.occupancy < 1.0);
        // With 128 regs * 256 threads = 32768 regs per block
        // Ampere has 65536 regs/SM → 2 blocks
        assert!(result.active_blocks_per_sm <= 2);
    }

    #[test]
    fn test_occupancy_smem_limited() {
        let arch = GpuArchSpec::ampere();
        // Large shared memory: 48KB per block
        let kernel = KernelResources::new(256, 32, 48 * 1024);
        let result = calculate_occupancy(&arch, &kernel);

        // Ampere: 164KB smem → ~3 blocks with 48KB each
        assert!(result.active_blocks_per_sm <= 4);
    }

    #[test]
    fn test_occupancy_full() {
        let arch = GpuArchSpec::ampere();
        // Small kernel: should achieve near 100%
        let kernel = KernelResources::new(64, 16, 0);
        let result = calculate_occupancy(&arch, &kernel);
        assert!(result.occupancy >= 0.5, "Expected high occupancy, got {}", result.occupancy);
    }

    #[test]
    fn test_suggest_block_size() {
        let arch = GpuArchSpec::ampere();
        let suggestion = suggest_block_size(&arch, 32, 0);

        assert!(suggestion.optimal_block_size >= 32);
        assert!(suggestion.optimal_block_size <= 1024);
        assert!(suggestion.max_occupancy > 0.0);
        assert!(!suggestion.all_results.is_empty());
    }

    #[test]
    fn test_hopper_arch() {
        let arch = GpuArchSpec::hopper();
        let kernel = KernelResources::new(256, 32, 0);
        let result = calculate_occupancy(&arch, &kernel);
        // Hopper: 2048 threads, 64 warps → 256 threads = 8 warps per block
        // 64/8 = 8 blocks (thread-limited)
        assert!(result.active_blocks_per_sm > 0);
        assert!(result.occupancy > 0.0);
    }

    #[test]
    fn test_amd_cdna3() {
        let arch = GpuArchSpec::cdna3();
        let kernel = KernelResources::new(256, 32, 0);
        let result = calculate_occupancy(&arch, &kernel);
        // AMD warp size = 64 → 256/64 = 4 warps per block
        assert!(result.active_blocks_per_sm > 0);
    }

    #[test]
    fn test_occupancy_display() {
        let result = OccupancyResult {
            active_blocks_per_sm: 8,
            active_warps_per_sm: 64,
            max_warps_per_sm: 64,
            occupancy: 1.0,
            limiting_factor: LimitingFactor::Threads,
            blocks_limited_by_threads: 8,
            blocks_limited_by_registers: 16,
            blocks_limited_by_smem: 32,
            blocks_limited_by_max_blocks: 32,
        };
        let s = format!("{}", result);
        assert!(s.contains("100.0%"));
        assert!(s.contains("Threads"));
    }

    #[test]
    fn test_dynamic_shared_memory() {
        let mut kernel = KernelResources::new(256, 32, 1024);
        kernel.shared_memory_dynamic = 2048;
        assert_eq!(kernel.total_shared_memory(), 3072);

        let arch = GpuArchSpec::ampere();
        let result = calculate_occupancy(&arch, &kernel);
        assert!(result.occupancy > 0.0);
    }
}
