//! Memory Coalescing Analyzer
//!
//! Analyzes memory access patterns in kernel code to detect coalesced
//! vs. scattered accesses. Coalesced accesses (threads in a warp accessing
//! consecutive addresses) are critical for GPU performance — up to 32x
//! difference between coalesced and uncoalesced patterns.
//!
//! This module provides:
//! - Static pattern analysis from array access expressions
//! - Runtime access pattern recording and analysis
//! - Optimization suggestions

use std::fmt;
use std::collections::HashMap;

/// Memory access pattern classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessPattern {
    /// Perfectly coalesced: thread i accesses address base + i * elem_size
    FullyCoalesced,
    /// Strided: thread i accesses base + i * stride (stride > elem_size)
    Strided { stride: usize },
    /// Random/scattered: no detectable pattern
    Scattered,
    /// Broadcast: all threads access same address
    Broadcast,
    /// Block-cyclic: threads access in groups
    BlockCyclic { block_size: usize },
}

/// A recorded memory access for analysis.
#[derive(Debug, Clone)]
pub struct MemoryAccess {
    /// Thread ID within warp (0-31).
    pub lane_id: u32,
    /// Byte address accessed.
    pub address: usize,
    /// Read or write.
    pub is_write: bool,
    /// Element size in bytes.
    pub elem_size: usize,
}

/// Result of coalescing analysis.
#[derive(Debug, Clone)]
pub struct CoalescingReport {
    /// Detected pattern.
    pub pattern: AccessPattern,
    /// Number of memory transactions needed (fewer = better).
    /// Ideal: 1 transaction for 32 threads. Worst: 32 transactions.
    pub transactions: u32,
    /// Efficiency: useful bytes / total bytes transferred (0.0 to 1.0).
    pub efficiency: f64,
    /// Cache line utilization (assuming 128-byte cache lines).
    pub cache_lines_touched: u32,
    /// Optimization suggestion.
    pub suggestion: String,
}

impl fmt::Display for CoalescingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coalescing: {:?}, {} transactions, {:.1}% efficiency — {}",
            self.pattern, self.transactions, self.efficiency * 100.0, self.suggestion)
    }
}

/// Cache line size in bytes (GPU L1 cache line).
const CACHE_LINE_SIZE: usize = 128;
/// GPU memory transaction size in bytes.
const TRANSACTION_SIZE: usize = 32;

/// Analyze a set of memory accesses from one warp (32 threads).
pub fn analyze_warp_access(accesses: &[MemoryAccess]) -> CoalescingReport {
    if accesses.is_empty() {
        return CoalescingReport {
            pattern: AccessPattern::FullyCoalesced,
            transactions: 0,
            efficiency: 1.0,
            cache_lines_touched: 0,
            suggestion: "No accesses to analyze".into(),
        };
    }

    let elem_size = accesses[0].elem_size;

    // Sort by lane_id
    let mut sorted = accesses.to_vec();
    sorted.sort_by_key(|a| a.lane_id);

    // Check for broadcast
    if sorted.windows(2).all(|w| w[0].address == w[1].address) {
        return CoalescingReport {
            pattern: AccessPattern::Broadcast,
            transactions: 1,
            efficiency: elem_size as f64 / TRANSACTION_SIZE as f64,
            cache_lines_touched: 1,
            suggestion: "Broadcast access — consider using shared memory or constant cache".into(),
        };
    }

    // Detect stride pattern
    let mut strides = Vec::new();
    for window in sorted.windows(2) {
        if window[1].address >= window[0].address {
            strides.push(window[1].address - window[0].address);
        }
    }

    // Count unique cache lines touched
    let mut cache_lines: Vec<usize> = accesses.iter()
        .map(|a| a.address / CACHE_LINE_SIZE)
        .collect();
    cache_lines.sort();
    cache_lines.dedup();
    let cache_lines_touched = cache_lines.len() as u32;

    // Count memory transactions (32-byte segments)
    let mut segments: Vec<usize> = accesses.iter()
        .map(|a| a.address / TRANSACTION_SIZE)
        .collect();
    segments.sort();
    segments.dedup();
    let transactions = segments.len() as u32;

    let useful_bytes = accesses.len() * elem_size;
    let total_bytes = transactions as usize * TRANSACTION_SIZE;
    let efficiency = if total_bytes > 0 {
        useful_bytes as f64 / total_bytes as f64
    } else {
        1.0
    };

    // Classify pattern
    let is_uniform_stride = !strides.is_empty() && strides.iter().all(|&s| s == strides[0]);

    let (pattern, suggestion) = if is_uniform_stride {
        let stride = strides[0];
        if stride == elem_size {
            (AccessPattern::FullyCoalesced,
             "Fully coalesced — optimal memory access pattern".into())
        } else if stride == 0 {
            (AccessPattern::Broadcast,
             "Broadcast — consider constant memory cache".into())
        } else {
            let stride_ratio = stride / elem_size;
            (AccessPattern::Strided { stride },
             format!("Stride-{} access — consider transposing data layout or using shared memory tiling", stride_ratio))
        }
    } else {
        (AccessPattern::Scattered,
         "Scattered access — consider sorting indices or using texture cache".into())
    };

    CoalescingReport {
        pattern,
        transactions,
        efficiency,
        cache_lines_touched,
        suggestion,
    }
}

/// Simulate warp access pattern for a linear index expression.
///
/// Models: `address = base + (thread_id * stride + offset) * elem_size`
pub fn simulate_linear_access(
    base: usize,
    stride: usize,
    offset: usize,
    elem_size: usize,
    warp_size: u32,
) -> Vec<MemoryAccess> {
    (0..warp_size).map(|lane| {
        MemoryAccess {
            lane_id: lane,
            address: base + (lane as usize * stride + offset) * elem_size,
            is_write: false,
            elem_size,
        }
    }).collect()
}

/// Simulate column-major access pattern (common anti-pattern).
///
/// Models: `address = base + (thread_id * num_cols + col) * elem_size`
pub fn simulate_column_access(
    base: usize,
    num_cols: usize,
    col: usize,
    elem_size: usize,
    warp_size: u32,
) -> Vec<MemoryAccess> {
    (0..warp_size).map(|lane| {
        MemoryAccess {
            lane_id: lane,
            address: base + (lane as usize * num_cols + col) * elem_size,
            is_write: false,
            elem_size,
        }
    }).collect()
}

/// Runtime access pattern recorder.
pub struct AccessRecorder {
    accesses: Vec<Vec<MemoryAccess>>,
    current_warp: Vec<MemoryAccess>,
}

impl AccessRecorder {
    /// Create a new recorder.
    pub fn new() -> Self {
        Self {
            accesses: Vec::new(),
            current_warp: Vec::new(),
        }
    }

    /// Record a memory access.
    pub fn record(&mut self, lane_id: u32, address: usize, elem_size: usize, is_write: bool) {
        self.current_warp.push(MemoryAccess {
            lane_id,
            address,
            is_write,
            elem_size,
        });

        if self.current_warp.len() >= 32 {
            self.flush_warp();
        }
    }

    /// Flush current warp to history.
    pub fn flush_warp(&mut self) {
        if !self.current_warp.is_empty() {
            self.accesses.push(std::mem::take(&mut self.current_warp));
        }
    }

    /// Analyze all recorded access patterns.
    pub fn analyze(&mut self) -> Vec<CoalescingReport> {
        self.flush_warp();
        self.accesses.iter().map(|warp| analyze_warp_access(warp)).collect()
    }

    /// Get a summary of all recorded patterns.
    pub fn summary(&mut self) -> AccessSummary {
        let reports = self.analyze();
        let mut pattern_counts: HashMap<String, usize> = HashMap::new();
        let mut total_efficiency = 0.0;
        let mut total_transactions = 0u32;

        for report in &reports {
            let key = format!("{:?}", report.pattern);
            *pattern_counts.entry(key).or_insert(0) += 1;
            total_efficiency += report.efficiency;
            total_transactions += report.transactions;
        }

        let count = reports.len();
        AccessSummary {
            total_warps_analyzed: count,
            avg_efficiency: if count > 0 { total_efficiency / count as f64 } else { 0.0 },
            total_transactions,
            pattern_distribution: pattern_counts,
        }
    }
}

/// Summary of access pattern analysis.
#[derive(Debug)]
pub struct AccessSummary {
    pub total_warps_analyzed: usize,
    pub avg_efficiency: f64,
    pub total_transactions: u32,
    pub pattern_distribution: HashMap<String, usize>,
}

impl fmt::Display for AccessSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Access Summary: {} warps, {:.1}% avg efficiency, {} transactions",
            self.total_warps_analyzed,
            self.avg_efficiency * 100.0,
            self.total_transactions)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coalesced_access() {
        // Thread i accesses address base + i * 4 (perfect coalescing for f32)
        let accesses = simulate_linear_access(0, 1, 0, 4, 32);
        let report = analyze_warp_access(&accesses);
        assert_eq!(report.pattern, AccessPattern::FullyCoalesced);
        assert!(report.efficiency > 0.9);
    }

    #[test]
    fn test_strided_access() {
        // Thread i accesses address base + i * 512 * 4 (column of 512-wide matrix)
        let accesses = simulate_column_access(0, 512, 0, 4, 32);
        let report = analyze_warp_access(&accesses);
        match report.pattern {
            AccessPattern::Strided { stride } => assert_eq!(stride, 512 * 4),
            _ => panic!("Expected strided pattern, got {:?}", report.pattern),
        }
        assert!(report.efficiency < 0.2, "Strided access should have low efficiency: {}", report.efficiency);
    }

    #[test]
    fn test_broadcast_access() {
        let accesses: Vec<MemoryAccess> = (0..32).map(|lane| {
            MemoryAccess { lane_id: lane, address: 1000, is_write: false, elem_size: 4 }
        }).collect();
        let report = analyze_warp_access(&accesses);
        assert_eq!(report.pattern, AccessPattern::Broadcast);
        assert_eq!(report.transactions, 1);
    }

    #[test]
    fn test_scattered_access() {
        let addresses = [100, 5000, 200, 9000, 50, 7000, 300, 2000,
                        400, 6000, 150, 8000, 250, 3000, 350, 1000,
                        450, 4000, 550, 10000, 650, 11000, 750, 12000,
                        850, 13000, 950, 14000, 1050, 15000, 1150, 16000];
        let accesses: Vec<MemoryAccess> = addresses.iter().enumerate().map(|(i, &addr)| {
            MemoryAccess { lane_id: i as u32, address: addr, is_write: false, elem_size: 4 }
        }).collect();
        let report = analyze_warp_access(&accesses);
        assert_eq!(report.pattern, AccessPattern::Scattered);
        assert!(report.transactions > 1);
    }

    #[test]
    fn test_recorder() {
        let mut recorder = AccessRecorder::new();
        for lane in 0..32 {
            recorder.record(lane, (lane as usize) * 4, 4, false);
        }
        let reports = recorder.analyze();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].pattern, AccessPattern::FullyCoalesced);
    }

    #[test]
    fn test_summary() {
        let mut recorder = AccessRecorder::new();
        // Two warps: one coalesced, one strided
        for lane in 0..32 {
            recorder.record(lane, (lane as usize) * 4, 4, false);
        }
        for lane in 0..32 {
            recorder.record(lane, (lane as usize) * 2048, 4, false);
        }
        let summary = recorder.summary();
        assert_eq!(summary.total_warps_analyzed, 2);
        assert!(summary.avg_efficiency > 0.0);
    }

    #[test]
    fn test_report_display() {
        let report = CoalescingReport {
            pattern: AccessPattern::FullyCoalesced,
            transactions: 4,
            efficiency: 1.0,
            cache_lines_touched: 1,
            suggestion: "Optimal".into(),
        };
        let s = format!("{}", report);
        assert!(s.contains("100.0%"));
    }

    #[test]
    fn test_empty_access() {
        let report = analyze_warp_access(&[]);
        assert_eq!(report.transactions, 0);
        assert_eq!(report.pattern, AccessPattern::FullyCoalesced);
    }
}
