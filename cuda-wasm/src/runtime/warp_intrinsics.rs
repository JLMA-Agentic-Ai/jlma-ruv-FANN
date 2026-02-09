//! Extended Warp Intrinsics
//!
//! Provides additional warp-level primitives beyond basic shuffles,
//! matching CUDA's warp-synchronous intrinsics from SM 7.0+:
//!
//! - `ballot_sync`  — each thread votes, returns bitmask of votes
//! - `match_any`    — find threads with matching values
//! - `match_all`    — check if all threads have the same value
//! - `reduce_sync`  — warp-wide reduction (sum, min, max, and, or, xor)
//! - `activemask`   — bitmask of active threads
//! - `lanemask_lt`  — bitmask of lanes < current lane
//! - `popc`         — population count (number of set bits)
//! - `ffs`          — find first set bit

/// Default warp size.
pub const WARP_SIZE: u32 = 32;

/// Full warp mask (all 32 lanes active).
pub const FULL_MASK: u32 = 0xFFFF_FFFF;

/// Ballot sync — each thread provides a predicate (bool), returns a bitmask
/// where bit `i` is set if thread `i` voted true.
///
/// Emulates `__ballot_sync(mask, predicate)`.
pub fn ballot_sync(mask: u32, predicates: &[bool]) -> u32 {
    let mut result = 0u32;
    for (lane, &pred) in predicates.iter().enumerate() {
        if lane >= 32 { break; }
        if (mask >> lane) & 1 == 1 && pred {
            result |= 1 << lane;
        }
    }
    result
}

/// All sync — returns true if all active threads in mask vote true.
///
/// Emulates `__all_sync(mask, predicate)`.
pub fn all_sync(mask: u32, predicates: &[bool]) -> bool {
    for lane in 0..32u32 {
        if (mask >> lane) & 1 == 1 {
            if let Some(&pred) = predicates.get(lane as usize) {
                if !pred {
                    return false;
                }
            }
        }
    }
    true
}

/// Any sync — returns true if any active thread in mask votes true.
///
/// Emulates `__any_sync(mask, predicate)`.
pub fn any_sync(mask: u32, predicates: &[bool]) -> bool {
    for lane in 0..32u32 {
        if (mask >> lane) & 1 == 1 {
            if let Some(&pred) = predicates.get(lane as usize) {
                if pred {
                    return true;
                }
            }
        }
    }
    false
}

/// Match any — returns a bitmask of threads that have the same value as `lane_id`.
///
/// Emulates `__match_any_sync(mask, value)`.
pub fn match_any_sync(mask: u32, values: &[u32], lane_id: u32) -> u32 {
    let target = values.get(lane_id as usize).copied().unwrap_or(0);
    let mut result = 0u32;
    for lane in 0..32u32 {
        if (mask >> lane) & 1 == 1 {
            if let Some(&v) = values.get(lane as usize) {
                if v == target {
                    result |= 1 << lane;
                }
            }
        }
    }
    result
}

/// Match all — returns (mask_of_matching, all_match).
/// If all active threads have the same value, returns (mask, true).
///
/// Emulates `__match_all_sync(mask, value, &pred)`.
pub fn match_all_sync(mask: u32, values: &[u32]) -> (u32, bool) {
    let mut first_value = None;
    let mut all_match = true;

    for lane in 0..32u32 {
        if (mask >> lane) & 1 == 1 {
            if let Some(&v) = values.get(lane as usize) {
                match first_value {
                    None => first_value = Some(v),
                    Some(fv) => {
                        if v != fv {
                            all_match = false;
                        }
                    }
                }
            }
        }
    }

    (if all_match { mask } else { 0 }, all_match)
}

/// Warp-wide reduction (sum).
///
/// Emulates `__reduce_add_sync(mask, value)` (SM 8.0+).
pub fn reduce_add_sync(mask: u32, values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for lane in 0..32u32 {
        if (mask >> lane) & 1 == 1 {
            if let Some(&v) = values.get(lane as usize) {
                sum += v;
            }
        }
    }
    sum
}

/// Warp-wide reduction (max).
pub fn reduce_max_sync(mask: u32, values: &[f32]) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for lane in 0..32u32 {
        if (mask >> lane) & 1 == 1 {
            if let Some(&v) = values.get(lane as usize) {
                if v > max { max = v; }
            }
        }
    }
    max
}

/// Warp-wide reduction (min).
pub fn reduce_min_sync(mask: u32, values: &[f32]) -> f32 {
    let mut min = f32::INFINITY;
    for lane in 0..32u32 {
        if (mask >> lane) & 1 == 1 {
            if let Some(&v) = values.get(lane as usize) {
                if v < min { min = v; }
            }
        }
    }
    min
}

/// Warp-wide bitwise AND reduction.
pub fn reduce_and_sync(mask: u32, values: &[u32]) -> u32 {
    let mut result = u32::MAX;
    for lane in 0..32u32 {
        if (mask >> lane) & 1 == 1 {
            if let Some(&v) = values.get(lane as usize) {
                result &= v;
            }
        }
    }
    result
}

/// Warp-wide bitwise OR reduction.
pub fn reduce_or_sync(mask: u32, values: &[u32]) -> u32 {
    let mut result = 0u32;
    for lane in 0..32u32 {
        if (mask >> lane) & 1 == 1 {
            if let Some(&v) = values.get(lane as usize) {
                result |= v;
            }
        }
    }
    result
}

/// Warp-wide bitwise XOR reduction.
pub fn reduce_xor_sync(mask: u32, values: &[u32]) -> u32 {
    let mut result = 0u32;
    for lane in 0..32u32 {
        if (mask >> lane) & 1 == 1 {
            if let Some(&v) = values.get(lane as usize) {
                result ^= v;
            }
        }
    }
    result
}

/// Warp-level inclusive prefix sum (scan).
///
/// Returns a vector where `output[i] = sum(values[0..=i])` for active lanes.
pub fn inclusive_scan_sync(mask: u32, values: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; values.len()];
    let mut running = 0.0f32;
    for lane in 0..32u32 {
        if (mask >> lane) & 1 == 1 {
            if let Some(&v) = values.get(lane as usize) {
                running += v;
            }
        }
        if (lane as usize) < output.len() {
            output[lane as usize] = running;
        }
    }
    output
}

/// Warp-level exclusive prefix sum.
pub fn exclusive_scan_sync(mask: u32, values: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; values.len()];
    let mut running = 0.0f32;
    for lane in 0..32u32 {
        if (lane as usize) < output.len() {
            output[lane as usize] = running;
        }
        if (mask >> lane) & 1 == 1 {
            if let Some(&v) = values.get(lane as usize) {
                running += v;
            }
        }
    }
    output
}

/// Population count — number of set bits.
///
/// Emulates `__popc(x)`.
pub fn popc(x: u32) -> u32 {
    x.count_ones()
}

/// Find first set bit (1-indexed, 0 if no bits set).
///
/// Emulates `__ffs(x)`.
pub fn ffs(x: u32) -> u32 {
    if x == 0 { 0 } else { x.trailing_zeros() + 1 }
}

/// Count leading zeros.
///
/// Emulates `__clz(x)`.
pub fn clz(x: u32) -> u32 {
    x.leading_zeros()
}

/// Lane mask less-than: bitmask of all lanes with ID < current lane.
///
/// Emulates `__lanemask_lt()`.
pub fn lanemask_lt(lane_id: u32) -> u32 {
    if lane_id == 0 { 0 } else { (1u32 << lane_id) - 1 }
}

/// Lane mask less-than-or-equal.
pub fn lanemask_le(lane_id: u32) -> u32 {
    if lane_id >= 31 { FULL_MASK } else { (1u32 << (lane_id + 1)) - 1 }
}

/// Lane mask greater-than.
pub fn lanemask_gt(lane_id: u32) -> u32 {
    !lanemask_le(lane_id)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ballot_sync() {
        let preds = vec![true, false, true, true, false, true, false, false];
        let result = ballot_sync(0xFF, &preds);
        assert_eq!(result & 0xFF, 0b00101101);
    }

    #[test]
    fn test_ballot_sync_with_mask() {
        let preds = vec![true, true, true, true];
        // Only lanes 0 and 2 active
        let result = ballot_sync(0b0101, &preds);
        assert_eq!(result, 0b0101);
    }

    #[test]
    fn test_all_sync() {
        assert!(all_sync(0xFF, &vec![true; 8]));
        assert!(!all_sync(0xFF, &vec![true, true, false, true, true, true, true, true]));
    }

    #[test]
    fn test_any_sync() {
        assert!(any_sync(0xFF, &vec![false, false, true, false, false, false, false, false]));
        assert!(!any_sync(0xFF, &vec![false; 8]));
    }

    #[test]
    fn test_match_any() {
        let values = vec![1, 2, 1, 3, 1, 2, 3, 1];
        let result = match_any_sync(0xFF, &values, 0); // lane 0 has value 1
        // Lanes 0, 2, 4, 7 all have value 1
        assert_eq!(result & 0xFF, 0b10010101);
    }

    #[test]
    fn test_match_all() {
        let uniform = vec![42; 8];
        let (mask, all) = match_all_sync(0xFF, &uniform);
        assert!(all);
        assert_eq!(mask, 0xFF);

        let mixed = vec![1, 2, 1, 1, 1, 1, 1, 1];
        let (_, all2) = match_all_sync(0xFF, &mixed);
        assert!(!all2);
    }

    #[test]
    fn test_reduce_add() {
        let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let sum = reduce_add_sync(0xFF, &values);
        assert!((sum - 28.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_max_min() {
        let values = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        assert!((reduce_max_sync(0xFF, &values) - 9.0).abs() < 1e-6);
        assert!((reduce_min_sync(0xFF, &values) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_bitwise() {
        let values = vec![0xFF, 0x0F, 0xF0, 0x00];
        assert_eq!(reduce_and_sync(0x0F, &values), 0x00);
        assert_eq!(reduce_or_sync(0x0F, &values), 0xFF);
    }

    #[test]
    fn test_inclusive_scan() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let result = inclusive_scan_sync(0x0F, &values);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
        assert!((result[2] - 6.0).abs() < 1e-6);
        assert!((result[3] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_exclusive_scan() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let result = exclusive_scan_sync(0x0F, &values);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
        assert!((result[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_popc() {
        assert_eq!(popc(0), 0);
        assert_eq!(popc(0xFF), 8);
        assert_eq!(popc(0b10101010), 4);
        assert_eq!(popc(FULL_MASK), 32);
    }

    #[test]
    fn test_ffs() {
        assert_eq!(ffs(0), 0);
        assert_eq!(ffs(1), 1);
        assert_eq!(ffs(0b1000), 4);
        assert_eq!(ffs(0b10100), 3);
    }

    #[test]
    fn test_clz() {
        assert_eq!(clz(0), 32);
        assert_eq!(clz(1), 31);
        assert_eq!(clz(0x80000000), 0);
    }

    #[test]
    fn test_lanemask() {
        assert_eq!(lanemask_lt(0), 0);
        assert_eq!(lanemask_lt(1), 0b1);
        assert_eq!(lanemask_lt(4), 0b1111);
        assert_eq!(lanemask_le(0), 0b1);
        assert_eq!(lanemask_gt(30), 0x80000000);
    }

    #[test]
    fn test_reduce_with_partial_mask() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // Only even lanes
        let sum = reduce_add_sync(0b01010101, &values);
        assert!((sum - (1.0 + 3.0 + 5.0 + 7.0)).abs() < 1e-6);
    }
}
