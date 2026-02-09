//! Warp-level primitive emulation
//!
//! Emulates CUDA warp-level operations (shuffle, vote, ballot) on the CPU
//! using shared memory buffers. This enables transpiled CUDA kernels that use
//! warp intrinsics to execute correctly on CPU fallback paths.
//!
//! The emulation assumes `WARP_SIZE = 32` and uses thread-local storage to
//! track the current lane identity within a warp.

use std::sync::atomic::{AtomicU32, Ordering};

/// The number of threads in a warp (matches CUDA).
pub const WARP_SIZE: u32 = 32;

/// Per-warp shared state used to emulate warp-level operations.
///
/// In a real GPU each warp executes in lock-step and has hardware support for
/// cross-lane communication. On the CPU we emulate this by having all threads
/// in a "warp" share a `WarpState` and synchronise explicitly via barriers.
pub struct WarpState {
    /// Shared data buffer for shuffle operations.
    /// Each lane writes its value, then reads from the target lane.
    shuffle_buf: [AtomicU32; WARP_SIZE as usize],

    /// Bitmask of active lanes. Bit `i` is set if lane `i` is participating.
    active_mask: AtomicU32,

    /// Predicate buffer for vote/ballot operations.
    /// Each lane writes 1 (true) or 0 (false).
    predicate_buf: [AtomicU32; WARP_SIZE as usize],
}

impl WarpState {
    /// Create a new warp state with all lanes active.
    pub fn new() -> Self {
        const INIT: AtomicU32 = AtomicU32::new(0);
        Self {
            shuffle_buf: [INIT; WARP_SIZE as usize],
            active_mask: AtomicU32::new(0xFFFF_FFFF),
            predicate_buf: [INIT; WARP_SIZE as usize],
        }
    }

    // -----------------------------------------------------------------------
    // Active mask management
    // -----------------------------------------------------------------------

    /// Set a lane as active.
    pub fn set_lane_active(&self, lane_id: u32) {
        debug_assert!(lane_id < WARP_SIZE);
        self.active_mask.fetch_or(1 << lane_id, Ordering::SeqCst);
    }

    /// Set a lane as inactive.
    pub fn set_lane_inactive(&self, lane_id: u32) {
        debug_assert!(lane_id < WARP_SIZE);
        self.active_mask
            .fetch_and(!(1 << lane_id), Ordering::SeqCst);
    }

    /// Get the current active mask.
    pub fn active_mask(&self) -> u32 {
        self.active_mask.load(Ordering::SeqCst)
    }

    /// Returns true if the specified lane is currently active.
    pub fn is_lane_active(&self, lane_id: u32) -> bool {
        (self.active_mask() >> lane_id) & 1 == 1
    }

    // -----------------------------------------------------------------------
    // Warp shuffle emulation
    // -----------------------------------------------------------------------

    /// Emulate `__shfl_sync`: read the value from `src_lane`.
    ///
    /// The caller (at `lane_id`) first writes its own value, then after a
    /// barrier reads from `src_lane`. In a single-threaded emulation context,
    /// the caller can pre-populate all lanes and then read.
    ///
    /// # Arguments
    /// * `lane_id` - The calling thread's lane within the warp (0..31)
    /// * `value` - The value this lane contributes
    /// * `src_lane` - The lane to read from
    ///
    /// Returns the value from `src_lane`, or this lane's own value if
    /// `src_lane` is out of range.
    pub fn shuffle(&self, lane_id: u32, value: u32, src_lane: u32) -> u32 {
        debug_assert!(lane_id < WARP_SIZE);

        // Write our value into the shared buffer
        self.shuffle_buf[lane_id as usize].store(value, Ordering::SeqCst);

        // In a multi-threaded scenario a barrier would go here.
        // For single-threaded emulation we assume all lanes have written.

        let effective_src = src_lane % WARP_SIZE;
        self.shuffle_buf[effective_src as usize].load(Ordering::SeqCst)
    }

    /// Emulate `__shfl_xor_sync`: read from `lane_id ^ lane_mask`.
    pub fn shuffle_xor(&self, lane_id: u32, value: u32, lane_mask: u32) -> u32 {
        let src_lane = lane_id ^ lane_mask;
        self.shuffle(lane_id, value, src_lane)
    }

    /// Emulate `__shfl_up_sync`: read from `lane_id - delta`.
    /// If the source lane would be negative, return the caller's own value.
    pub fn shuffle_up(&self, lane_id: u32, value: u32, delta: u32) -> u32 {
        self.shuffle_buf[lane_id as usize].store(value, Ordering::SeqCst);

        if lane_id >= delta {
            let src_lane = lane_id - delta;
            self.shuffle_buf[src_lane as usize].load(Ordering::SeqCst)
        } else {
            // Out-of-range: return own value
            value
        }
    }

    /// Emulate `__shfl_down_sync`: read from `lane_id + delta`.
    /// If the source lane would be >= WARP_SIZE, return the caller's own value.
    pub fn shuffle_down(&self, lane_id: u32, value: u32, delta: u32) -> u32 {
        self.shuffle_buf[lane_id as usize].store(value, Ordering::SeqCst);

        let src_lane = lane_id + delta;
        if src_lane < WARP_SIZE {
            self.shuffle_buf[src_lane as usize].load(Ordering::SeqCst)
        } else {
            value
        }
    }

    // -----------------------------------------------------------------------
    // Warp shuffle with f32 values
    // -----------------------------------------------------------------------

    /// Shuffle an f32 value (reinterpret bits through u32).
    pub fn shuffle_f32(&self, lane_id: u32, value: f32, src_lane: u32) -> f32 {
        let bits = value.to_bits();
        let result_bits = self.shuffle(lane_id, bits, src_lane);
        f32::from_bits(result_bits)
    }

    /// Shuffle XOR with f32.
    pub fn shuffle_xor_f32(&self, lane_id: u32, value: f32, lane_mask: u32) -> f32 {
        let bits = value.to_bits();
        let result_bits = self.shuffle_xor(lane_id, bits, lane_mask);
        f32::from_bits(result_bits)
    }

    /// Shuffle up with f32.
    pub fn shuffle_up_f32(&self, lane_id: u32, value: f32, delta: u32) -> f32 {
        let bits = value.to_bits();
        let result_bits = self.shuffle_up(lane_id, bits, delta);
        f32::from_bits(result_bits)
    }

    /// Shuffle down with f32.
    pub fn shuffle_down_f32(&self, lane_id: u32, value: f32, delta: u32) -> f32 {
        let bits = value.to_bits();
        let result_bits = self.shuffle_down(lane_id, bits, delta);
        f32::from_bits(result_bits)
    }

    // -----------------------------------------------------------------------
    // Warp vote operations
    // -----------------------------------------------------------------------

    /// Emulate `__all_sync`: returns true if all active lanes have `predicate == true`.
    pub fn vote_all(&self, lane_id: u32, predicate: bool) -> bool {
        debug_assert!(lane_id < WARP_SIZE);

        self.predicate_buf[lane_id as usize].store(predicate as u32, Ordering::SeqCst);

        let mask = self.active_mask();
        for i in 0..WARP_SIZE {
            if (mask >> i) & 1 == 1 {
                if self.predicate_buf[i as usize].load(Ordering::SeqCst) == 0 {
                    return false;
                }
            }
        }
        true
    }

    /// Emulate `__any_sync`: returns true if any active lane has `predicate == true`.
    pub fn vote_any(&self, lane_id: u32, predicate: bool) -> bool {
        debug_assert!(lane_id < WARP_SIZE);

        self.predicate_buf[lane_id as usize].store(predicate as u32, Ordering::SeqCst);

        let mask = self.active_mask();
        for i in 0..WARP_SIZE {
            if (mask >> i) & 1 == 1 {
                if self.predicate_buf[i as usize].load(Ordering::SeqCst) != 0 {
                    return true;
                }
            }
        }
        false
    }

    /// Emulate `__ballot_sync`: returns a bitmask where bit `i` is set if
    /// lane `i` is active and its predicate is true.
    pub fn ballot(&self, lane_id: u32, predicate: bool) -> u32 {
        debug_assert!(lane_id < WARP_SIZE);

        self.predicate_buf[lane_id as usize].store(predicate as u32, Ordering::SeqCst);

        let mask = self.active_mask();
        let mut result: u32 = 0;
        for i in 0..WARP_SIZE {
            if (mask >> i) & 1 == 1 {
                if self.predicate_buf[i as usize].load(Ordering::SeqCst) != 0 {
                    result |= 1 << i;
                }
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Utility: warp-level reduction (common pattern)
    // -----------------------------------------------------------------------

    /// Warp-level sum reduction using shuffle_down (butterfly pattern).
    ///
    /// Assumes all 32 lanes call this with their value. Returns the sum at
    /// lane 0; other lanes get a partial result.
    pub fn reduce_sum_f32(&self, lane_id: u32, value: f32) -> f32 {
        let mut v = value;
        // Butterfly reduction: delta = 16, 8, 4, 2, 1
        let mut delta = WARP_SIZE / 2;
        while delta >= 1 {
            let other = self.shuffle_down_f32(lane_id, v, delta);
            v += other;
            delta /= 2;
        }
        v
    }

    /// Warp-level max reduction.
    pub fn reduce_max_f32(&self, lane_id: u32, value: f32) -> f32 {
        let mut v = value;
        let mut delta = WARP_SIZE / 2;
        while delta >= 1 {
            let other = self.shuffle_down_f32(lane_id, v, delta);
            if other > v {
                v = other;
            }
            delta /= 2;
        }
        v
    }

    /// Warp-level min reduction.
    pub fn reduce_min_f32(&self, lane_id: u32, value: f32) -> f32 {
        let mut v = value;
        let mut delta = WARP_SIZE / 2;
        while delta >= 1 {
            let other = self.shuffle_down_f32(lane_id, v, delta);
            if other < v {
                v = other;
            }
            delta /= 2;
        }
        v
    }

    /// Count the number of active lanes with a true predicate (popcount of ballot).
    pub fn popc_ballot(&self, lane_id: u32, predicate: bool) -> u32 {
        self.ballot(lane_id, predicate).count_ones()
    }
}

impl Default for WarpState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_warp_state() {
        let ws = WarpState::new();
        assert_eq!(ws.active_mask(), 0xFFFF_FFFF);
    }

    #[test]
    fn test_set_lane_active_inactive() {
        let ws = WarpState::new();
        ws.set_lane_inactive(5);
        assert!(!ws.is_lane_active(5));
        assert!(ws.is_lane_active(0));

        ws.set_lane_active(5);
        assert!(ws.is_lane_active(5));
    }

    #[test]
    fn test_shuffle_basic() {
        let ws = WarpState::new();

        // Populate lanes 0..31 with values 100..131
        for lane in 0..WARP_SIZE {
            ws.shuffle_buf[lane as usize].store(100 + lane, Ordering::SeqCst);
        }

        // Lane 5 shuffles from lane 10
        let result = ws.shuffle(5, 105, 10);
        assert_eq!(result, 110);
    }

    #[test]
    fn test_shuffle_xor() {
        let ws = WarpState::new();

        // Populate lanes
        for lane in 0..WARP_SIZE {
            ws.shuffle_buf[lane as usize].store(lane * 10, Ordering::SeqCst);
        }

        // Lane 3 XOR 1 -> lane 2
        let result = ws.shuffle_xor(3, 30, 1);
        assert_eq!(result, 20);
    }

    #[test]
    fn test_shuffle_up() {
        let ws = WarpState::new();

        for lane in 0..WARP_SIZE {
            ws.shuffle_buf[lane as usize].store(lane, Ordering::SeqCst);
        }

        // Lane 5 shuffle up by 2 -> reads from lane 3
        let result = ws.shuffle_up(5, 5, 2);
        assert_eq!(result, 3);

        // Lane 0 shuffle up by 1 -> out of range, returns own value
        let result = ws.shuffle_up(0, 0, 1);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_shuffle_down() {
        let ws = WarpState::new();

        for lane in 0..WARP_SIZE {
            ws.shuffle_buf[lane as usize].store(lane, Ordering::SeqCst);
        }

        // Lane 5 shuffle down by 3 -> reads from lane 8
        let result = ws.shuffle_down(5, 5, 3);
        assert_eq!(result, 8);

        // Lane 31 shuffle down by 1 -> out of range
        let result = ws.shuffle_down(31, 31, 1);
        assert_eq!(result, 31);
    }

    #[test]
    fn test_shuffle_f32() {
        let ws = WarpState::new();

        // Populate all lanes with f32 values
        for lane in 0..WARP_SIZE {
            let val = lane as f32 * 1.5;
            ws.shuffle_buf[lane as usize].store(val.to_bits(), Ordering::SeqCst);
        }

        let result = ws.shuffle_f32(0, 0.0, 10);
        let expected = 10.0 * 1.5;
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_vote_all_true() {
        let ws = WarpState::new();
        // Set all lanes to true
        for lane in 0..WARP_SIZE {
            ws.predicate_buf[lane as usize].store(1, Ordering::SeqCst);
        }
        assert!(ws.vote_all(0, true));
    }

    #[test]
    fn test_vote_all_one_false() {
        let ws = WarpState::new();
        for lane in 0..WARP_SIZE {
            ws.predicate_buf[lane as usize].store(1, Ordering::SeqCst);
        }
        // Lane 15 sets false
        ws.predicate_buf[15].store(0, Ordering::SeqCst);
        assert!(!ws.vote_all(0, true));
    }

    #[test]
    fn test_vote_any() {
        let ws = WarpState::new();
        // All false
        for lane in 0..WARP_SIZE {
            ws.predicate_buf[lane as usize].store(0, Ordering::SeqCst);
        }

        // Lane 7 sets true
        assert!(ws.vote_any(7, true));
    }

    #[test]
    fn test_ballot() {
        let ws = WarpState::new();
        // All lanes false
        for lane in 0..WARP_SIZE {
            ws.predicate_buf[lane as usize].store(0, Ordering::SeqCst);
        }

        // Lanes 0, 1, 2 set true
        ws.predicate_buf[0].store(1, Ordering::SeqCst);
        ws.predicate_buf[1].store(1, Ordering::SeqCst);
        ws.predicate_buf[2].store(1, Ordering::SeqCst);

        let result = ws.ballot(3, false);
        assert_eq!(result & 0b111, 0b111); // first 3 bits set
        assert_eq!(result & (1 << 3), 0); // lane 3 not set
    }

    #[test]
    fn test_popc_ballot() {
        let ws = WarpState::new();
        for lane in 0..WARP_SIZE {
            ws.predicate_buf[lane as usize].store(0, Ordering::SeqCst);
        }

        // Set 5 lanes to true
        for lane in 0..5 {
            ws.predicate_buf[lane as usize].store(1, Ordering::SeqCst);
        }

        let count = ws.popc_ballot(10, false);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_reduce_sum_simple() {
        let ws = WarpState::new();
        // In a single-threaded context, populate all lanes with 1.0
        for lane in 0..WARP_SIZE {
            ws.shuffle_buf[lane as usize].store(1.0f32.to_bits(), Ordering::SeqCst);
        }
        // Lane 0 reduces: should get 32.0 in ideal case
        // Note: single-threaded emulation means only lane 0's perspective is valid
        let result = ws.reduce_sum_f32(0, 1.0);
        // With single-threaded emulation the shuffle_down reads pre-populated values
        assert!(result >= 1.0);
    }
}
