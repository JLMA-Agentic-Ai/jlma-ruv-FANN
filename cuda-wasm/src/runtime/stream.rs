//! CUDA stream abstraction for asynchronous operations
//!
//! Streams provide ordered execution queues. On CPU backends all operations
//! are synchronous, so "synchronize" and "is_complete" reflect wall-clock
//! state tracked via an atomic counter.

use crate::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use super::Device;

/// Stream for asynchronous GPU operations.
///
/// Tracks a monotonically increasing "pending operations" counter.
/// Each operation increments the counter when submitted and decrements
/// when complete. On CPU backends the counter is always zero after
/// synchronous execution.
pub struct Stream {
    device: Arc<Device>,
    /// Number of in-flight operations.
    pending: AtomicU64,
    /// Total operations submitted through this stream.
    total_ops: AtomicU64,
}

impl Stream {
    /// Create a new stream associated with `device`.
    pub fn new(device: Arc<Device>) -> Result<Self> {
        Ok(Self {
            device,
            pending: AtomicU64::new(0),
            total_ops: AtomicU64::new(0),
        })
    }

    /// Get the device associated with this stream.
    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    /// Record a submitted operation (increment pending counter).
    pub fn record_submit(&self) {
        self.pending.fetch_add(1, Ordering::SeqCst);
        self.total_ops.fetch_add(1, Ordering::SeqCst);
    }

    /// Record a completed operation (decrement pending counter).
    pub fn record_complete(&self) {
        self.pending.fetch_sub(1, Ordering::SeqCst);
    }

    /// Synchronize the stream — block until all pending operations complete.
    ///
    /// On CPU backends all operations are already synchronous, so this
    /// simply verifies the pending counter is zero.
    pub fn synchronize(&self) -> Result<()> {
        // CPU backend: operations complete inline so counter should be 0.
        // Spin briefly to handle any race on decrement.
        let mut spins = 0u32;
        while self.pending.load(Ordering::SeqCst) > 0 {
            std::thread::yield_now();
            spins += 1;
            if spins > 10_000 {
                return Err(crate::runtime_error!(
                    "Stream synchronize timed out with {} pending operations",
                    self.pending.load(Ordering::SeqCst)
                ));
            }
        }
        Ok(())
    }

    /// Check if all stream operations are complete.
    pub fn is_complete(&self) -> bool {
        self.pending.load(Ordering::SeqCst) == 0
    }

    /// Get the number of pending operations.
    pub fn pending_ops(&self) -> u64 {
        self.pending.load(Ordering::SeqCst)
    }

    /// Get the total number of operations submitted.
    pub fn total_ops(&self) -> u64 {
        self.total_ops.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Device;

    #[test]
    fn test_stream_creation() {
        let device = Device::get_default().unwrap();
        let stream = Stream::new(device).unwrap();
        assert!(stream.is_complete());
        assert_eq!(stream.pending_ops(), 0);
    }

    #[test]
    fn test_stream_operation_tracking() {
        let device = Device::get_default().unwrap();
        let stream = Stream::new(device).unwrap();

        stream.record_submit();
        assert!(!stream.is_complete());
        assert_eq!(stream.pending_ops(), 1);

        stream.record_complete();
        assert!(stream.is_complete());
        assert_eq!(stream.total_ops(), 1);
    }

    #[test]
    fn test_stream_synchronize() {
        let device = Device::get_default().unwrap();
        let stream = Stream::new(device).unwrap();
        assert!(stream.synchronize().is_ok());
    }
}
