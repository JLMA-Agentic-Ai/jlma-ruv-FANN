//! CUDA event abstraction for timing and synchronization
//!
//! Events record timestamps and support elapsed-time queries, mirroring
//! CUDA's `cudaEvent_t` semantics using host-side `Instant`.

use crate::Result;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Event for GPU synchronization and timing.
///
/// On CPU and emulated backends the "recording" simply snapshots `Instant::now()`.
/// On real GPU backends the timestamps would come from device-side queries.
pub struct Event {
    recorded_at: Mutex<Option<Instant>>,
}

impl Event {
    /// Create a new event (not yet recorded).
    pub fn new() -> Result<Self> {
        Ok(Self {
            recorded_at: Mutex::new(None),
        })
    }

    /// Record the event — captures the current timestamp.
    pub fn record(&self) -> Result<()> {
        let mut ts = self.recorded_at.lock().map_err(|e| {
            crate::runtime_error!("Event lock poisoned: {}", e)
        })?;
        *ts = Some(Instant::now());
        Ok(())
    }

    /// Synchronize on the event (wait until the recorded work completes).
    ///
    /// On CPU backends all work is synchronous so this returns immediately
    /// once the event has been recorded.
    pub fn synchronize(&self) -> Result<()> {
        let ts = self.recorded_at.lock().map_err(|e| {
            crate::runtime_error!("Event lock poisoned: {}", e)
        })?;
        if ts.is_none() {
            return Err(crate::runtime_error!("Event has not been recorded"));
        }
        // On CPU backends work is already complete at record time.
        Ok(())
    }

    /// Calculate elapsed time between this (start) event and `end`.
    ///
    /// Both events must have been recorded. Returns the wall-clock duration.
    pub fn elapsed_time(&self, end: &Event) -> Result<Duration> {
        let start_ts = self.recorded_at.lock().map_err(|e| {
            crate::runtime_error!("Event lock poisoned: {}", e)
        })?;
        let end_ts = end.recorded_at.lock().map_err(|e| {
            crate::runtime_error!("Event lock poisoned: {}", e)
        })?;

        let start = start_ts.ok_or_else(|| {
            crate::runtime_error!("Start event has not been recorded")
        })?;
        let end = end_ts.ok_or_else(|| {
            crate::runtime_error!("End event has not been recorded")
        })?;

        Ok(end.duration_since(start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_record_and_elapsed() {
        let start = Event::new().unwrap();
        let end = Event::new().unwrap();

        start.record().unwrap();
        std::thread::sleep(Duration::from_millis(10));
        end.record().unwrap();

        let elapsed = start.elapsed_time(&end).unwrap();
        assert!(elapsed >= Duration::from_millis(5));
    }

    #[test]
    fn test_event_synchronize() {
        let e = Event::new().unwrap();
        assert!(e.synchronize().is_err()); // not recorded yet
        e.record().unwrap();
        assert!(e.synchronize().is_ok());
    }

    #[test]
    fn test_event_not_recorded_error() {
        let start = Event::new().unwrap();
        let end = Event::new().unwrap();
        assert!(start.elapsed_time(&end).is_err());
    }
}
