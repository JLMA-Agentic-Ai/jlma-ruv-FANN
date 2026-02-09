//! Async Pipeline — overlapping compute and memory operations
//!
//! Models CUDA's asynchronous memory pipeline where H2D copies, kernel
//! execution, and D2H copies can overlap across streams. Implements a
//! multi-stage pipeline scheduler that maximizes throughput by keeping
//! the GPU busy while data transfers happen concurrently.

use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

/// A stage in the async pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineStage {
    /// Host-to-Device memory transfer.
    HostToDevice,
    /// Kernel compute execution.
    Compute,
    /// Device-to-Host memory transfer.
    DeviceToHost,
}

/// A pipeline operation with timing info.
#[derive(Debug, Clone)]
pub struct PipelineOp {
    pub id: usize,
    pub stage: PipelineStage,
    pub data_bytes: usize,
    /// Estimated duration (for planning).
    pub estimated_duration: Duration,
    /// Actual start time (set during execution).
    pub start_time: Option<Instant>,
    /// Actual end time (set during execution).
    pub end_time: Option<Instant>,
    /// Stream ID this op is assigned to.
    pub stream_id: usize,
}

/// Multi-stage async pipeline scheduler.
///
/// Models the classic triple-buffered pipeline:
/// ```text
/// Stream 0:  [H2D_0] [Compute_0] [D2H_0]
/// Stream 1:          [H2D_1] [Compute_1] [D2H_1]
/// Stream 2:                  [H2D_2] [Compute_2] [D2H_2]
/// ```
pub struct AsyncPipeline {
    /// Number of concurrent streams.
    num_streams: usize,
    /// Pipeline depth (number of stages in flight).
    pipeline_depth: usize,
    /// Queued operations per stream.
    stream_queues: Vec<VecDeque<PipelineOp>>,
    /// Completed operations.
    completed: Vec<PipelineOp>,
    /// Next operation ID.
    next_id: usize,
    /// Batch counter (for round-robin stream assignment).
    batch_count: usize,
    /// H2D bandwidth (bytes/sec) for estimation.
    h2d_bandwidth: f64,
    /// D2H bandwidth (bytes/sec) for estimation.
    d2h_bandwidth: f64,
    /// Compute throughput (FLOPS) for estimation.
    compute_throughput: f64,
}

impl AsyncPipeline {
    /// Create a pipeline with the given number of streams.
    pub fn new(num_streams: usize) -> Self {
        Self {
            num_streams,
            pipeline_depth: num_streams,
            stream_queues: (0..num_streams).map(|_| VecDeque::new()).collect(),
            completed: Vec::new(),
            next_id: 0,
            batch_count: 0,
            h2d_bandwidth: 12e9,       // 12 GB/s PCIe 4.0 x16
            d2h_bandwidth: 12e9,
            compute_throughput: 20e12, // 20 TFLOPS
        }
    }

    /// Set bandwidth parameters for estimation.
    pub fn with_bandwidth(mut self, h2d: f64, d2h: f64, compute: f64) -> Self {
        self.h2d_bandwidth = h2d;
        self.d2h_bandwidth = d2h;
        self.compute_throughput = compute;
        self
    }

    /// Enqueue a batch: H2D → Compute → D2H as a pipeline stage.
    pub fn enqueue_batch(&mut self, data_bytes: usize, compute_flops: u64) -> PipelineBatch {
        let stream_id = self.batch_count % self.num_streams;
        self.batch_count += 1;

        let h2d = PipelineOp {
            id: self.next_id,
            stage: PipelineStage::HostToDevice,
            data_bytes,
            estimated_duration: Duration::from_secs_f64(data_bytes as f64 / self.h2d_bandwidth),
            start_time: None,
            end_time: None,
            stream_id,
        };
        self.next_id += 1;

        let compute = PipelineOp {
            id: self.next_id,
            stage: PipelineStage::Compute,
            data_bytes: 0,
            estimated_duration: Duration::from_secs_f64(compute_flops as f64 / self.compute_throughput),
            start_time: None,
            end_time: None,
            stream_id,
        };
        self.next_id += 1;

        let d2h = PipelineOp {
            id: self.next_id,
            stage: PipelineStage::DeviceToHost,
            data_bytes,
            estimated_duration: Duration::from_secs_f64(data_bytes as f64 / self.d2h_bandwidth),
            start_time: None,
            end_time: None,
            stream_id,
        };
        self.next_id += 1;

        let batch = PipelineBatch {
            h2d_id: h2d.id,
            compute_id: compute.id,
            d2h_id: d2h.id,
            stream_id,
            total_estimated: h2d.estimated_duration + compute.estimated_duration + d2h.estimated_duration,
        };

        self.stream_queues[stream_id].push_back(h2d);
        self.stream_queues[stream_id].push_back(compute);
        self.stream_queues[stream_id].push_back(d2h);

        batch
    }

    /// Simulate pipeline execution and return timeline.
    pub fn simulate(&mut self) -> PipelineTimeline {
        let start = Instant::now();
        let mut events = Vec::new();
        let mut stream_end_times = vec![Duration::ZERO; self.num_streams];

        // Drain all queues, simulating execution
        loop {
            let mut any_progress = false;

            for stream_id in 0..self.num_streams {
                if let Some(mut op) = self.stream_queues[stream_id].pop_front() {
                    any_progress = true;

                    let op_start = stream_end_times[stream_id];
                    let op_end = op_start + op.estimated_duration;
                    stream_end_times[stream_id] = op_end;

                    op.start_time = Some(start + op_start);
                    op.end_time = Some(start + op_end);

                    events.push(PipelineEvent {
                        op_id: op.id,
                        stage: op.stage,
                        stream_id: op.stream_id,
                        start_offset: op_start,
                        end_offset: op_end,
                        data_bytes: op.data_bytes,
                    });

                    self.completed.push(op);
                }
            }

            if !any_progress {
                break;
            }
        }

        let total_time = stream_end_times.iter().cloned().max().unwrap_or(Duration::ZERO);

        // Calculate sequential time (no overlap)
        let sequential_time: Duration = events.iter()
            .map(|e| e.end_offset - e.start_offset)
            .sum();

        PipelineTimeline {
            events,
            total_time,
            sequential_time,
            speedup: if total_time.as_secs_f64() > 0.0 {
                sequential_time.as_secs_f64() / total_time.as_secs_f64()
            } else {
                1.0
            },
            num_streams: self.num_streams,
        }
    }

    /// Get pipeline utilization estimate.
    pub fn utilization(&self) -> PipelineUtilization {
        let total_ops: usize = self.stream_queues.iter().map(|q| q.len()).sum();
        let active_streams = self.stream_queues.iter().filter(|q| !q.is_empty()).count();

        PipelineUtilization {
            total_pending_ops: total_ops,
            active_streams,
            total_streams: self.num_streams,
            pipeline_depth: self.pipeline_depth,
            utilization: if self.num_streams > 0 {
                active_streams as f64 / self.num_streams as f64
            } else {
                0.0
            },
        }
    }
}

/// A batch of H2D → Compute → D2H operations.
#[derive(Debug, Clone)]
pub struct PipelineBatch {
    pub h2d_id: usize,
    pub compute_id: usize,
    pub d2h_id: usize,
    pub stream_id: usize,
    pub total_estimated: Duration,
}

/// Timeline event for visualization.
#[derive(Debug, Clone)]
pub struct PipelineEvent {
    pub op_id: usize,
    pub stage: PipelineStage,
    pub stream_id: usize,
    pub start_offset: Duration,
    pub end_offset: Duration,
    pub data_bytes: usize,
}

/// Simulated pipeline timeline.
#[derive(Debug)]
pub struct PipelineTimeline {
    pub events: Vec<PipelineEvent>,
    pub total_time: Duration,
    pub sequential_time: Duration,
    pub speedup: f64,
    pub num_streams: usize,
}

impl fmt::Display for PipelineTimeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pipeline: {:.2}ms total (sequential: {:.2}ms), {:.2}x speedup, {} streams",
            self.total_time.as_secs_f64() * 1000.0,
            self.sequential_time.as_secs_f64() * 1000.0,
            self.speedup,
            self.num_streams)
    }
}

/// Pipeline utilization stats.
#[derive(Debug)]
pub struct PipelineUtilization {
    pub total_pending_ops: usize,
    pub active_streams: usize,
    pub total_streams: usize,
    pub pipeline_depth: usize,
    pub utilization: f64,
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_basic() {
        let mut pipeline = AsyncPipeline::new(3);
        let batch = pipeline.enqueue_batch(1024 * 1024, 1_000_000);
        assert_eq!(batch.stream_id, 0);
        assert!(batch.total_estimated > Duration::ZERO);
    }

    #[test]
    fn test_pipeline_multi_stream() {
        let mut pipeline = AsyncPipeline::new(3);
        let b0 = pipeline.enqueue_batch(1_000_000, 1_000_000);
        let b1 = pipeline.enqueue_batch(1_000_000, 1_000_000);
        let b2 = pipeline.enqueue_batch(1_000_000, 1_000_000);

        assert_eq!(b0.stream_id, 0);
        assert_eq!(b1.stream_id, 1);
        assert_eq!(b2.stream_id, 2);
    }

    #[test]
    fn test_pipeline_simulate() {
        let mut pipeline = AsyncPipeline::new(2);
        pipeline.enqueue_batch(1_000_000, 1_000_000_000);
        pipeline.enqueue_batch(1_000_000, 1_000_000_000);

        let timeline = pipeline.simulate();
        assert!(!timeline.events.is_empty());
        assert!(timeline.total_time > Duration::ZERO);
        assert!(timeline.speedup >= 1.0);
    }

    #[test]
    fn test_pipeline_speedup() {
        let mut pipeline = AsyncPipeline::new(3)
            .with_bandwidth(10e9, 10e9, 10e12);

        for _ in 0..6 {
            pipeline.enqueue_batch(10_000_000, 100_000_000_000);
        }

        let timeline = pipeline.simulate();
        // With 3 streams and 6 batches, pipeline should provide some speedup
        assert!(timeline.speedup >= 1.0,
            "Expected speedup >= 1.0, got {}", timeline.speedup);
    }

    #[test]
    fn test_pipeline_utilization() {
        let mut pipeline = AsyncPipeline::new(4);
        pipeline.enqueue_batch(1024, 1000);
        pipeline.enqueue_batch(1024, 1000);

        let util = pipeline.utilization();
        assert_eq!(util.total_streams, 4);
        assert_eq!(util.active_streams, 2);
        assert!(util.utilization > 0.0 && util.utilization <= 1.0);
    }

    #[test]
    fn test_pipeline_empty() {
        let mut pipeline = AsyncPipeline::new(2);
        let timeline = pipeline.simulate();
        assert!(timeline.events.is_empty());
        assert_eq!(timeline.total_time, Duration::ZERO);
    }

    #[test]
    fn test_pipeline_single_stream() {
        let mut pipeline = AsyncPipeline::new(1);
        pipeline.enqueue_batch(1024, 1000);
        pipeline.enqueue_batch(1024, 1000);

        let timeline = pipeline.simulate();
        // Single stream: no overlap possible
        assert!(timeline.speedup <= 1.01);
    }

    #[test]
    fn test_pipeline_display() {
        let mut pipeline = AsyncPipeline::new(2);
        pipeline.enqueue_batch(1_000_000, 1_000_000);
        let timeline = pipeline.simulate();
        let s = format!("{}", timeline);
        assert!(s.contains("Pipeline:"));
        assert!(s.contains("speedup"));
    }
}
