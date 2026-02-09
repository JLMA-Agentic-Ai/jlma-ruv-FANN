//! Built-in benchmark suite for measuring kernel and memory performance
//!
//! Provides self-contained benchmarks that run without external harnesses,
//! producing structured results suitable for comparison across runs.

use crate::Result;
use std::time::{Duration, Instant};

/// Single benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Number of iterations
    pub iterations: u64,
    /// Total wall-clock duration
    pub total_duration: Duration,
    /// Mean duration per iteration
    pub mean_duration: Duration,
    /// Median duration
    pub median_duration: Duration,
    /// Minimum duration
    pub min_duration: Duration,
    /// Maximum duration
    pub max_duration: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Throughput (operations per second)
    pub throughput_ops: f64,
    /// Optional throughput in bytes/second
    pub throughput_bytes: Option<f64>,
}

impl BenchmarkResult {
    /// Format a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "{}: {:.2?}/iter ({} iters, {:.2?} total, {:.0} ops/s)",
            self.name,
            self.mean_duration,
            self.iterations,
            self.total_duration,
            self.throughput_ops,
        )
    }
}

/// Benchmark runner
pub struct BenchmarkRunner {
    warmup_iterations: u64,
    min_iterations: u64,
    max_iterations: u64,
    target_time: Duration,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner with default settings
    pub fn new() -> Self {
        Self {
            warmup_iterations: 10,
            min_iterations: 100,
            max_iterations: 10_000,
            target_time: Duration::from_secs(2),
        }
    }

    /// Set warmup iterations
    pub fn warmup(mut self, n: u64) -> Self {
        self.warmup_iterations = n;
        self
    }

    /// Set minimum iterations
    pub fn min_iters(mut self, n: u64) -> Self {
        self.min_iterations = n;
        self
    }

    /// Set maximum iterations
    pub fn max_iters(mut self, n: u64) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set target time
    pub fn target_time(mut self, d: Duration) -> Self {
        self.target_time = d;
        self
    }

    /// Run a benchmark
    pub fn bench<F>(&self, name: &str, mut f: F) -> BenchmarkResult
    where
        F: FnMut(),
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            f();
        }

        // Measure
        let mut durations = Vec::new();
        let global_start = Instant::now();

        for i in 0..self.max_iterations {
            let start = Instant::now();
            f();
            let elapsed = start.elapsed();
            durations.push(elapsed);

            if i >= self.min_iterations && global_start.elapsed() >= self.target_time {
                break;
            }
        }

        let iterations = durations.len() as u64;
        self.compute_result(name, &durations, iterations, None)
    }

    /// Run a benchmark with throughput measured in bytes
    pub fn bench_throughput<F>(
        &self,
        name: &str,
        bytes_per_iter: usize,
        mut f: F,
    ) -> BenchmarkResult
    where
        F: FnMut(),
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            f();
        }

        // Measure
        let mut durations = Vec::new();
        let global_start = Instant::now();

        for i in 0..self.max_iterations {
            let start = Instant::now();
            f();
            let elapsed = start.elapsed();
            durations.push(elapsed);

            if i >= self.min_iterations && global_start.elapsed() >= self.target_time {
                break;
            }
        }

        let iterations = durations.len() as u64;
        self.compute_result(name, &durations, iterations, Some(bytes_per_iter))
    }

    fn compute_result(
        &self,
        name: &str,
        durations: &[Duration],
        iterations: u64,
        bytes_per_iter: Option<usize>,
    ) -> BenchmarkResult {
        let total: Duration = durations.iter().sum();
        let mean = total / iterations as u32;

        let mut sorted: Vec<Duration> = durations.to_vec();
        sorted.sort();
        let median = sorted[sorted.len() / 2];
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        // Standard deviation
        let mean_nanos = mean.as_nanos() as f64;
        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / iterations as f64;
        let std_dev_nanos = variance.sqrt();
        let std_dev = Duration::from_nanos(std_dev_nanos as u64);

        let throughput_ops = if mean.as_nanos() > 0 {
            1_000_000_000.0 / mean_nanos
        } else {
            f64::INFINITY
        };

        let throughput_bytes = bytes_per_iter.map(|bpi| {
            throughput_ops * bpi as f64
        });

        BenchmarkResult {
            name: name.to_string(),
            iterations,
            total_duration: total,
            mean_duration: mean,
            median_duration: median,
            min_duration: min,
            max_duration: max,
            std_dev,
            throughput_ops,
            throughput_bytes,
        }
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark suite with named groups
pub struct BenchmarkSuite {
    name: String,
    results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    /// Create a new suite
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            results: Vec::new(),
        }
    }

    /// Add a result
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Get all results
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Get suite name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Print a formatted report
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("=== Benchmark Suite: {} ===", self.name));
        lines.push(String::new());

        let max_name_len = self.results.iter().map(|r| r.name.len()).max().unwrap_or(20);

        lines.push(format!(
            "{:<width$}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
            "Benchmark", "Mean", "Median", "Min", "Max", "Ops/s",
            width = max_name_len
        ));
        lines.push("-".repeat(max_name_len + 66));

        for r in &self.results {
            lines.push(format!(
                "{:<width$}  {:>12.2?}  {:>12.2?}  {:>12.2?}  {:>12.2?}  {:>12.0}",
                r.name,
                r.mean_duration,
                r.median_duration,
                r.min_duration,
                r.max_duration,
                r.throughput_ops,
                width = max_name_len
            ));
        }

        lines.push(String::new());
        lines.push(format!("Total benchmarks: {}", self.results.len()));
        lines.join("\n")
    }
}

/// Run the built-in benchmark suite for this crate
pub fn run_builtin_benchmarks() -> Result<BenchmarkSuite> {
    let runner = BenchmarkRunner::new()
        .warmup(5)
        .min_iters(50)
        .max_iters(1000)
        .target_time(Duration::from_millis(500));

    let mut suite = BenchmarkSuite::new("cuda-rust-wasm");

    // --- Memory allocation benchmarks ---
    suite.add_result(runner.bench("pool_allocate_1kb", || {
        let pool = crate::memory::MemoryPool::new();
        let buf = pool.allocate(1024);
        pool.deallocate(buf);
    }));

    suite.add_result(runner.bench("pool_allocate_64kb", || {
        let pool = crate::memory::MemoryPool::new();
        let buf = pool.allocate(65536);
        pool.deallocate(buf);
    }));

    suite.add_result(runner.bench_throughput("host_buffer_fill_1kb", 1024, || {
        let mut buf = crate::memory::HostBuffer::<u8>::new(1024).unwrap();
        buf.fill(0xFF);
    }));

    // --- Kernel launch benchmarks ---
    use crate::runtime::kernel::{KernelFunction, ThreadContext, LaunchConfig};
    use crate::runtime::grid::{Grid, Block};

    struct NoopKernel;
    impl KernelFunction<()> for NoopKernel {
        fn execute(&self, _: (), _ctx: ThreadContext) {}
        fn name(&self) -> &str { "noop" }
    }

    suite.add_result(runner.bench("kernel_launch_1x1", || {
        let _ = crate::runtime::kernel::launch_kernel(
            NoopKernel,
            LaunchConfig::new(Grid::new(1u32), Block::new(1u32)),
            (),
        );
    }));

    suite.add_result(runner.bench("kernel_launch_1x256", || {
        let _ = crate::runtime::kernel::launch_kernel(
            NoopKernel,
            LaunchConfig::new(Grid::new(1u32), Block::new(256u32)),
            (),
        );
    }));

    suite.add_result(runner.bench("kernel_launch_4x256", || {
        let _ = crate::runtime::kernel::launch_kernel(
            NoopKernel,
            LaunchConfig::new(Grid::new(4u32), Block::new(256u32)),
            (),
        );
    }));

    // --- Transpiler benchmarks ---
    let simple_cuda = r#"
        __global__ void add(float* a, float* b, float* c) {
            int i = threadIdx.x;
            c[i] = a[i] + b[i];
        }
    "#;

    suite.add_result(runner.bench("transpile_simple_kernel", || {
        let t = crate::transpiler::CudaTranspiler::new();
        let _ = t.transpile(simple_cuda, false, false);
    }));

    suite.add_result(runner.bench("transpile_with_optimization", || {
        let t = crate::transpiler::CudaTranspiler::new();
        let _ = t.transpile(simple_cuda, true, true);
    }));

    // --- Parser benchmarks ---
    suite.add_result(runner.bench("parse_simple_kernel", || {
        let p = crate::parser::CudaParser::new();
        let _ = p.parse(simple_cuda);
    }));

    // --- Half-precision benchmarks ---
    suite.add_result(runner.bench("half_f32_roundtrip_1000", || {
        for i in 0..1000 {
            let h = crate::runtime::half::Half::from_f32(i as f32);
            std::hint::black_box(h.to_f32());
        }
    }));

    suite.add_result(runner.bench("half_dot_product_256", || {
        let a: Vec<_> = (0..256).map(|i| crate::runtime::half::Half::from_f32(i as f32 * 0.01)).collect();
        let b: Vec<_> = (0..256).map(|i| crate::runtime::half::Half::from_f32(i as f32 * 0.01)).collect();
        std::hint::black_box(crate::runtime::half::half_dot(&a, &b));
    }));

    Ok(suite)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runner_basic() {
        let runner = BenchmarkRunner::new()
            .warmup(2)
            .min_iters(10)
            .max_iters(100)
            .target_time(Duration::from_millis(100));

        let mut counter = 0u64;
        let result = runner.bench("counter_increment", || {
            counter += 1;
        });

        assert!(result.iterations >= 10);
        assert!(result.throughput_ops > 0.0);
        assert!(result.mean_duration <= result.max_duration);
        assert!(result.min_duration <= result.mean_duration);
    }

    #[test]
    fn test_benchmark_throughput() {
        let runner = BenchmarkRunner::new()
            .warmup(1)
            .min_iters(10)
            .max_iters(50)
            .target_time(Duration::from_millis(50));

        let result = runner.bench_throughput("memcpy_1kb", 1024, || {
            let src = vec![0u8; 1024];
            std::hint::black_box(&src);
        });

        assert!(result.throughput_bytes.is_some());
        assert!(result.throughput_bytes.unwrap() > 0.0);
    }

    #[test]
    fn test_benchmark_suite() {
        let runner = BenchmarkRunner::new()
            .warmup(1)
            .min_iters(5)
            .max_iters(10)
            .target_time(Duration::from_millis(10));

        let mut suite = BenchmarkSuite::new("test_suite");
        suite.add_result(runner.bench("a", || {}));
        suite.add_result(runner.bench("b", || {}));

        assert_eq!(suite.results().len(), 2);
        assert_eq!(suite.name(), "test_suite");

        let report = suite.report();
        assert!(report.contains("test_suite"));
        assert!(report.contains("a"));
        assert!(report.contains("b"));
    }

    #[test]
    fn test_builtin_benchmarks() {
        let suite = run_builtin_benchmarks().unwrap();
        assert!(!suite.results().is_empty());
        // Verify each benchmark has meaningful results
        for r in suite.results() {
            assert!(r.iterations > 0, "Benchmark {} had 0 iterations", r.name);
            assert!(r.throughput_ops > 0.0, "Benchmark {} had 0 throughput", r.name);
        }
    }

    #[test]
    fn test_benchmark_result_summary() {
        let result = BenchmarkResult {
            name: "test".to_string(),
            iterations: 100,
            total_duration: Duration::from_millis(100),
            mean_duration: Duration::from_millis(1),
            median_duration: Duration::from_millis(1),
            min_duration: Duration::from_micros(500),
            max_duration: Duration::from_millis(2),
            std_dev: Duration::from_micros(200),
            throughput_ops: 1000.0,
            throughput_bytes: None,
        };
        let summary = result.summary();
        assert!(summary.contains("test"));
        assert!(summary.contains("100 iters"));
    }
}
