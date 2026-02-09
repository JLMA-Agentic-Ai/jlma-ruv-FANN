//! Vector addition example demonstrating CUDA-Rust-WASM capabilities

use cuda_rust_wasm::prelude::*;
use cuda_rust_wasm::Result;
use std::sync::{Arc, Mutex};

/// Vector addition kernel implemented as a KernelFunction
struct VectorAddKernel {
    a: Vec<f32>,
    b: Vec<f32>,
    c: Arc<Mutex<Vec<f32>>>,
}

impl KernelFunction<()> for VectorAddKernel {
    fn execute(&self, _args: (), ctx: ThreadContext) {
        let tid = ctx.global_thread_id();
        if tid < self.a.len() {
            let mut c = self.c.lock().unwrap();
            c[tid] = self.a[tid] + self.b[tid];
        }
    }

    fn name(&self) -> &str {
        "vector_add"
    }
}

fn main() -> Result<()> {
    println!("=== CUDA-Rust-WASM Vector Addition Example ===\n");

    // Problem size
    let n = 1024;
    println!("Vector size: {}", n);

    // Allocate and initialize host memory
    let h_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let h_b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();

    println!("\nFirst 10 elements:");
    println!("a: {:?}", &h_a[..10]);
    println!("b: {:?}", &h_b[..10]);

    // Output buffer
    let h_c = Arc::new(Mutex::new(vec![0.0f32; n]));

    // Launch kernel
    let block_size = 256;
    let grid_size = ((n + block_size - 1) / block_size) as u32;

    println!("\nLaunching kernel with:");
    println!("  Grid size: {}", grid_size);
    println!("  Block size: {}", block_size);

    let kernel = VectorAddKernel {
        a: h_a.clone(),
        b: h_b.clone(),
        c: h_c.clone(),
    };

    let config = LaunchConfig::new(
        Grid::new(grid_size),
        Block::new(block_size as u32),
    );

    launch_kernel(kernel, config, ())?;
    println!("\nKernel execution completed");

    // Verify results
    let result = h_c.lock().unwrap();
    println!("\nFirst 10 results:");
    println!("c = a + b: {:?}", &result[..10]);

    // Check correctness
    let mut correct = true;
    for i in 0..n {
        let expected = h_a[i] + h_b[i];
        if (result[i] - expected).abs() > 1e-5 {
            println!("Error at index {}: {} != {}", i, result[i], expected);
            correct = false;
            break;
        }
    }

    if correct {
        println!("\nVector addition completed successfully!");
    } else {
        println!("\nVector addition failed verification!");
    }

    // Print device info
    let device = Device::get_default()?;
    let props = device.properties();
    println!("\nDevice properties:");
    println!("  Name: {}", props.name);
    println!("  Backend: {:?}", device.backend());
    println!("  Total memory: {} MB", props.total_memory / (1024 * 1024));
    println!("  Max threads per block: {}", props.max_threads_per_block);
    println!("  Compute capability: {}.{}", props.compute_capability.0, props.compute_capability.1);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_addition() {
        let n = 100;
        let h_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let h_b: Vec<f32> = (0..n).map(|i| i as f32 * 2.0).collect();
        let h_c = Arc::new(Mutex::new(vec![0.0f32; n]));

        let kernel = VectorAddKernel {
            a: h_a.clone(),
            b: h_b.clone(),
            c: h_c.clone(),
        };

        let config = LaunchConfig::new(Grid::new(1u32), Block::new(128u32));
        launch_kernel(kernel, config, ()).unwrap();

        let result = h_c.lock().unwrap();
        for i in 0..n {
            assert_eq!(result[i], h_a[i] + h_b[i]);
        }
    }
}
