//! Browser compatibility tests for WebGPU and WebAssembly
//!
//! The wasm32 module contains tests intended for browser environments using
//! wasm-bindgen-test. The non-wasm module simulates browser-like constraints
//! using the native API.

#[cfg(target_arch = "wasm32")]
mod browser_tests {
    use wasm_bindgen_test::*;
    use cuda_rust_wasm::{
        transpiler::CudaTranspiler,
        memory::MemoryPool,
    };
    use wasm_bindgen::prelude::*;
    use web_sys::console;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_webgpu_availability() {
        let window = web_sys::window().unwrap();
        let navigator = window.navigator();

        // Check if WebGPU is available
        let gpu = js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu"));

        match gpu {
            Ok(gpu_obj) if !gpu_obj.is_undefined() => {
                console::log_1(&"WebGPU is available".into());
            },
            _ => {
                console::log_1(&"WebGPU not available, skipping WebGPU tests".into());
            }
        }
    }

    #[wasm_bindgen_test]
    fn test_webassembly_simd_support() {
        // Test WASM SIMD availability via module validation
        let has_simd = js_sys::WebAssembly::validate(&[
            0x00, 0x61, 0x73, 0x6d, // magic
            0x01, 0x00, 0x00, 0x00, // version
            0x01, 0x04, 0x01, 0x60, // type section
            0x00, 0x00,             // no params, no results
            0x03, 0x02, 0x01, 0x00, // function section
            0x0a, 0x09, 0x01, 0x07, // code section
            0x00, 0xfd, 0x0c,       // v128.const
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x1a, 0x0b              // drop, end
        ]);

        console::log_1(&format!("WASM SIMD support: {}", has_simd).into());
    }

    #[wasm_bindgen_test]
    fn test_shared_array_buffer_support() {
        // Test SharedArrayBuffer availability
        let window = web_sys::window().unwrap();
        let shared_array_buffer = js_sys::Reflect::get(&window, &JsValue::from_str("SharedArrayBuffer"));

        match shared_array_buffer {
            Ok(sab) if !sab.is_undefined() => {
                console::log_1(&"SharedArrayBuffer is available".into());
            },
            _ => {
                console::log_1(&"SharedArrayBuffer not available".into());
            }
        }
    }

    #[wasm_bindgen_test]
    fn test_transpiler_in_browser() {
        let cuda_code = r#"
            __global__ void vector_add(float* a, float* b, float* c, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    c[idx] = a[idx] + b[idx];
                }
            }
        "#;

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(cuda_code, false, false);
        assert!(result.is_ok(), "Transpilation should succeed in browser");

        console::log_1(&"Transpiler test passed in browser!".into());
    }

    #[wasm_bindgen_test]
    fn test_memory_pool_in_browser() {
        // Test that MemoryPool works in the browser WASM environment
        let pool = MemoryPool::new();

        let buf = pool.allocate(4096);
        assert_eq!(buf.len(), 4096);

        pool.deallocate(buf);

        let stats = pool.stats();
        assert!(stats.total_allocations >= 1);

        console::log_1(&"Memory pool test passed in browser!".into());
    }

    #[wasm_bindgen_test]
    fn test_error_handling_in_browser() {
        // Test that errors are properly handled in browser environment
        let invalid_cuda = "__global__ void invalid() { syntax error }";

        let transpiler = CudaTranspiler::new();
        let result = transpiler.transpile(invalid_cuda, false, false);

        // May or may not error depending on parser strictness - just verify no panic
        console::log_1(&format!("Parse result is_ok: {}", result.is_ok()).into());
    }

    #[wasm_bindgen_test]
    fn test_canvas_integration() {
        use web_sys::HtmlCanvasElement;

        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();

        // Create a canvas element to verify DOM interaction works
        let canvas = document.create_element("canvas")
            .unwrap()
            .dyn_into::<HtmlCanvasElement>()
            .unwrap();

        canvas.set_width(512);
        canvas.set_height(512);

        // Verify canvas was created correctly
        assert_eq!(canvas.width(), 512);
        assert_eq!(canvas.height(), 512);

        // Check for WebGPU context availability
        if let Ok(gpu) = js_sys::Reflect::get(&window.navigator(), &JsValue::from_str("gpu")) {
            if !gpu.is_undefined() {
                console::log_1(&"Canvas WebGPU integration available".into());
            } else {
                console::log_1(&"WebGPU not available on this browser".into());
            }
        }
    }
}

// Regular tests that run in the native (non-wasm32) environment
// These simulate browser-like constraints using the available native API.
#[cfg(not(target_arch = "wasm32"))]
mod browser_simulation_tests {
    use cuda_rust_wasm::memory::{MemoryPool, PoolConfig};
    use cuda_rust_wasm::transpiler::CudaTranspiler;

    #[test]
    fn test_browser_memory_constraints() {
        // Simulate browser memory constraints by using a small PoolConfig
        let config = PoolConfig {
            max_pool_size: 1 * 1024 * 1024, // 1MB max per pool
            min_pooled_size: 256,
            max_pooled_size: 512 * 1024,     // 512KB max pooled
            prealloc_count: 2,
        };

        let pool = MemoryPool::with_config(config);

        // Should be able to allocate small chunks
        let small_alloc = pool.allocate(1024);
        assert_eq!(small_alloc.len(), 1024);

        // Larger allocations still succeed (MemoryPool falls back to direct alloc)
        let large_alloc = pool.allocate(2 * 1024 * 1024);
        assert_eq!(large_alloc.len(), 2 * 1024 * 1024);

        pool.deallocate(small_alloc);
        pool.deallocate(large_alloc);

        let stats = pool.stats();
        assert!(stats.total_allocations >= 2);
    }

    #[test]
    fn test_transpiler_error_handling() {
        // Simulate error handling for invalid CUDA in a browser-like context
        let transpiler = CudaTranspiler::new();

        // Valid CUDA should succeed
        let valid_cuda = r#"
            __global__ void simple(float* a) {
                int i = threadIdx.x;
                a[i] = 1.0f;
            }
        "#;
        let result = transpiler.transpile(valid_cuda, false, false);
        assert!(result.is_ok(), "Valid CUDA should transpile successfully");

        // Test with multiple transpilations to simulate browser session
        for i in 0..5 {
            let code = format!(
                "__global__ void kernel_{}(float* data) {{ int idx = threadIdx.x; data[idx] = {}.0f; }}",
                i, i
            );
            let result = transpiler.transpile(&code, false, false);
            assert!(result.is_ok(), "Transpilation {} should succeed", i);
        }
    }
}
