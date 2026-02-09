//! CUDA fidelity test suite entry point
//!
//! This file declares all submodules in the cuda_fidelity/ directory.

#[path = "cuda_fidelity/parser_fidelity_tests.rs"]
mod parser_fidelity_tests;

#[path = "cuda_fidelity/transpiler_fidelity_tests.rs"]
mod transpiler_fidelity_tests;

#[path = "cuda_fidelity/warp_tests.rs"]
mod warp_tests;

#[path = "cuda_fidelity/atomic_tests.rs"]
mod atomic_tests;

#[path = "cuda_fidelity/memory_tests_extended.rs"]
mod memory_tests_extended;
