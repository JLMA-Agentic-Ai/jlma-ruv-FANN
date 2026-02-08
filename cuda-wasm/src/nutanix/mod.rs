//! Nutanix platform integration for cuda-wasm
//!
//! This module provides integration with Nutanix infrastructure for deploying
//! GPU-accelerated cuda-wasm workloads on Nutanix clusters. It includes:
//!
//! - **Discovery**: GPU resource discovery via Nutanix Prism Central API
//! - **Deployment**: Kubernetes/NKE deployment manifest generation
//! - **Config**: Configuration types for Nutanix connection and workload settings

pub mod config;
pub mod discovery;
pub mod deployment;

pub use config::{
    NutanixConfig, DeploymentConfig, GpuNode, GpuInfo, HostCapabilities, GpuClusterSummary,
    GpuVendor, GpuModel,
};
pub use discovery::NutanixClient;
pub use deployment::DeploymentGenerator;
