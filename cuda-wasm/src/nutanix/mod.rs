//! Nutanix platform integration for cuda-wasm
//!
//! This module provides integration with Nutanix infrastructure for deploying
//! GPU-accelerated cuda-wasm workloads on Nutanix clusters. It includes:
//!
//! - **Discovery**: GPU resource discovery via Nutanix Prism Central API
//! - **Deployment**: Kubernetes/NKE deployment manifest generation
//! - **Config**: Configuration types for Nutanix connection and workload settings
//! - **vGPU Scheduler**: GPU partitioning and multi-tenant workload scheduling
//! - **Monitoring**: GPU telemetry, health assessment, and capacity forecasting
//! - **NC2**: Nutanix Cloud Clusters integration for hybrid/multi-cloud GPU workloads

pub mod config;
pub mod discovery;
pub mod deployment;
pub mod vgpu_scheduler;
pub mod monitoring;
pub mod nc2;

pub use config::{
    NutanixConfig, DeploymentConfig, GpuNode, GpuInfo, HostCapabilities, GpuClusterSummary,
    GpuVendor, GpuModel,
};
pub use discovery::NutanixClient;
pub use deployment::DeploymentGenerator;
pub use vgpu_scheduler::{
    VgpuScheduler, VgpuProfile, SchedulingPolicy, WorkloadRequest,
    ScheduleResult, MigrationPlan,
};
pub use monitoring::{
    GpuMonitor, GpuMetrics, NodeHealth, HealthStatus, Alert, AlertSeverity,
    CapacityForecast,
};
pub use nc2::{
    Nc2Client, CloudProvider, Nc2Cluster, ClusterStatus, WorkloadPlacement,
    CostEstimate, MigrationStatus,
};
