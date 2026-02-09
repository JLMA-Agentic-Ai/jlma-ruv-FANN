//! Nutanix GPU workload deployment example
//!
//! Demonstrates how to:
//! 1. Connect to Nutanix Prism Central
//! 2. Discover GPU-equipped hosts across clusters
//! 3. Generate Kubernetes deployment manifests for cuda-wasm
//! 4. Handle multi-vendor GPU selection (NVIDIA, AMD)
//!
//! # Running
//! ```bash
//! # With mock data (no Nutanix connection required)
//! cargo run --example deploy_gpu_workload
//!
//! # With real Nutanix connection
//! cargo run --example deploy_gpu_workload --features nutanix
//! ```
//!
//! # Environment Variables (for real connections)
//! - `NUTANIX_PRISM_URL`: Prism Central URL (e.g., "https://prism.example.com:9440")
//! - `NUTANIX_API_KEY`: API key for authentication
//! - `NUTANIX_USERNAME`: Username for basic auth (alternative to API key)
//! - `NUTANIX_PASSWORD`: Password for basic auth

use cuda_rust_wasm::nutanix::{
    NutanixClient, NutanixConfig, DeploymentConfig, GpuVendor,
    deployment::DeploymentGenerator,
};

fn get_config_from_env() -> NutanixConfig {
    let base_url = std::env::var("NUTANIX_PRISM_URL")
        .unwrap_or_else(|_| "https://prism-central.example.com:9440".to_string());

    let api_key = std::env::var("NUTANIX_API_KEY").unwrap_or_default();

    if api_key.is_empty() {
        let username = std::env::var("NUTANIX_USERNAME")
            .unwrap_or_else(|_| "admin".to_string());
        let password = std::env::var("NUTANIX_PASSWORD")
            .unwrap_or_else(|_| "".to_string());
        NutanixConfig::with_basic_auth(base_url, username, password)
            .with_insecure_ssl() // For lab environments
    } else {
        NutanixConfig::new(base_url, api_key)
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== cuda-wasm Nutanix GPU Workload Deployment ===\n");

    // Step 1: Connect to Prism Central
    let config = get_config_from_env();
    println!("Prism Central: {}", config.base_url);
    println!("API Version:   {}", config.api_version);
    println!();

    let client = NutanixClient::new(config)?;

    // Step 2: Discover GPU nodes
    println!("--- GPU Node Discovery ---\n");
    let gpu_nodes = client.discover_gpu_nodes().await?;

    println!("Found {} GPU-equipped hosts:\n", gpu_nodes.len());
    for node in &gpu_nodes {
        println!("  Host: {} ({})", node.host_name, node.ip_address);
        println!("    Cluster:    {}", node.cluster_name);
        println!("    Arch:       {}", node.capabilities.cpu_arch);
        println!("    CPU Cores:  {}", node.capabilities.cpu_cores);
        println!("    RAM:        {} GB", node.capabilities.ram_bytes / (1024 * 1024 * 1024));
        println!("    Hypervisor: {}", node.capabilities.hypervisor);
        println!("    GPUs (total/available): {}/{}",
            node.total_gpus.len(), node.available_gpus.len());

        for gpu in &node.available_gpus {
            println!("      - {} {} ({} GB, {})",
                gpu.vendor,
                gpu.model,
                gpu.memory_bytes / (1024 * 1024 * 1024),
                gpu.mode,
            );
        }
        println!();
    }

    // Step 3: Get cluster GPU summary
    println!("--- Cluster GPU Summary ---\n");
    let summary = client.get_cluster_gpu_summary(None).await?;

    println!("  Cluster:          {}", summary.cluster_name);
    println!("  GPU Hosts:        {}", summary.gpu_host_count);
    println!("  Total GPUs:       {}", summary.total_gpu_count);
    println!("  Available GPUs:   {}", summary.available_gpu_count);
    println!("  Total GPU Memory: {} GB", summary.total_gpu_memory_bytes / (1024 * 1024 * 1024));
    println!("  Multi-vendor:     {}", summary.is_multi_vendor());

    println!("\n  GPUs by vendor:");
    for (vendor, count) in &summary.gpus_by_vendor {
        println!("    {}: {}", vendor, count);
    }
    println!("\n  GPUs by model:");
    for (model, count) in &summary.gpus_by_model {
        println!("    {}: {}", model, count);
    }
    println!();

    // Step 4: Find best nodes for NVIDIA workloads
    println!("--- Node Selection ---\n");
    let nvidia_nodes = client.find_best_nodes(&GpuVendor::Nvidia, 1, false).await?;
    println!("Best nodes for NVIDIA workloads ({} candidates):", nvidia_nodes.len());
    for node in &nvidia_nodes {
        println!("  {} - {} available NVIDIA GPU(s)",
            node.host_name,
            node.available_gpu_count(&GpuVendor::Nvidia),
        );
    }
    println!();

    let amd_nodes = client.find_best_nodes(&GpuVendor::Amd, 1, false).await?;
    println!("Best nodes for AMD workloads ({} candidates):", amd_nodes.len());
    for node in &amd_nodes {
        println!("  {} - {} available AMD GPU(s)",
            node.host_name,
            node.available_gpu_count(&GpuVendor::Amd),
        );
    }
    println!();

    // Step 5: Generate NVIDIA deployment
    println!("--- Kubernetes Deployment (NVIDIA) ---\n");
    let nvidia_deploy_config = DeploymentConfig::new(
        "cuda-wasm-nvidia",
        "registry.example.com/cuda-wasm:latest",
    )
    .with_gpu_vendor(GpuVendor::Nvidia)
    .with_gpus(2)
    .with_hpa(1, 8, 70)
    .with_nke_annotation("nke.nutanix.com/gpu-driver", "nvidia-535")
    .with_nke_annotation("nke.nutanix.com/priority", "high");

    let nvidia_generator = DeploymentGenerator::new(nvidia_deploy_config);
    let nvidia_yaml = nvidia_generator.generate_all();
    println!("{}", nvidia_yaml);
    println!();

    // Step 6: Generate AMD deployment
    println!("--- Kubernetes Deployment (AMD) ---\n");
    let amd_deploy_config = DeploymentConfig::new(
        "cuda-wasm-amd",
        "registry.example.com/cuda-wasm:latest-rocm",
    )
    .with_gpu_vendor(GpuVendor::Amd)
    .with_gpus(1)
    .with_hpa(1, 4, 75);

    let amd_generator = DeploymentGenerator::new(amd_deploy_config);
    let amd_yaml = amd_generator.generate_all();
    println!("{}", amd_yaml);
    println!();

    // Step 7: Get host capabilities for a specific node
    println!("--- Host Capabilities ---\n");
    if let Some(first_node) = gpu_nodes.first() {
        let caps = client.get_host_capabilities(&first_node.host_id).await?;
        println!("Host: {} ({})", caps.host_name, caps.host_id);
        println!("  Architecture:    {}", caps.cpu_arch);
        println!("  CPU Cores:       {}", caps.cpu_cores);
        println!("  RAM:             {} GB", caps.ram_bytes / (1024 * 1024 * 1024));
        println!("  Hypervisor:      {}", caps.hypervisor);
        println!("  Has NVIDIA:      {}", caps.has_nvidia);
        println!("  Has AMD:         {}", caps.has_amd);
        println!("  Is ARM:          {}", caps.is_arm);
        println!("  GPU Passthrough: {}", caps.gpu_passthrough_supported);
        println!("  vGPU Support:    {}", caps.vgpu_supported);
        println!("  GPU Count:       {}", caps.gpus.len());
    }

    println!("\nDone.");
    Ok(())
}
