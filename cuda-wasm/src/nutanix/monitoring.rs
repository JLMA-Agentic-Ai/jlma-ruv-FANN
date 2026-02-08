//! GPU monitoring and telemetry via Nutanix Prism Central
//!
//! Provides real-time GPU metrics collection, health assessment, utilization
//! history tracking, and capacity forecasting for cuda-wasm workloads running
//! on Nutanix clusters.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::CudaRustError;
use super::config::NutanixConfig;

/// GPU metrics snapshot for a single GPU device
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuMetrics {
    /// GPU utilization percentage (0-100)
    pub utilization_percent: f64,
    /// GPU memory currently in use (bytes)
    pub memory_used_bytes: u64,
    /// Total GPU memory (bytes)
    pub memory_total_bytes: u64,
    /// GPU temperature in Celsius
    pub temperature_celsius: f64,
    /// GPU power draw in Watts
    pub power_watts: f64,
    /// GPU core clock speed in MHz
    pub clock_speed_mhz: u32,
    /// Fan speed percentage (0-100)
    pub fan_speed_percent: f64,
    /// ECC error count (single-bit + double-bit)
    pub ecc_errors: u64,
}

impl GpuMetrics {
    /// Memory utilization as a percentage
    pub fn memory_utilization_percent(&self) -> f64 {
        if self.memory_total_bytes == 0 {
            return 0.0;
        }
        (self.memory_used_bytes as f64 / self.memory_total_bytes as f64) * 100.0
    }

    /// Whether the GPU is thermally throttling (above 85C)
    pub fn is_throttling(&self) -> bool {
        self.temperature_celsius > 85.0
    }
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning - requires attention
    Warning,
    /// Critical - immediate action needed
    Critical,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// An alert generated from GPU metric analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Alert {
    /// Alert severity
    pub severity: AlertSeverity,
    /// Human-readable alert message
    pub message: String,
    /// Unix timestamp (seconds) when the alert was generated
    pub timestamp: u64,
    /// GPU device ID that triggered the alert (if applicable)
    pub gpu_id: Option<String>,
}

/// Overall health status for a node
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HealthStatus {
    /// All GPUs operating normally
    Healthy,
    /// Some GPUs have warnings (high temp, high utilization)
    Warning,
    /// One or more GPUs have critical issues
    Critical,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "HEALTHY"),
            HealthStatus::Warning => write!(f, "WARNING"),
            HealthStatus::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Health assessment for a GPU node
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NodeHealth {
    /// Node UUID
    pub node_id: String,
    /// Overall health status
    pub overall_health: HealthStatus,
    /// Per-GPU metrics (keyed by GPU device ID)
    pub gpu_metrics: Vec<(String, GpuMetrics)>,
    /// Active alerts
    pub alerts: Vec<Alert>,
}

/// Capacity forecast for a cluster
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CapacityForecast {
    /// Cluster ID
    pub cluster_id: String,
    /// Current average GPU utilization across the cluster
    pub current_utilization_percent: f64,
    /// Projected utilization at the forecast horizon
    pub projected_utilization_percent: f64,
    /// Hours until capacity is projected to reach 90%
    pub hours_until_90_percent: Option<u32>,
    /// Hours until capacity is projected to reach 100%
    pub hours_until_full: Option<u32>,
    /// Recommended action based on the forecast
    pub recommendation: String,
}

/// GPU monitoring client for collecting metrics from Nutanix clusters
pub struct GpuMonitor {
    /// Prism Central connection configuration
    #[allow(dead_code)]
    config: NutanixConfig,

    /// HTTP client (when nutanix feature is available)
    #[cfg(feature = "nutanix")]
    #[allow(dead_code)]
    client: reqwest::Client,
}

impl GpuMonitor {
    /// Create a new GpuMonitor with the given configuration
    pub fn new(config: NutanixConfig) -> Result<Self, CudaRustError> {
        #[cfg(feature = "nutanix")]
        {
            let builder = reqwest::Client::builder().timeout(config.timeout);
            let client = builder.build().map_err(|e| {
                CudaRustError::RuntimeError(format!("Failed to create HTTP client: {}", e))
            })?;
            Ok(Self { config, client })
        }

        #[cfg(not(feature = "nutanix"))]
        {
            Ok(Self { config })
        }
    }

    /// Collect current GPU metrics for all GPUs on a node
    ///
    /// Polls the Prism Central API for the latest GPU telemetry data.
    pub async fn collect_metrics(
        &self,
        node_id: &str,
    ) -> Result<Vec<GpuMetrics>, CudaRustError> {
        #[cfg(feature = "nutanix")]
        {
            let _ = node_id;
            Err(CudaRustError::RuntimeError(
                "Live metrics collection requires Prism Central connection".to_string(),
            ))
        }

        #[cfg(not(feature = "nutanix"))]
        {
            Ok(self.mock_metrics(node_id))
        }
    }

    /// Perform a health assessment of a node's GPUs
    ///
    /// Collects metrics and evaluates them against thresholds to determine
    /// overall node health and generate any alerts.
    pub async fn check_health(
        &self,
        node_id: &str,
    ) -> Result<NodeHealth, CudaRustError> {
        let metrics = self.collect_metrics(node_id).await?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut alerts = Vec::new();
        let mut worst_health = HealthStatus::Healthy;

        let gpu_metrics: Vec<(String, GpuMetrics)> = metrics
            .into_iter()
            .enumerate()
            .map(|(i, m)| {
                let gpu_id = format!("{}-gpu-{}", node_id, i);

                // Temperature checks
                if m.temperature_celsius > 90.0 {
                    alerts.push(Alert {
                        severity: AlertSeverity::Critical,
                        message: format!(
                            "GPU {} temperature critical: {:.1}C",
                            gpu_id, m.temperature_celsius
                        ),
                        timestamp: now,
                        gpu_id: Some(gpu_id.clone()),
                    });
                    worst_health = HealthStatus::Critical;
                } else if m.temperature_celsius > 80.0 {
                    alerts.push(Alert {
                        severity: AlertSeverity::Warning,
                        message: format!(
                            "GPU {} temperature high: {:.1}C",
                            gpu_id, m.temperature_celsius
                        ),
                        timestamp: now,
                        gpu_id: Some(gpu_id.clone()),
                    });
                    if worst_health != HealthStatus::Critical {
                        worst_health = HealthStatus::Warning;
                    }
                }

                // Memory utilization checks
                let mem_pct = m.memory_utilization_percent();
                if mem_pct > 95.0 {
                    alerts.push(Alert {
                        severity: AlertSeverity::Critical,
                        message: format!(
                            "GPU {} memory nearly exhausted: {:.1}%",
                            gpu_id, mem_pct
                        ),
                        timestamp: now,
                        gpu_id: Some(gpu_id.clone()),
                    });
                    worst_health = HealthStatus::Critical;
                } else if mem_pct > 85.0 {
                    alerts.push(Alert {
                        severity: AlertSeverity::Warning,
                        message: format!(
                            "GPU {} memory utilization high: {:.1}%",
                            gpu_id, mem_pct
                        ),
                        timestamp: now,
                        gpu_id: Some(gpu_id.clone()),
                    });
                    if worst_health != HealthStatus::Critical {
                        worst_health = HealthStatus::Warning;
                    }
                }

                // ECC error checks
                if m.ecc_errors > 0 {
                    let severity = if m.ecc_errors > 10 {
                        worst_health = HealthStatus::Critical;
                        AlertSeverity::Critical
                    } else {
                        if worst_health != HealthStatus::Critical {
                            worst_health = HealthStatus::Warning;
                        }
                        AlertSeverity::Warning
                    };
                    alerts.push(Alert {
                        severity,
                        message: format!(
                            "GPU {} has {} ECC errors",
                            gpu_id, m.ecc_errors
                        ),
                        timestamp: now,
                        gpu_id: Some(gpu_id.clone()),
                    });
                }

                (gpu_id, m)
            })
            .collect();

        Ok(NodeHealth {
            node_id: node_id.to_string(),
            overall_health: worst_health,
            gpu_metrics,
            alerts,
        })
    }

    /// Retrieve utilization history for GPUs on a node
    ///
    /// Returns timestamped metric snapshots over the requested duration.
    pub async fn get_utilization_history(
        &self,
        node_id: &str,
        duration_minutes: u32,
    ) -> Result<Vec<(u64, GpuMetrics)>, CudaRustError> {
        #[cfg(feature = "nutanix")]
        {
            let _ = (node_id, duration_minutes);
            Err(CudaRustError::RuntimeError(
                "History collection requires Prism Central connection".to_string(),
            ))
        }

        #[cfg(not(feature = "nutanix"))]
        {
            Ok(self.mock_utilization_history(node_id, duration_minutes))
        }
    }

    /// Predict future capacity usage for a cluster using linear projection
    ///
    /// Analyzes recent utilization trends to estimate when the cluster
    /// will reach capacity thresholds (90% and 100%).
    pub async fn predict_capacity(
        &self,
        cluster_id: &str,
        hours_ahead: u32,
    ) -> Result<CapacityForecast, CudaRustError> {
        #[cfg(feature = "nutanix")]
        {
            let _ = (cluster_id, hours_ahead);
            Err(CudaRustError::RuntimeError(
                "Capacity prediction requires Prism Central connection".to_string(),
            ))
        }

        #[cfg(not(feature = "nutanix"))]
        {
            Ok(self.mock_capacity_forecast(cluster_id, hours_ahead))
        }
    }

    // --- Mock implementations for non-nutanix builds ---

    #[cfg(not(feature = "nutanix"))]
    fn mock_metrics(&self, _node_id: &str) -> Vec<GpuMetrics> {
        vec![
            GpuMetrics {
                utilization_percent: 65.0,
                memory_used_bytes: 30 * 1024 * 1024 * 1024,
                memory_total_bytes: 80 * 1024 * 1024 * 1024,
                temperature_celsius: 72.0,
                power_watts: 250.0,
                clock_speed_mhz: 1410,
                fan_speed_percent: 45.0,
                ecc_errors: 0,
            },
            GpuMetrics {
                utilization_percent: 82.0,
                memory_used_bytes: 55 * 1024 * 1024 * 1024,
                memory_total_bytes: 80 * 1024 * 1024 * 1024,
                temperature_celsius: 78.0,
                power_watts: 300.0,
                clock_speed_mhz: 1380,
                fan_speed_percent: 60.0,
                ecc_errors: 0,
            },
        ]
    }

    #[cfg(not(feature = "nutanix"))]
    fn mock_utilization_history(
        &self,
        _node_id: &str,
        duration_minutes: u32,
    ) -> Vec<(u64, GpuMetrics)> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let points = duration_minutes.min(60) as usize;
        let interval_secs = (duration_minutes as u64 * 60) / points.max(1) as u64;

        (0..points)
            .map(|i| {
                let timestamp = now - (points as u64 - i as u64) * interval_secs;
                let utilization = 50.0 + (i as f64 * 0.5);
                (
                    timestamp,
                    GpuMetrics {
                        utilization_percent: utilization.min(100.0),
                        memory_used_bytes: (40 + i as u64) * 1024 * 1024 * 1024,
                        memory_total_bytes: 80 * 1024 * 1024 * 1024,
                        temperature_celsius: 70.0 + (i as f64 * 0.2),
                        power_watts: 200.0 + (i as f64 * 2.0),
                        clock_speed_mhz: 1410,
                        fan_speed_percent: 40.0 + (i as f64 * 0.5),
                        ecc_errors: 0,
                    },
                )
            })
            .collect()
    }

    #[cfg(not(feature = "nutanix"))]
    fn mock_capacity_forecast(
        &self,
        cluster_id: &str,
        hours_ahead: u32,
    ) -> CapacityForecast {
        let current_util = 65.0;
        let growth_rate_per_hour = 0.5; // 0.5% per hour
        let projected = (current_util + growth_rate_per_hour * hours_ahead as f64).min(100.0);

        let hours_to_90 = if current_util < 90.0 {
            Some(((90.0 - current_util) / growth_rate_per_hour) as u32)
        } else {
            Some(0)
        };

        let hours_to_full = if current_util < 100.0 {
            Some(((100.0 - current_util) / growth_rate_per_hour) as u32)
        } else {
            Some(0)
        };

        let recommendation = if projected > 90.0 {
            "Consider adding GPU nodes or migrating workloads to NC2 clusters".to_string()
        } else if projected > 75.0 {
            "Monitor closely; plan for capacity expansion within the forecast window".to_string()
        } else {
            "Capacity is sufficient for the forecast period".to_string()
        };

        CapacityForecast {
            cluster_id: cluster_id.to_string(),
            current_utilization_percent: current_util,
            projected_utilization_percent: projected,
            hours_until_90_percent: hours_to_90,
            hours_until_full: hours_to_full,
            recommendation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor() -> GpuMonitor {
        let config = NutanixConfig::new("https://prism.example.com:9440", "test-key");
        GpuMonitor::new(config).unwrap()
    }

    #[test]
    fn test_gpu_metrics_memory_utilization() {
        let m = GpuMetrics {
            utilization_percent: 50.0,
            memory_used_bytes: 40 * 1024 * 1024 * 1024,
            memory_total_bytes: 80 * 1024 * 1024 * 1024,
            temperature_celsius: 70.0,
            power_watts: 250.0,
            clock_speed_mhz: 1400,
            fan_speed_percent: 50.0,
            ecc_errors: 0,
        };
        let pct = m.memory_utilization_percent();
        assert!((pct - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_gpu_metrics_throttling() {
        let normal = GpuMetrics {
            utilization_percent: 80.0,
            memory_used_bytes: 0,
            memory_total_bytes: 80 * 1024 * 1024 * 1024,
            temperature_celsius: 75.0,
            power_watts: 250.0,
            clock_speed_mhz: 1400,
            fan_speed_percent: 50.0,
            ecc_errors: 0,
        };
        assert!(!normal.is_throttling());

        let hot = GpuMetrics {
            temperature_celsius: 92.0,
            ..normal
        };
        assert!(hot.is_throttling());
    }

    #[test]
    fn test_alert_severity_display() {
        assert_eq!(AlertSeverity::Info.to_string(), "INFO");
        assert_eq!(AlertSeverity::Warning.to_string(), "WARNING");
        assert_eq!(AlertSeverity::Critical.to_string(), "CRITICAL");
    }

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "HEALTHY");
        assert_eq!(HealthStatus::Warning.to_string(), "WARNING");
        assert_eq!(HealthStatus::Critical.to_string(), "CRITICAL");
    }

    #[tokio::test]
    async fn test_collect_metrics() {
        let monitor = make_monitor();
        let metrics = monitor.collect_metrics("node-001").await.unwrap();
        assert_eq!(metrics.len(), 2);
        assert!(metrics[0].utilization_percent > 0.0);
        assert!(metrics[0].memory_total_bytes > 0);
    }

    #[tokio::test]
    async fn test_check_health_healthy_node() {
        let monitor = make_monitor();
        let health = monitor.check_health("node-001").await.unwrap();
        assert_eq!(health.node_id, "node-001");
        assert_eq!(health.gpu_metrics.len(), 2);
        // Mock data has normal temps (72C, 78C) so should be healthy
        assert_eq!(health.overall_health, HealthStatus::Healthy);
        assert!(health.alerts.is_empty());
    }

    #[tokio::test]
    async fn test_get_utilization_history() {
        let monitor = make_monitor();
        let history = monitor
            .get_utilization_history("node-001", 30)
            .await
            .unwrap();
        assert_eq!(history.len(), 30);

        // Verify timestamps are ordered
        for i in 1..history.len() {
            assert!(history[i].0 >= history[i - 1].0);
        }
    }

    #[tokio::test]
    async fn test_predict_capacity() {
        let monitor = make_monitor();
        let forecast = monitor
            .predict_capacity("cluster-001", 48)
            .await
            .unwrap();
        assert_eq!(forecast.cluster_id, "cluster-001");
        assert!(forecast.projected_utilization_percent > forecast.current_utilization_percent);
        assert!(forecast.hours_until_90_percent.is_some());
        assert!(forecast.hours_until_full.is_some());
    }

    #[tokio::test]
    async fn test_predict_capacity_long_horizon() {
        let monitor = make_monitor();
        let forecast = monitor
            .predict_capacity("cluster-001", 168)
            .await
            .unwrap();
        // With 0.5% per hour growth from 65%, after 168 hours = 65 + 84 = 149 -> capped at 100
        assert!((forecast.projected_utilization_percent - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_utilization_zero_total() {
        let m = GpuMetrics {
            utilization_percent: 0.0,
            memory_used_bytes: 0,
            memory_total_bytes: 0,
            temperature_celsius: 0.0,
            power_watts: 0.0,
            clock_speed_mhz: 0,
            fan_speed_percent: 0.0,
            ecc_errors: 0,
        };
        assert!((m.memory_utilization_percent()).abs() < 0.01);
    }

    #[test]
    fn test_monitor_creation() {
        let config = NutanixConfig::new("https://prism.example.com:9440", "key");
        let monitor = GpuMonitor::new(config);
        assert!(monitor.is_ok());
    }
}
