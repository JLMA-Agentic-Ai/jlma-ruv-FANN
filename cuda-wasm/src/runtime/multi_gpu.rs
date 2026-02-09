//! Multi-GPU support for device enumeration and peer-to-peer operations
//!
//! Provides multi-device management, peer-to-peer memory access,
//! and workload distribution across multiple GPU devices.

use crate::{Result, runtime_error};
use crate::runtime::device::{Device, DeviceProperties, BackendType};
use std::sync::Arc;

/// Multi-GPU context for managing multiple devices
pub struct MultiGpuContext {
    /// Available devices
    devices: Vec<Arc<Device>>,
    /// Active device index
    active_device: usize,
    /// Peer access matrix (devices[i] can access devices[j])
    peer_access: Vec<Vec<bool>>,
}

impl MultiGpuContext {
    /// Create a multi-GPU context by enumerating all available devices
    pub fn new() -> Result<Self> {
        let mut devices = Vec::new();

        // Get the default device (always available)
        let default_device = Device::get_default()?;
        devices.push(default_device);

        // Probe for additional devices based on backend
        // In a real implementation, this would enumerate CUDA devices
        // via cuDeviceGetCount or hipGetDeviceCount
        let additional = Self::probe_additional_devices();
        devices.extend(additional);

        let device_count = devices.len();
        let peer_access = vec![vec![false; device_count]; device_count];

        let mut ctx = Self {
            devices,
            active_device: 0,
            peer_access,
        };

        // Enable peer access where supported (same backend type)
        ctx.setup_peer_access();
        Ok(ctx)
    }

    /// Get number of available devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get a device by index
    pub fn device(&self, index: usize) -> Result<&Arc<Device>> {
        self.devices.get(index).ok_or_else(|| {
            runtime_error!("Device index {} out of range (have {})", index, self.devices.len())
        })
    }

    /// Get the active device
    pub fn active_device(&self) -> &Arc<Device> {
        &self.devices[self.active_device]
    }

    /// Get the active device index
    pub fn active_device_index(&self) -> usize {
        self.active_device
    }

    /// Set the active device
    pub fn set_device(&mut self, index: usize) -> Result<()> {
        if index >= self.devices.len() {
            return Err(runtime_error!(
                "Device index {} out of range (have {})",
                index, self.devices.len()
            ));
        }
        self.active_device = index;
        Ok(())
    }

    /// Check if peer access is enabled between two devices
    pub fn can_access_peer(&self, src: usize, dst: usize) -> Result<bool> {
        if src >= self.devices.len() || dst >= self.devices.len() {
            return Err(runtime_error!("Device index out of range"));
        }
        Ok(self.peer_access[src][dst])
    }

    /// Enable peer access between two devices
    pub fn enable_peer_access(&mut self, src: usize, dst: usize) -> Result<()> {
        if src >= self.devices.len() || dst >= self.devices.len() {
            return Err(runtime_error!("Device index out of range"));
        }
        if src == dst {
            return Ok(()); // Self-access is always allowed
        }

        // Check backend compatibility
        let src_backend = self.devices[src].backend();
        let dst_backend = self.devices[dst].backend();
        if src_backend != dst_backend {
            return Err(runtime_error!(
                "Cannot enable peer access between different backends ({:?} and {:?})",
                src_backend, dst_backend
            ));
        }

        self.peer_access[src][dst] = true;
        self.peer_access[dst][src] = true;
        Ok(())
    }

    /// Disable peer access between two devices
    pub fn disable_peer_access(&mut self, src: usize, dst: usize) -> Result<()> {
        if src >= self.devices.len() || dst >= self.devices.len() {
            return Err(runtime_error!("Device index out of range"));
        }
        self.peer_access[src][dst] = false;
        self.peer_access[dst][src] = false;
        Ok(())
    }

    /// Get properties for all devices
    pub fn all_properties(&self) -> Vec<&DeviceProperties> {
        self.devices.iter().map(|d| d.properties()).collect()
    }

    /// Distribute a 1D range across all devices (simple round-robin)
    pub fn distribute_range(&self, total: usize) -> Vec<DeviceRange> {
        let n = self.devices.len();
        let chunk = total / n;
        let remainder = total % n;

        let mut ranges = Vec::with_capacity(n);
        let mut offset = 0;

        for i in 0..n {
            let len = chunk + if i < remainder { 1 } else { 0 };
            ranges.push(DeviceRange {
                device_index: i,
                offset,
                length: len,
            });
            offset += len;
        }

        ranges
    }

    /// Probe for additional GPU devices beyond the default
    fn probe_additional_devices() -> Vec<Arc<Device>> {
        // Probe nvidia-smi for multi-GPU systems
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=count", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Ok(count) = stdout.trim().parse::<usize>() {
                    if count > 1 {
                        // Return additional virtual devices
                        let mut additional = Vec::new();
                        for id in 1..count {
                            if let Ok(dev) = Device::get_by_id(id) {
                                additional.push(dev);
                            }
                        }
                        return additional;
                    }
                }
            }
        }
        Vec::new()
    }

    /// Setup peer access based on backend compatibility
    fn setup_peer_access(&mut self) {
        let n = self.devices.len();
        for i in 0..n {
            self.peer_access[i][i] = true; // Self-access always allowed
            for j in (i + 1)..n {
                let same_backend = self.devices[i].backend() == self.devices[j].backend();
                if same_backend {
                    self.peer_access[i][j] = true;
                    self.peer_access[j][i] = true;
                }
            }
        }
    }
}

impl Default for MultiGpuContext {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback: single device with no peer access
            Self {
                devices: vec![Device::get_default().expect("default device should be available")],
                active_device: 0,
                peer_access: vec![vec![true]],
            }
        })
    }
}

/// Describes a range of work assigned to a device
#[derive(Debug, Clone)]
pub struct DeviceRange {
    pub device_index: usize,
    pub offset: usize,
    pub length: usize,
}

/// Peer-to-peer memory copy placeholder
pub fn memcpy_peer(
    _dst_device: usize,
    _src_device: usize,
    _size: usize,
) -> Result<()> {
    // In CPU emulation mode, all memory is shared, so peer copy is a no-op
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_gpu_creation() {
        let ctx = MultiGpuContext::new().unwrap();
        assert!(ctx.device_count() >= 1);
        assert_eq!(ctx.active_device_index(), 0);
    }

    #[test]
    fn test_device_access() {
        let ctx = MultiGpuContext::new().unwrap();
        let dev = ctx.device(0);
        assert!(dev.is_ok());
    }

    #[test]
    fn test_device_out_of_range() {
        let ctx = MultiGpuContext::new().unwrap();
        let result = ctx.device(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_active_device() {
        let mut ctx = MultiGpuContext::new().unwrap();
        assert!(ctx.set_device(0).is_ok());
        assert_eq!(ctx.active_device_index(), 0);
    }

    #[test]
    fn test_set_device_out_of_range() {
        let mut ctx = MultiGpuContext::new().unwrap();
        assert!(ctx.set_device(999).is_err());
    }

    #[test]
    fn test_self_peer_access() {
        let ctx = MultiGpuContext::new().unwrap();
        assert!(ctx.can_access_peer(0, 0).unwrap());
    }

    #[test]
    fn test_distribute_range() {
        let ctx = MultiGpuContext::new().unwrap();
        let ranges = ctx.distribute_range(100);
        assert!(!ranges.is_empty());

        // Total length should sum to 100
        let total: usize = ranges.iter().map(|r| r.length).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_distribute_range_uneven() {
        let ctx = MultiGpuContext::new().unwrap();
        let n = ctx.device_count();
        let total = n * 10 + 3; // Not evenly divisible
        let ranges = ctx.distribute_range(total);

        let sum: usize = ranges.iter().map(|r| r.length).sum();
        assert_eq!(sum, total);

        // Each range should be contiguous
        let mut offset = 0;
        for r in &ranges {
            assert_eq!(r.offset, offset);
            offset += r.length;
        }
    }

    #[test]
    fn test_all_properties() {
        let ctx = MultiGpuContext::new().unwrap();
        let props = ctx.all_properties();
        assert_eq!(props.len(), ctx.device_count());
    }

    #[test]
    fn test_memcpy_peer() {
        // Should succeed in CPU emulation mode
        assert!(memcpy_peer(0, 0, 1024).is_ok());
    }

    #[test]
    fn test_default_context() {
        let ctx = MultiGpuContext::default();
        assert!(ctx.device_count() >= 1);
    }
}
