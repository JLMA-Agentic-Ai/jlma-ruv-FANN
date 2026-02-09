//! Device memory allocation and management

use crate::{Result, runtime_error};
use crate::runtime::{Device, BackendType};
use std::marker::PhantomData;
use std::sync::Arc;
use std::alloc::{alloc, dealloc, Layout};

/// Raw device memory pointer
pub struct DevicePtr {
    raw: *mut u8,
    size: usize,
    backend: BackendType,
}

impl DevicePtr {
    /// Allocate raw device memory
    pub fn allocate(size: usize, device: &Arc<Device>) -> Result<Self> {
        if size == 0 {
            return Err(runtime_error!("Cannot allocate zero-sized buffer"));
        }

        let backend = device.backend();
        let raw = match backend {
            BackendType::Native => {
                // Host-memory allocation serves as the compute target for CPU and
                // fallback backends.  When a native GPU backend is active, the
                // Backend::allocate_memory path handles device-side allocation.
                unsafe {
                    let layout = Layout::from_size_align(size, 8)
                        .map_err(|e| runtime_error!("Invalid layout: {}", e))?;
                    alloc(layout)
                }
            }
            BackendType::WebGPU => {
                // Host-memory allocation for the runtime abstraction layer.
                // The high-level WebGPU backend manages its own device-side
                // buffer objects; DevicePtr provides the host-side mirror.
                unsafe {
                    let layout = Layout::from_size_align(size, 8)
                        .map_err(|e| runtime_error!("Invalid layout: {}", e))?;
                    alloc(layout)
                }
            }
            BackendType::CPU => {
                // CPU backend uses regular heap allocation
                unsafe {
                    let layout = Layout::from_size_align(size, 8)
                        .map_err(|e| runtime_error!("Invalid layout: {}", e))?;
                    alloc(layout)
                }
            }
        };

        if raw.is_null() {
            return Err(runtime_error!("Failed to allocate {} bytes of device memory", size));
        }

        Ok(Self { raw, size, backend })
    }

    /// Get raw pointer
    pub fn as_ptr(&self) -> *const u8 {
        self.raw
    }

    /// Get mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.raw
    }

    /// Get allocation size
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for DevicePtr {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            match self.backend {
                BackendType::Native => {
                    // Host-side deallocation for the runtime abstraction.
                    // Native GPU backends handle their own device-side frees.
                    unsafe {
                        if let Ok(layout) = Layout::from_size_align(self.size, 8) {
                            dealloc(self.raw, layout);
                        }
                    }
                }
                BackendType::WebGPU => {
                    // Host-side deallocation for the runtime abstraction.
                    // WebGPU device buffers are released by the backend layer.
                    unsafe {
                        if let Ok(layout) = Layout::from_size_align(self.size, 8) {
                            dealloc(self.raw, layout);
                        }
                    }
                }
                BackendType::CPU => {
                    unsafe {
                        if let Ok(layout) = Layout::from_size_align(self.size, 8) {
                            dealloc(self.raw, layout);
                        }
                    }
                }
            }
        }
    }
}

/// Device memory buffer
pub struct DeviceBuffer<T> {
    ptr: DevicePtr,
    len: usize,
    device: Arc<Device>,
    phantom: PhantomData<T>,
}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocate a new device buffer
    pub fn new(len: usize, device: Arc<Device>) -> Result<Self> {
        if len == 0 {
            return Err(runtime_error!("Cannot allocate zero-length buffer"));
        }

        let size = len * std::mem::size_of::<T>();
        let ptr = DevicePtr::allocate(size, &device)?;

        Ok(Self {
            ptr,
            len,
            device,
            phantom: PhantomData,
        })
    }
    
    /// Get buffer length
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the device this buffer is allocated on
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get raw device pointer
    /// 
    /// # Safety
    /// The caller must ensure that the returned pointer is not used after the `DeviceBuffer` is dropped.
    /// The caller must also ensure that the memory is not accessed concurrently.
    pub unsafe fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr() as *const T
    }

    /// Get mutable raw device pointer
    /// 
    /// # Safety
    /// The caller must ensure that the returned pointer is not used after the `DeviceBuffer` is dropped.
    /// The caller must also ensure that the memory is not accessed concurrently.
    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_mut_ptr() as *mut T
    }
    
    /// Copy from host memory
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(runtime_error!(
                "Host buffer length {} doesn't match device buffer length {}",
                data.len(),
                self.len
            ));
        }

        let size = self.len * std::mem::size_of::<T>();
        
        match self.device.backend() {
            BackendType::Native => {
                // Runtime-level host-to-host copy.  The native GPU backend
                // performs its own host-to-device transfers when dispatching
                // kernels; this path keeps the host mirror in sync.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        self.ptr.as_mut_ptr(),
                        size
                    );
                }
            }
            BackendType::WebGPU => {
                // Runtime-level host copy.  The WebGPU backend writes to its
                // own device buffers independently; this maintains the host mirror.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        self.ptr.as_mut_ptr(),
                        size
                    );
                }
            }
            BackendType::CPU => {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        self.ptr.as_mut_ptr(),
                        size
                    );
                }
            }
        }

        Ok(())
    }
    
    /// Copy to host memory
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<()> {
        if data.len() != self.len {
            return Err(runtime_error!(
                "Host buffer length {} doesn't match device buffer length {}",
                data.len(),
                self.len
            ));
        }

        let size = self.len * std::mem::size_of::<T>();
        
        match self.device.backend() {
            BackendType::Native => {
                // Runtime-level host-to-host copy.  The native GPU backend
                // performs its own device-to-host transfers after kernel
                // execution; this path reads from the host mirror.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.ptr.as_ptr(),
                        data.as_mut_ptr() as *mut u8,
                        size
                    );
                }
            }
            BackendType::WebGPU => {
                // Runtime-level host copy.  The WebGPU backend reads from its
                // own device buffers independently; this reads the host mirror.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.ptr.as_ptr(),
                        data.as_mut_ptr() as *mut u8,
                        size
                    );
                }
            }
            BackendType::CPU => {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.ptr.as_ptr(),
                        data.as_mut_ptr() as *mut u8,
                        size
                    );
                }
            }
        }

        Ok(())
    }

    /// Fill buffer with a value
    pub fn fill(&mut self, value: T) -> Result<()> {
        // For now, copy to host, fill, and copy back
        // TODO: Optimize with kernel-based fill
        let host_data = vec![value; self.len];
        self.copy_from_host(&host_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Device;

    #[test]
    fn test_device_buffer_allocation() {
        let device = Device::get_default().unwrap();
        let buffer = DeviceBuffer::<f32>::new(1024, device).unwrap();
        assert_eq!(buffer.len(), 1024);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_host_device_copy() {
        let device = Device::get_default().unwrap();
        let mut buffer = DeviceBuffer::<f32>::new(100, device).unwrap();
        
        // Create test data
        let host_data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        
        // Copy to device
        buffer.copy_from_host(&host_data).unwrap();
        
        // Copy back
        let mut result = vec![0.0; 100];
        buffer.copy_to_host(&mut result).unwrap();
        
        // Verify
        assert_eq!(host_data, result);
    }
}