//! Texture memory for GPU-style 2D/3D data access with interpolation
//!
//! Provides a software emulation of CUDA texture memory, supporting
//! nearest-neighbor and bilinear filtering, clamping and wrapping address
//! modes, and normalized coordinate access.

use crate::{Result, memory_error};
use std::sync::Arc;

/// Texture addressing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    /// Clamp to edge (repeat edge texels)
    Clamp,
    /// Wrap around (modulo)
    Wrap,
    /// Mirror at boundaries
    Mirror,
    /// Return zero outside [0, dim)
    Border,
}

/// Texture filter mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    /// Nearest-neighbor sampling (point)
    Point,
    /// Bilinear interpolation
    Linear,
}

/// Texture descriptor
#[derive(Debug, Clone)]
pub struct TextureDescriptor {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub address_mode: AddressMode,
    pub filter_mode: FilterMode,
    pub normalized_coords: bool,
}

impl TextureDescriptor {
    /// Create a 1D texture descriptor
    pub fn new_1d(width: usize) -> Self {
        Self {
            width,
            height: 1,
            depth: 1,
            address_mode: AddressMode::Clamp,
            filter_mode: FilterMode::Point,
            normalized_coords: false,
        }
    }

    /// Create a 2D texture descriptor
    pub fn new_2d(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            depth: 1,
            address_mode: AddressMode::Clamp,
            filter_mode: FilterMode::Point,
            normalized_coords: false,
        }
    }

    /// Create a 3D texture descriptor
    pub fn new_3d(width: usize, height: usize, depth: usize) -> Self {
        Self {
            width,
            height,
            depth,
            address_mode: AddressMode::Clamp,
            filter_mode: FilterMode::Point,
            normalized_coords: false,
        }
    }

    /// Set address mode
    pub fn with_address_mode(mut self, mode: AddressMode) -> Self {
        self.address_mode = mode;
        self
    }

    /// Set filter mode
    pub fn with_filter_mode(mut self, mode: FilterMode) -> Self {
        self.filter_mode = mode;
        self
    }

    /// Enable normalized coordinates
    pub fn with_normalized_coords(mut self, normalized: bool) -> Self {
        self.normalized_coords = normalized;
        self
    }
}

/// Texture memory object providing GPU-style texture sampling
///
/// Supports 1D, 2D, and 3D textures with configurable addressing
/// and filtering modes. Data is stored as `f32` values internally.
pub struct TextureMemory {
    data: Vec<f32>,
    descriptor: TextureDescriptor,
}

impl TextureMemory {
    /// Create a new texture from data and descriptor
    pub fn new(data: Vec<f32>, descriptor: TextureDescriptor) -> Result<Self> {
        let expected = descriptor.width * descriptor.height * descriptor.depth;
        if data.len() != expected {
            return Err(memory_error!(
                "Texture data length {} doesn't match dimensions {}x{}x{} = {}",
                data.len(), descriptor.width, descriptor.height, descriptor.depth, expected
            ));
        }
        Ok(Self { data, descriptor })
    }

    /// Create a zeroed texture
    pub fn zeroed(descriptor: TextureDescriptor) -> Self {
        let size = descriptor.width * descriptor.height * descriptor.depth;
        Self {
            data: vec![0.0; size],
            descriptor,
        }
    }

    /// Get texture descriptor
    pub fn descriptor(&self) -> &TextureDescriptor {
        &self.descriptor
    }

    /// Get width
    pub fn width(&self) -> usize {
        self.descriptor.width
    }

    /// Get height
    pub fn height(&self) -> usize {
        self.descriptor.height
    }

    /// Get depth
    pub fn depth(&self) -> usize {
        self.descriptor.depth
    }

    /// Bind data to the texture (copy from slice)
    pub fn bind(&mut self, data: &[f32]) -> Result<()> {
        let expected = self.descriptor.width * self.descriptor.height * self.descriptor.depth;
        if data.len() != expected {
            return Err(memory_error!(
                "Data length {} doesn't match texture size {}",
                data.len(), expected
            ));
        }
        self.data.copy_from_slice(data);
        Ok(())
    }

    /// Sample the texture at 1D coordinate
    pub fn sample_1d(&self, x: f32) -> f32 {
        let fx = if self.descriptor.normalized_coords {
            x * self.descriptor.width as f32
        } else {
            x
        };

        match self.descriptor.filter_mode {
            FilterMode::Point => {
                let ix = self.address_coord(fx.round() as isize, self.descriptor.width);
                self.data[ix]
            }
            FilterMode::Linear => {
                let x0 = fx.floor();
                let frac = fx - x0;
                let i0 = self.address_coord(x0 as isize, self.descriptor.width);
                let i1 = self.address_coord(x0 as isize + 1, self.descriptor.width);
                self.data[i0] * (1.0 - frac) + self.data[i1] * frac
            }
        }
    }

    /// Sample the texture at 2D coordinates
    pub fn sample_2d(&self, x: f32, y: f32) -> f32 {
        let fx = if self.descriptor.normalized_coords {
            x * self.descriptor.width as f32
        } else {
            x
        };
        let fy = if self.descriptor.normalized_coords {
            y * self.descriptor.height as f32
        } else {
            y
        };

        match self.descriptor.filter_mode {
            FilterMode::Point => {
                let ix = self.address_coord(fx.round() as isize, self.descriptor.width);
                let iy = self.address_coord(fy.round() as isize, self.descriptor.height);
                self.data[iy * self.descriptor.width + ix]
            }
            FilterMode::Linear => {
                let x0 = fx.floor();
                let y0 = fy.floor();
                let fx_frac = fx - x0;
                let fy_frac = fy - y0;

                let ix0 = self.address_coord(x0 as isize, self.descriptor.width);
                let ix1 = self.address_coord(x0 as isize + 1, self.descriptor.width);
                let iy0 = self.address_coord(y0 as isize, self.descriptor.height);
                let iy1 = self.address_coord(y0 as isize + 1, self.descriptor.height);

                let w = self.descriptor.width;
                let v00 = self.data[iy0 * w + ix0];
                let v10 = self.data[iy0 * w + ix1];
                let v01 = self.data[iy1 * w + ix0];
                let v11 = self.data[iy1 * w + ix1];

                let top = v00 * (1.0 - fx_frac) + v10 * fx_frac;
                let bot = v01 * (1.0 - fx_frac) + v11 * fx_frac;
                top * (1.0 - fy_frac) + bot * fy_frac
            }
        }
    }

    /// Sample the texture at 3D coordinates
    pub fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let fx = if self.descriptor.normalized_coords {
            x * self.descriptor.width as f32
        } else {
            x
        };
        let fy = if self.descriptor.normalized_coords {
            y * self.descriptor.height as f32
        } else {
            y
        };
        let fz = if self.descriptor.normalized_coords {
            z * self.descriptor.depth as f32
        } else {
            z
        };

        let ix = self.address_coord(fx.round() as isize, self.descriptor.width);
        let iy = self.address_coord(fy.round() as isize, self.descriptor.height);
        let iz = self.address_coord(fz.round() as isize, self.descriptor.depth);

        let w = self.descriptor.width;
        let h = self.descriptor.height;
        self.data[iz * w * h + iy * w + ix]
    }

    /// Read raw data back
    pub fn read_data(&self) -> &[f32] {
        &self.data
    }

    /// Write to a specific texel
    pub fn write_texel(&mut self, x: usize, y: usize, value: f32) -> Result<()> {
        if x >= self.descriptor.width || y >= self.descriptor.height {
            return Err(memory_error!(
                "Texel ({}, {}) out of bounds ({}x{})",
                x, y, self.descriptor.width, self.descriptor.height
            ));
        }
        self.data[y * self.descriptor.width + x] = value;
        Ok(())
    }

    /// Apply address mode to a coordinate
    fn address_coord(&self, coord: isize, dim: usize) -> usize {
        let d = dim as isize;
        match self.descriptor.address_mode {
            AddressMode::Clamp => coord.clamp(0, d - 1) as usize,
            AddressMode::Wrap => ((coord % d + d) % d) as usize,
            AddressMode::Mirror => {
                let c = ((coord % (2 * d) + 2 * d) % (2 * d)) as usize;
                if c < dim { c } else { 2 * dim - c - 1 }
            }
            AddressMode::Border => {
                if coord < 0 || coord >= d { 0 } else { coord as usize }
            }
        }
    }
}

/// Shared texture handle
pub type SharedTexture = Arc<TextureMemory>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texture_1d_point_sampling() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let desc = TextureDescriptor::new_1d(4);
        let tex = TextureMemory::new(data, desc).unwrap();

        assert_eq!(tex.sample_1d(0.0), 1.0);
        assert_eq!(tex.sample_1d(1.0), 2.0);
        assert_eq!(tex.sample_1d(3.0), 4.0);
    }

    #[test]
    fn test_texture_1d_linear_sampling() {
        let data = vec![0.0, 10.0, 20.0, 30.0];
        let desc = TextureDescriptor::new_1d(4).with_filter_mode(FilterMode::Linear);
        let tex = TextureMemory::new(data, desc).unwrap();

        assert!((tex.sample_1d(0.5) - 5.0).abs() < 1e-5);
        assert!((tex.sample_1d(1.5) - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_texture_2d_point_sampling() {
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let desc = TextureDescriptor::new_2d(3, 2);
        let tex = TextureMemory::new(data, desc).unwrap();

        assert_eq!(tex.sample_2d(0.0, 0.0), 1.0);
        assert_eq!(tex.sample_2d(2.0, 0.0), 3.0);
        assert_eq!(tex.sample_2d(0.0, 1.0), 4.0);
        assert_eq!(tex.sample_2d(2.0, 1.0), 6.0);
    }

    #[test]
    fn test_texture_2d_bilinear_sampling() {
        let data = vec![
            0.0, 10.0,
            10.0, 20.0,
        ];
        let desc = TextureDescriptor::new_2d(2, 2).with_filter_mode(FilterMode::Linear);
        let tex = TextureMemory::new(data, desc).unwrap();

        // Center should be average of all four
        let center = tex.sample_2d(0.5, 0.5);
        assert!((center - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_texture_address_clamp() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let desc = TextureDescriptor::new_1d(4).with_address_mode(AddressMode::Clamp);
        let tex = TextureMemory::new(data, desc).unwrap();

        // Out of bounds should clamp to edge
        assert_eq!(tex.sample_1d(-1.0), 1.0);
        assert_eq!(tex.sample_1d(10.0), 4.0);
    }

    #[test]
    fn test_texture_address_wrap() {
        let data = vec![10.0, 20.0, 30.0, 40.0];
        let desc = TextureDescriptor::new_1d(4).with_address_mode(AddressMode::Wrap);
        let tex = TextureMemory::new(data, desc).unwrap();

        assert_eq!(tex.sample_1d(4.0), 10.0); // wraps to 0
        assert_eq!(tex.sample_1d(5.0), 20.0); // wraps to 1
    }

    #[test]
    fn test_texture_normalized_coords() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let desc = TextureDescriptor::new_1d(4).with_normalized_coords(true);
        let tex = TextureMemory::new(data, desc).unwrap();

        // 0.0 maps to index 0, 0.5 maps to index 2
        assert_eq!(tex.sample_1d(0.0), 1.0);
        assert_eq!(tex.sample_1d(0.5), 3.0);
    }

    #[test]
    fn test_texture_bind_data() {
        let desc = TextureDescriptor::new_1d(4);
        let mut tex = TextureMemory::zeroed(desc);

        assert_eq!(tex.sample_1d(0.0), 0.0);
        tex.bind(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(tex.sample_1d(0.0), 5.0);
        assert_eq!(tex.sample_1d(3.0), 8.0);
    }

    #[test]
    fn test_texture_write_texel() {
        let desc = TextureDescriptor::new_2d(4, 4);
        let mut tex = TextureMemory::zeroed(desc);

        tex.write_texel(2, 1, 42.0).unwrap();
        assert_eq!(tex.sample_2d(2.0, 1.0), 42.0);
    }

    #[test]
    fn test_texture_write_texel_out_of_bounds() {
        let desc = TextureDescriptor::new_2d(4, 4);
        let mut tex = TextureMemory::zeroed(desc);

        assert!(tex.write_texel(10, 0, 1.0).is_err());
    }

    #[test]
    fn test_texture_data_size_mismatch() {
        let desc = TextureDescriptor::new_2d(3, 3);
        let result = TextureMemory::new(vec![0.0; 5], desc);
        assert!(result.is_err());
    }

    #[test]
    fn test_texture_3d_sampling() {
        let data = vec![0.0; 2 * 2 * 2];
        let desc = TextureDescriptor::new_3d(2, 2, 2);
        let mut tex = TextureMemory::new(data, desc).unwrap();

        // Write to (1, 1, 1)
        tex.data[1 * 2 * 2 + 1 * 2 + 1] = 99.0;
        assert_eq!(tex.sample_3d(1.0, 1.0, 1.0), 99.0);
    }
}
