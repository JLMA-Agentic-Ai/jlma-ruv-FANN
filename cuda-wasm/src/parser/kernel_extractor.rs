//! Kernel extraction utilities
//!
//! Extracts kernel definitions and metadata from a parsed CUDA AST.

use super::ast::*;

/// Information about an extracted kernel
#[derive(Debug, Clone)]
pub struct KernelInfo {
    /// Kernel name
    pub name: String,
    /// Kernel parameters
    pub params: Vec<Parameter>,
    /// Kernel attributes (launch bounds, etc.)
    pub attributes: Vec<KernelAttribute>,
    /// Whether the kernel uses shared memory
    pub uses_shared_memory: bool,
    /// Whether the kernel uses syncthreads
    pub uses_sync_threads: bool,
    /// Set of CUDA builtins referenced (threadIdx, blockIdx, etc.)
    pub referenced_builtins: Vec<String>,
    /// Names of functions called from within the kernel
    pub called_functions: Vec<String>,
}

/// Extract all kernel definitions from an AST
pub fn extract_kernels(ast: &Ast) -> Vec<KernelInfo> {
    ast.items
        .iter()
        .filter_map(|item| {
            if let Item::Kernel(kernel) = item {
                Some(analyze_kernel(kernel))
            } else {
                None
            }
        })
        .collect()
}

/// Extract a single kernel by name
pub fn extract_kernel_by_name<'a>(ast: &'a Ast, name: &str) -> Option<&'a KernelDef> {
    ast.items.iter().find_map(|item| {
        if let Item::Kernel(kernel) = item {
            if kernel.name == name {
                return Some(kernel);
            }
        }
        None
    })
}

/// Extract all device functions from the AST
pub fn extract_device_functions(ast: &Ast) -> Vec<&FunctionDef> {
    ast.items
        .iter()
        .filter_map(|item| {
            if let Item::DeviceFunction(func) = item {
                Some(func)
            } else {
                None
            }
        })
        .collect()
}

/// Analyze a kernel definition to produce KernelInfo
fn analyze_kernel(kernel: &KernelDef) -> KernelInfo {
    let mut info = KernelInfo {
        name: kernel.name.clone(),
        params: kernel.params.clone(),
        attributes: kernel.attributes.clone(),
        uses_shared_memory: false,
        uses_sync_threads: false,
        referenced_builtins: Vec::new(),
        called_functions: Vec::new(),
    };

    visit_block(&kernel.body, &mut info);

    // Deduplicate
    info.referenced_builtins.sort();
    info.referenced_builtins.dedup();
    info.called_functions.sort();
    info.called_functions.dedup();

    info
}

fn visit_block(block: &Block, info: &mut KernelInfo) {
    for stmt in &block.statements {
        visit_statement(stmt, info);
    }
}

fn visit_statement(stmt: &Statement, info: &mut KernelInfo) {
    match stmt {
        Statement::VarDecl { storage, init, .. } => {
            if matches!(storage, StorageClass::Shared) {
                info.uses_shared_memory = true;
            }
            if let Some(expr) = init {
                visit_expression(expr, info);
            }
        }
        Statement::Expr(expr) => {
            visit_expression(expr, info);
        }
        Statement::Block(block) => {
            visit_block(block, info);
        }
        Statement::If { condition, then_branch, else_branch } => {
            visit_expression(condition, info);
            visit_statement(then_branch, info);
            if let Some(else_stmt) = else_branch {
                visit_statement(else_stmt, info);
            }
        }
        Statement::For { init, condition, update, body } => {
            if let Some(init_stmt) = init {
                visit_statement(init_stmt, info);
            }
            if let Some(cond) = condition {
                visit_expression(cond, info);
            }
            if let Some(upd) = update {
                visit_expression(upd, info);
            }
            visit_statement(body, info);
        }
        Statement::While { condition, body } => {
            visit_expression(condition, info);
            visit_statement(body, info);
        }
        Statement::Return(Some(expr)) => {
            visit_expression(expr, info);
        }
        Statement::SyncThreads => {
            info.uses_sync_threads = true;
        }
        _ => {}
    }
}

fn visit_expression(expr: &Expression, info: &mut KernelInfo) {
    match expr {
        Expression::ThreadIdx(dim) => {
            info.referenced_builtins.push(format!("threadIdx.{}", dim_str(dim)));
        }
        Expression::BlockIdx(dim) => {
            info.referenced_builtins.push(format!("blockIdx.{}", dim_str(dim)));
        }
        Expression::BlockDim(dim) => {
            info.referenced_builtins.push(format!("blockDim.{}", dim_str(dim)));
        }
        Expression::GridDim(dim) => {
            info.referenced_builtins.push(format!("gridDim.{}", dim_str(dim)));
        }
        Expression::Binary { left, right, .. } => {
            visit_expression(left, info);
            visit_expression(right, info);
        }
        Expression::Unary { expr, .. } => {
            visit_expression(expr, info);
        }
        Expression::Call { name, args } => {
            if name != "__syncthreads" && name != "__ternary__" && name != "sizeof" {
                info.called_functions.push(name.clone());
            }
            if name == "__syncthreads" {
                info.uses_sync_threads = true;
            }
            for arg in args {
                visit_expression(arg, info);
            }
        }
        Expression::Index { array, index } => {
            visit_expression(array, info);
            visit_expression(index, info);
        }
        Expression::Member { object, .. } => {
            visit_expression(object, info);
        }
        Expression::Cast { expr, .. } => {
            visit_expression(expr, info);
        }
        Expression::WarpPrimitive { args, .. } => {
            for arg in args {
                visit_expression(arg, info);
            }
        }
        _ => {}
    }
}

fn dim_str(dim: &Dimension) -> &'static str {
    match dim {
        Dimension::X => "x",
        Dimension::Y => "y",
        Dimension::Z => "z",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::CudaParser;

    #[test]
    fn test_extract_vector_add() {
        let src = r#"
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"#;
        let parser = CudaParser::new();
        let ast = parser.parse(src).unwrap();
        let kernels = extract_kernels(&ast);
        assert_eq!(kernels.len(), 1);
        let k = &kernels[0];
        assert_eq!(k.name, "vectorAdd");
        assert_eq!(k.params.len(), 4);
        assert!(!k.uses_shared_memory);
        assert!(!k.uses_sync_threads);
        assert!(k.referenced_builtins.contains(&"threadIdx.x".to_string()));
        assert!(k.referenced_builtins.contains(&"blockIdx.x".to_string()));
        assert!(k.referenced_builtins.contains(&"blockDim.x".to_string()));
    }

    #[test]
    fn test_extract_shared_memory_kernel() {
        let src = r#"
__global__ void matMul(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + 15) / 16; t++) {
        sA[threadIdx.y][threadIdx.x] = A[row * K + t * 16 + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < 16; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
"#;
        let parser = CudaParser::new();
        let ast = parser.parse(src).unwrap();
        let kernels = extract_kernels(&ast);
        assert_eq!(kernels.len(), 1);
        let k = &kernels[0];
        assert_eq!(k.name, "matMul");
        assert!(k.uses_shared_memory);
        assert!(k.uses_sync_threads);
    }

    #[test]
    fn test_extract_kernel_by_name() {
        let src = r#"
__global__ void kernel1(int* a) { a[threadIdx.x] = 0; }
__global__ void kernel2(float* b) { b[threadIdx.x] = 1.0f; }
"#;
        let parser = CudaParser::new();
        let ast = parser.parse(src).unwrap();
        assert!(extract_kernel_by_name(&ast, "kernel1").is_some());
        assert!(extract_kernel_by_name(&ast, "kernel2").is_some());
        assert!(extract_kernel_by_name(&ast, "kernel3").is_none());
    }
}
