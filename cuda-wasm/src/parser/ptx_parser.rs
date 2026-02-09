//! PTX (Parallel Thread Execution) parser
//!
//! Parses NVIDIA PTX assembly into a structured representation and converts
//! it to the common CUDA AST for downstream transpilation to WGSL/Rust.

use crate::{translation_error, Result};

/// PTX module — top-level compilation unit
#[derive(Debug, Clone)]
pub struct PtxModule {
    /// PTX ISA version (e.g., "7.8")
    pub version: String,
    /// Target architecture (e.g., "sm_80")
    pub target: String,
    /// Address size in bits (32 or 64)
    pub address_size: u32,
    /// Top-level directives (functions, variables, etc.)
    pub directives: Vec<PtxDirective>,
}

/// PTX top-level directive
#[derive(Debug, Clone)]
pub enum PtxDirective {
    /// Kernel entry point (.entry)
    Entry(PtxFunction),
    /// Device function (.func)
    Function(PtxFunction),
    /// Global variable (.global)
    GlobalVar(PtxVariable),
    /// Constant variable (.const)
    ConstVar(PtxVariable),
    /// Shared variable (.shared)
    SharedVar(PtxVariable),
}

/// PTX function (entry or helper)
#[derive(Debug, Clone)]
pub struct PtxFunction {
    /// Function name
    pub name: String,
    /// Parameters (.param declarations)
    pub params: Vec<PtxVariable>,
    /// Register declarations (.reg)
    pub registers: Vec<PtxRegDecl>,
    /// Local variable declarations
    pub locals: Vec<PtxVariable>,
    /// Instruction body
    pub body: Vec<PtxStatement>,
    /// Whether this is an entry point
    pub is_entry: bool,
}

/// PTX register declaration
#[derive(Debug, Clone)]
pub struct PtxRegDecl {
    /// Register type
    pub reg_type: PtxType,
    /// Register names
    pub names: Vec<String>,
    /// Number of registers (for array decls like .reg .f32 %f<32>)
    pub count: Option<u32>,
}

/// PTX variable
#[derive(Debug, Clone)]
pub struct PtxVariable {
    /// Variable name
    pub name: String,
    /// Type
    pub var_type: PtxType,
    /// Storage space
    pub space: PtxSpace,
    /// Array size (number of elements, if any)
    pub array_size: Option<u32>,
    /// Alignment
    pub alignment: Option<u32>,
}

/// PTX data types
#[derive(Debug, Clone, PartialEq)]
pub enum PtxType {
    Pred,
    B8, B16, B32, B64,
    S8, S16, S32, S64,
    U8, U16, U32, U64,
    F16, F32, F64,
}

/// PTX address spaces
#[derive(Debug, Clone, PartialEq)]
pub enum PtxSpace {
    Reg,
    Param,
    Local,
    Shared,
    Global,
    Const,
}

/// PTX statement (instruction or label)
#[derive(Debug, Clone)]
pub enum PtxStatement {
    /// Label target
    Label(String),
    /// Instruction (possibly predicated)
    Instruction(PtxInstruction),
}

/// PTX instruction
#[derive(Debug, Clone)]
pub struct PtxInstruction {
    /// Optional predicate guard (@p or @!p)
    pub predicate: Option<PtxPredicate>,
    /// Opcode (e.g., "add", "ld", "st", "setp", "bra")
    pub opcode: String,
    /// Type suffix (e.g., ".f32", ".s32")
    pub type_suffix: Option<PtxType>,
    /// Modifier suffixes (e.g., ".rn", ".uni", ".wide", ".lu")
    pub modifiers: Vec<String>,
    /// Operands
    pub operands: Vec<PtxOperand>,
}

/// PTX operand
#[derive(Debug, Clone)]
pub enum PtxOperand {
    /// Register (%r0, %f1, %p0)
    Register(String),
    /// Special register (%tid.x, %ctaid.y, %ntid.z, %laneid, %warpid)
    SpecialReg(String),
    /// Immediate integer
    ImmInt(i64),
    /// Immediate float
    ImmFloat(f64),
    /// Label reference
    Label(String),
    /// Memory address [%r0], [%r0+4], [name]
    Address { base: String, offset: Option<i64> },
    /// Vector operand {%r0, %r1, %r2, %r3}
    Vector(Vec<String>),
}

/// PTX predicate guard
#[derive(Debug, Clone)]
pub struct PtxPredicate {
    /// Register name (e.g., "p0")
    pub register: String,
    /// Whether negated (@!p)
    pub negated: bool,
}

/// Parse a PTX source string into a PtxModule
pub fn parse_ptx(input: &str) -> Result<PtxModule> {
    let mut module = PtxModule {
        version: String::new(),
        target: String::new(),
        address_size: 64,
        directives: Vec::new(),
    };

    let lines: Vec<&str> = input.lines().map(|l| l.trim()).collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with("//") {
            i += 1;
            continue;
        }

        if line.starts_with(".version") {
            module.version = extract_value(line, ".version");
        } else if line.starts_with(".target") {
            module.target = extract_value(line, ".target");
        } else if line.starts_with(".address_size") {
            module.address_size = extract_value(line, ".address_size")
                .parse()
                .unwrap_or(64);
        } else if line.contains(".entry") || line.contains(".func") {
            let is_entry = line.contains(".entry");
            let (func, end_idx) = parse_function(&lines, i, is_entry)?;
            let directive = if is_entry {
                PtxDirective::Entry(func)
            } else {
                PtxDirective::Function(func)
            };
            module.directives.push(directive);
            i = end_idx;
        } else if line.starts_with(".global") {
            if let Some(var) = parse_variable(line, PtxSpace::Global) {
                module.directives.push(PtxDirective::GlobalVar(var));
            }
        } else if line.starts_with(".const") {
            if let Some(var) = parse_variable(line, PtxSpace::Const) {
                module.directives.push(PtxDirective::ConstVar(var));
            }
        } else if line.starts_with(".shared") {
            if let Some(var) = parse_variable(line, PtxSpace::Shared) {
                module.directives.push(PtxDirective::SharedVar(var));
            }
        }

        i += 1;
    }

    Ok(module)
}

/// Convert a PtxModule to the common CUDA AST for downstream transpilation
pub fn ptx_to_ast(module: &PtxModule) -> Result<crate::parser::ast::Ast> {
    use crate::parser::ast::*;

    let mut items = Vec::new();

    for directive in &module.directives {
        match directive {
            PtxDirective::Entry(func) => {
                let params = func.params.iter().map(|p| Parameter {
                    name: clean_name(&p.name),
                    ty: ptx_type_to_ast(&p.var_type),
                    qualifiers: vec![],
                }).collect();

                let body = ptx_body_to_ast(&func.body)?;

                items.push(Item::Kernel(KernelDef {
                    name: clean_name(&func.name),
                    params,
                    body,
                    attributes: vec![],
                }));
            }
            PtxDirective::Function(func) => {
                let params = func.params.iter().map(|p| Parameter {
                    name: clean_name(&p.name),
                    ty: ptx_type_to_ast(&p.var_type),
                    qualifiers: vec![],
                }).collect();

                let body = ptx_body_to_ast(&func.body)?;

                items.push(Item::DeviceFunction(FunctionDef {
                    name: clean_name(&func.name),
                    return_type: Type::Void,
                    params,
                    body,
                    qualifiers: vec![FunctionQualifier::Device],
                }));
            }
            PtxDirective::GlobalVar(var) => {
                items.push(Item::GlobalVar(GlobalVar {
                    name: clean_name(&var.name),
                    ty: ptx_type_to_ast(&var.var_type),
                    storage: StorageClass::Global,
                    init: None,
                }));
            }
            PtxDirective::ConstVar(var) => {
                items.push(Item::GlobalVar(GlobalVar {
                    name: clean_name(&var.name),
                    ty: ptx_type_to_ast(&var.var_type),
                    storage: StorageClass::Constant,
                    init: None,
                }));
            }
            PtxDirective::SharedVar(var) => {
                items.push(Item::GlobalVar(GlobalVar {
                    name: clean_name(&var.name),
                    ty: ptx_type_to_ast(&var.var_type),
                    storage: StorageClass::Shared,
                    init: None,
                }));
            }
        }
    }

    Ok(Ast { items })
}

// --- Internal helpers ---

fn extract_value(line: &str, prefix: &str) -> String {
    line.trim_start_matches(prefix)
        .trim()
        .trim_end_matches(';')
        .trim()
        .to_string()
}

fn clean_name(name: &str) -> String {
    name.trim_start_matches('%').trim_start_matches('_').to_string()
}

fn parse_type(s: &str) -> Option<PtxType> {
    match s.trim_start_matches('.') {
        "pred" => Some(PtxType::Pred),
        "b8" => Some(PtxType::B8), "b16" => Some(PtxType::B16),
        "b32" => Some(PtxType::B32), "b64" => Some(PtxType::B64),
        "s8" => Some(PtxType::S8), "s16" => Some(PtxType::S16),
        "s32" => Some(PtxType::S32), "s64" => Some(PtxType::S64),
        "u8" => Some(PtxType::U8), "u16" => Some(PtxType::U16),
        "u32" => Some(PtxType::U32), "u64" => Some(PtxType::U64),
        "f16" => Some(PtxType::F16), "f32" => Some(PtxType::F32),
        "f64" => Some(PtxType::F64),
        _ => None,
    }
}

fn parse_variable(line: &str, space: PtxSpace) -> Option<PtxVariable> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < 3 { return None; }

    let var_type = parse_type(tokens[1]).unwrap_or(PtxType::B32);
    let name = tokens.last()?.trim_end_matches(';').to_string();

    Some(PtxVariable {
        name,
        var_type,
        space,
        array_size: None,
        alignment: None,
    })
}

fn parse_function(lines: &[&str], start: usize, is_entry: bool) -> Result<(PtxFunction, usize)> {
    let header = lines[start];
    let name = extract_func_name(header);

    let mut func = PtxFunction {
        name,
        params: Vec::new(),
        registers: Vec::new(),
        locals: Vec::new(),
        body: Vec::new(),
        is_entry,
    };

    let mut i = start + 1;
    let mut in_body = false;
    let mut brace_depth = if header.contains('{') { 1 } else { 0 };

    if brace_depth > 0 { in_body = true; }

    while i < lines.len() {
        let line = lines[i];

        if !in_body {
            if line.contains('{') {
                in_body = true;
                brace_depth += line.matches('{').count();
                brace_depth -= line.matches('}').count();
                if brace_depth == 0 { return Ok((func, i)); }
                i += 1;
                continue;
            }
            if line.contains(".param") {
                if let Some(var) = parse_variable(line, PtxSpace::Param) {
                    func.params.push(var);
                }
            }
        } else {
            brace_depth += line.matches('{').count();
            brace_depth -= line.matches('}').count();

            if brace_depth == 0 {
                return Ok((func, i));
            }

            if line.contains(".reg") {
                let tokens: Vec<&str> = line.split_whitespace().collect();
                if tokens.len() >= 3 {
                    let reg_type = parse_type(tokens[1]).unwrap_or(PtxType::B32);
                    let name_part = tokens[2].trim_end_matches(';');
                    // Handle array syntax %f<32>
                    let (names, count) = if name_part.contains('<') {
                        let parts: Vec<&str> = name_part.split('<').collect();
                        let base = parts[0].to_string();
                        let cnt: u32 = parts.get(1)
                            .and_then(|s| s.trim_end_matches('>').parse().ok())
                            .unwrap_or(1);
                        (vec![base], Some(cnt))
                    } else {
                        (vec![name_part.to_string()], None)
                    };
                    func.registers.push(PtxRegDecl { reg_type, names, count });
                }
            } else if line.contains(".local") || line.contains(".shared") {
                let space = if line.contains(".shared") { PtxSpace::Shared } else { PtxSpace::Local };
                if let Some(var) = parse_variable(line, space) {
                    func.locals.push(var);
                }
            } else if !line.is_empty() && !line.starts_with("//") {
                if let Some(stmt) = parse_statement(line) {
                    func.body.push(stmt);
                }
            }
        }

        i += 1;
    }

    Ok((func, lines.len() - 1))
}

fn extract_func_name(line: &str) -> String {
    // Look for name after .entry or .func
    let after_keyword = line
        .replace(".visible", "")
        .replace(".entry", "|")
        .replace(".func", "|");
    let parts: Vec<&str> = after_keyword.split('|').collect();
    if parts.len() > 1 {
        let name_part = parts[1].trim();
        name_part
            .split(|c: char| c.is_whitespace() || c == '(' || c == '{')
            .next()
            .unwrap_or("unknown")
            .to_string()
    } else {
        "unknown".to_string()
    }
}

fn parse_statement(line: &str) -> Option<PtxStatement> {
    let trimmed = line.trim().trim_end_matches(';').trim();

    // Label
    if trimmed.ends_with(':') && !trimmed.starts_with('@') {
        return Some(PtxStatement::Label(trimmed.trim_end_matches(':').to_string()));
    }

    // Instruction (possibly predicated)
    let (predicate, rest) = if trimmed.starts_with('@') {
        let parts: Vec<&str> = trimmed.splitn(2, char::is_whitespace).collect();
        let pred_str = &parts[0][1..]; // skip @
        let negated = pred_str.starts_with('!');
        let reg = if negated { &pred_str[1..] } else { pred_str }.to_string();
        let rest = parts.get(1).unwrap_or(&"").trim();
        (Some(PtxPredicate { register: reg, negated }), rest.to_string())
    } else {
        (None, trimmed.to_string())
    };

    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.is_empty() { return None; }

    let opcode_full = tokens[0];
    let opcode_parts: Vec<&str> = opcode_full.split('.').collect();
    let opcode = opcode_parts[0].to_string();

    let type_suffix = opcode_parts.iter().skip(1).find_map(|p| parse_type(p));
    let modifiers: Vec<String> = opcode_parts.iter().skip(1)
        .filter(|p| parse_type(p).is_none())
        .map(|s| s.to_string())
        .collect();

    let operand_str = tokens[1..].join(" ");
    let operands = parse_operands(&operand_str);

    Some(PtxStatement::Instruction(PtxInstruction {
        predicate,
        opcode,
        type_suffix,
        modifiers,
        operands,
    }))
}

fn parse_operands(s: &str) -> Vec<PtxOperand> {
    if s.is_empty() { return vec![]; }

    s.split(',')
        .map(|part| {
            let t = part.trim();
            if t.starts_with('%') {
                let name = t.trim_start_matches('%');
                if name.contains("tid.") || name.contains("ctaid.") || name.contains("ntid.")
                    || name.contains("nctaid.") || name == "laneid" || name == "warpid"
                {
                    PtxOperand::SpecialReg(t.to_string())
                } else {
                    PtxOperand::Register(t.to_string())
                }
            } else if t.starts_with('[') && t.ends_with(']') {
                let inner = &t[1..t.len()-1];
                if let Some(plus) = inner.find('+') {
                    let base = inner[..plus].trim().to_string();
                    let offset = inner[plus+1..].trim().parse().ok();
                    PtxOperand::Address { base, offset }
                } else {
                    PtxOperand::Address { base: inner.trim().to_string(), offset: None }
                }
            } else if t.starts_with('{') {
                let inner = t.trim_matches(|c| c == '{' || c == '}');
                let regs: Vec<String> = inner.split(',').map(|r| r.trim().to_string()).collect();
                PtxOperand::Vector(regs)
            } else if let Ok(v) = t.parse::<i64>() {
                PtxOperand::ImmInt(v)
            } else if let Ok(v) = t.parse::<f64>() {
                PtxOperand::ImmFloat(v)
            } else {
                PtxOperand::Label(t.to_string())
            }
        })
        .collect()
}

fn ptx_type_to_ast(ty: &PtxType) -> crate::parser::ast::Type {
    use crate::parser::ast::{Type, IntType, FloatType};
    match ty {
        PtxType::Pred => Type::Bool,
        PtxType::B8 | PtxType::U8 => Type::Int(IntType::U8),
        PtxType::B16 | PtxType::U16 => Type::Int(IntType::U16),
        PtxType::B32 | PtxType::U32 => Type::Int(IntType::U32),
        PtxType::B64 | PtxType::U64 => Type::Int(IntType::U64),
        PtxType::S8 => Type::Int(IntType::I8),
        PtxType::S16 => Type::Int(IntType::I16),
        PtxType::S32 => Type::Int(IntType::I32),
        PtxType::S64 => Type::Int(IntType::I64),
        PtxType::F16 => Type::Float(FloatType::F16),
        PtxType::F32 => Type::Float(FloatType::F32),
        PtxType::F64 => Type::Float(FloatType::F64),
    }
}

fn ptx_body_to_ast(stmts: &[PtxStatement]) -> Result<crate::parser::ast::Block> {
    use crate::parser::ast::*;

    let mut statements = Vec::new();

    for stmt in stmts {
        match stmt {
            PtxStatement::Label(_) => {
                // Labels are used for control flow; skip in high-level AST
            }
            PtxStatement::Instruction(inst) => {
                let ast_stmt = ptx_instruction_to_ast(inst)?;
                if let Some(s) = ast_stmt {
                    statements.push(s);
                }
            }
        }
    }

    Ok(Block { statements })
}

fn ptx_instruction_to_ast(inst: &PtxInstruction) -> Result<Option<crate::parser::ast::Statement>> {
    use crate::parser::ast::*;

    match inst.opcode.as_str() {
        "ret" => Ok(Some(Statement::Return(None))),
        "bar" => Ok(Some(Statement::SyncThreads)),
        "add" | "sub" | "mul" | "div" | "rem" | "and" | "or" | "xor" | "shl" | "shr" => {
            if inst.operands.len() >= 3 {
                let dst = operand_to_var(&inst.operands[0]);
                let lhs = operand_to_expr(&inst.operands[1]);
                let rhs = operand_to_expr(&inst.operands[2]);
                let op = match inst.opcode.as_str() {
                    "add" => BinaryOp::Add, "sub" => BinaryOp::Sub,
                    "mul" => BinaryOp::Mul, "div" => BinaryOp::Div,
                    "rem" => BinaryOp::Mod, "and" => BinaryOp::And,
                    "or" => BinaryOp::Or, "xor" => BinaryOp::Xor,
                    "shl" => BinaryOp::Shl, "shr" => BinaryOp::Shr,
                    _ => BinaryOp::Add,
                };
                Ok(Some(Statement::Expr(Expression::Binary {
                    op: BinaryOp::Assign,
                    left: Box::new(Expression::Var(dst)),
                    right: Box::new(Expression::Binary {
                        op,
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    }),
                })))
            } else {
                Ok(None)
            }
        }
        "mov" => {
            if inst.operands.len() >= 2 {
                let dst = operand_to_var(&inst.operands[0]);
                let src = operand_to_expr(&inst.operands[1]);
                Ok(Some(Statement::Expr(Expression::Binary {
                    op: BinaryOp::Assign,
                    left: Box::new(Expression::Var(dst)),
                    right: Box::new(src),
                })))
            } else {
                Ok(None)
            }
        }
        "ld" => {
            if inst.operands.len() >= 2 {
                let dst = operand_to_var(&inst.operands[0]);
                let src = operand_to_expr(&inst.operands[1]);
                Ok(Some(Statement::Expr(Expression::Binary {
                    op: BinaryOp::Assign,
                    left: Box::new(Expression::Var(dst)),
                    right: Box::new(src),
                })))
            } else {
                Ok(None)
            }
        }
        "st" => {
            if inst.operands.len() >= 2 {
                let dst = operand_to_expr(&inst.operands[0]);
                let src = operand_to_expr(&inst.operands[1]);
                Ok(Some(Statement::Expr(Expression::Binary {
                    op: BinaryOp::Assign,
                    left: Box::new(dst),
                    right: Box::new(src),
                })))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None), // Skip unhandled opcodes
    }
}

fn operand_to_var(op: &PtxOperand) -> String {
    match op {
        PtxOperand::Register(r) => clean_name(r),
        PtxOperand::SpecialReg(r) => clean_name(r),
        PtxOperand::Label(l) => l.clone(),
        _ => "unknown".to_string(),
    }
}

fn operand_to_expr(op: &PtxOperand) -> crate::parser::ast::Expression {
    use crate::parser::ast::*;
    match op {
        PtxOperand::Register(r) => Expression::Var(clean_name(r)),
        PtxOperand::SpecialReg(r) => {
            let name = r.trim_start_matches('%');
            match name {
                "tid.x" => Expression::ThreadIdx(Dimension::X),
                "tid.y" => Expression::ThreadIdx(Dimension::Y),
                "tid.z" => Expression::ThreadIdx(Dimension::Z),
                "ctaid.x" => Expression::BlockIdx(Dimension::X),
                "ctaid.y" => Expression::BlockIdx(Dimension::Y),
                "ctaid.z" => Expression::BlockIdx(Dimension::Z),
                "ntid.x" => Expression::BlockDim(Dimension::X),
                "ntid.y" => Expression::BlockDim(Dimension::Y),
                "ntid.z" => Expression::BlockDim(Dimension::Z),
                "nctaid.x" => Expression::GridDim(Dimension::X),
                "nctaid.y" => Expression::GridDim(Dimension::Y),
                "nctaid.z" => Expression::GridDim(Dimension::Z),
                _ => Expression::Var(name.to_string()),
            }
        }
        PtxOperand::ImmInt(v) => Expression::Literal(Literal::Int(*v)),
        PtxOperand::ImmFloat(v) => Expression::Literal(Literal::Float(*v)),
        PtxOperand::Address { base, offset } => {
            let base_expr = Expression::Var(clean_name(base));
            match offset {
                Some(off) => Expression::Index {
                    array: Box::new(base_expr),
                    index: Box::new(Expression::Literal(Literal::Int(*off))),
                },
                None => base_expr,
            }
        }
        PtxOperand::Label(l) => Expression::Var(l.clone()),
        PtxOperand::Vector(regs) => {
            // Return first register as a simple expression
            Expression::Var(clean_name(regs.first().map(|s| s.as_str()).unwrap_or("v0")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_version() {
        let ptx = ".version 7.8\n.target sm_80\n.address_size 64\n";
        let module = parse_ptx(ptx).unwrap();
        assert_eq!(module.version, "7.8");
        assert_eq!(module.target, "sm_80");
        assert_eq!(module.address_size, 64);
    }

    #[test]
    fn test_parse_entry_function() {
        let ptx = r#"
.version 7.8
.target sm_80
.address_size 64

.visible .entry vectorAdd(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c
)
{
    .reg .f32 %f<4>;
    .reg .u32 %r<4>;
    mov.u32 %r0, %tid.x;
    add.f32 %f2, %f0, %f1;
    ret;
}
"#;
        let module = parse_ptx(ptx).unwrap();
        assert_eq!(module.directives.len(), 1);
        match &module.directives[0] {
            PtxDirective::Entry(func) => {
                assert_eq!(func.name, "vectorAdd");
                assert_eq!(func.params.len(), 3);
                assert!(func.is_entry);
                assert!(!func.body.is_empty());
            }
            other => panic!("Expected entry directive, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_type() {
        assert_eq!(parse_type(".f32"), Some(PtxType::F32));
        assert_eq!(parse_type(".s64"), Some(PtxType::S64));
        assert_eq!(parse_type(".pred"), Some(PtxType::Pred));
        assert_eq!(parse_type(".b16"), Some(PtxType::B16));
        assert_eq!(parse_type(".invalid"), None);
    }

    #[test]
    fn test_parse_instruction_basic() {
        let stmt = parse_statement("add.f32 %f2, %f0, %f1;").unwrap();
        match stmt {
            PtxStatement::Instruction(inst) => {
                assert_eq!(inst.opcode, "add");
                assert_eq!(inst.type_suffix, Some(PtxType::F32));
                assert_eq!(inst.operands.len(), 3);
            }
            other => panic!("Expected instruction, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_predicated_instruction() {
        let stmt = parse_statement("@p0 bra LOOP;").unwrap();
        match stmt {
            PtxStatement::Instruction(inst) => {
                assert!(inst.predicate.is_some());
                let pred = inst.predicate.unwrap();
                assert_eq!(pred.register, "p0");
                assert!(!pred.negated);
                assert_eq!(inst.opcode, "bra");
            }
            other => panic!("Expected instruction, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_negated_predicate() {
        let stmt = parse_statement("@!p1 ret;").unwrap();
        match stmt {
            PtxStatement::Instruction(inst) => {
                let pred = inst.predicate.unwrap();
                assert_eq!(pred.register, "p1");
                assert!(pred.negated);
            }
            other => panic!("Expected instruction, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_label() {
        let stmt = parse_statement("LOOP:").unwrap();
        match stmt {
            PtxStatement::Label(name) => assert_eq!(name, "LOOP"),
            other => panic!("Expected label, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_special_registers() {
        let operands = parse_operands("%tid.x, %ctaid.y");
        assert_eq!(operands.len(), 2);
        match &operands[0] {
            PtxOperand::SpecialReg(r) => assert_eq!(r, "%tid.x"),
            other => panic!("Expected special register, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_memory_address() {
        let operands = parse_operands("[%r0+4]");
        match &operands[0] {
            PtxOperand::Address { base, offset } => {
                assert_eq!(base, "%r0");
                assert_eq!(*offset, Some(4));
            }
            other => panic!("Expected address, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_immediate() {
        let operands = parse_operands("42");
        match &operands[0] {
            PtxOperand::ImmInt(v) => assert_eq!(*v, 42),
            other => panic!("Expected immediate int, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_global_variable() {
        let ptx = ".version 7.8\n.target sm_80\n.address_size 64\n.global .f32 result;\n";
        let module = parse_ptx(ptx).unwrap();
        assert_eq!(module.directives.len(), 1);
        match &module.directives[0] {
            PtxDirective::GlobalVar(var) => {
                assert_eq!(var.var_type, PtxType::F32);
                assert_eq!(var.space, PtxSpace::Global);
            }
            other => panic!("Expected global var, got {:?}", other),
        }
    }

    #[test]
    fn test_ptx_to_ast() {
        let ptx = r#"
.version 7.8
.target sm_80
.address_size 64

.visible .entry simple(
    .param .u64 data
)
{
    .reg .u32 %r<2>;
    mov.u32 %r0, %tid.x;
    ret;
}
"#;
        let module = parse_ptx(ptx).unwrap();
        let ast = ptx_to_ast(&module).unwrap();
        assert_eq!(ast.items.len(), 1);
        match &ast.items[0] {
            crate::parser::ast::Item::Kernel(k) => {
                assert_eq!(k.name, "simple");
            }
            other => panic!("Expected kernel, got {:?}", other),
        }
    }

    #[test]
    fn test_ptx_type_conversion() {
        use crate::parser::ast::{Type, IntType, FloatType};
        assert!(matches!(ptx_type_to_ast(&PtxType::F32), Type::Float(FloatType::F32)));
        assert!(matches!(ptx_type_to_ast(&PtxType::S32), Type::Int(IntType::I32)));
        assert!(matches!(ptx_type_to_ast(&PtxType::Pred), Type::Bool));
    }
}
