//! CUDA lexer using logos for tokenization

use logos::Logos;

/// Token types for CUDA source code
#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\r\n\f]+")]
pub enum Token {
    // ── Keywords ──────────────────────────────────────────────
    #[token("__global__")]
    Global,
    #[token("__device__")]
    Device,
    #[token("__host__")]
    Host,
    #[token("__shared__")]
    Shared,
    #[token("__constant__")]
    Constant,
    #[token("extern")]
    Extern,
    #[token("void")]
    Void,
    #[token("int")]
    Int,
    #[token("unsigned")]
    Unsigned,
    #[token("float")]
    Float,
    #[token("double")]
    Double,
    #[token("char")]
    Char,
    #[token("short")]
    Short,
    #[token("long")]
    Long,
    #[token("bool")]
    Bool,
    #[token("const")]
    Const,
    #[token("volatile")]
    Volatile,
    #[token("__restrict__")]
    Restrict,
    #[token("restrict")]
    RestrictC,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("for")]
    For,
    #[token("while")]
    While,
    #[token("do")]
    Do,
    #[token("return")]
    Return,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("struct")]
    Struct,
    #[token("typedef")]
    Typedef,
    #[token("sizeof")]
    Sizeof,
    #[token("register")]
    Register,
    #[token("static")]
    Static,
    #[token("inline")]
    Inline,
    #[token("__inline__")]
    InlineAlt,
    #[token("__forceinline__")]
    ForceInline,

    // ── CUDA builtins ────────────────────────────────────────
    #[token("threadIdx")]
    ThreadIdx,
    #[token("blockIdx")]
    BlockIdx,
    #[token("blockDim")]
    BlockDim,
    #[token("gridDim")]
    GridDim,
    #[token("__syncthreads")]
    SyncThreads,

    // ── Vector types ─────────────────────────────────────────
    #[token("float2")]
    Float2,
    #[token("float3")]
    Float3,
    #[token("float4")]
    Float4,
    #[token("int2")]
    Int2,
    #[token("int3")]
    Int3,
    #[token("int4")]
    Int4,
    #[token("double2")]
    Double2,
    #[token("double3")]
    Double3,
    #[token("double4")]
    Double4,
    #[token("dim3")]
    Dim3,

    // ── Literals ─────────────────────────────────────────────
    #[regex(r"0[xX][0-9a-fA-F]+[uUlL]*", |lex| lex.slice().to_string())]
    HexLiteral(String),
    #[regex(r"[0-9]+\.[0-9]*([eE][+-]?[0-9]+)?[fF]?", |lex| lex.slice().to_string())]
    FloatLiteral(String),
    #[regex(r"\.[0-9]+([eE][+-]?[0-9]+)?[fF]?", |lex| lex.slice().to_string())]
    FloatLiteralDot(String),
    #[regex(r"[0-9]+[eE][+-]?[0-9]+[fF]?", |lex| lex.slice().to_string())]
    FloatLiteralExp(String),
    #[regex(r"[0-9]+[fF]", |lex| lex.slice().to_string())]
    FloatLiteralSuffix(String),
    #[regex(r"[0-9]+[uUlL]*", priority = 2, callback = |lex| lex.slice().to_string())]
    IntLiteral(String),
    #[regex(r#""([^"\\]|\\.)*""#, |lex| lex.slice().to_string())]
    StringLiteral(String),
    #[regex(r"'([^'\\]|\\.)'", |lex| lex.slice().to_string())]
    CharLiteral(String),

    // ── Identifiers ──────────────────────────────────────────
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", priority = 1, callback = |lex| lex.slice().to_string())]
    Ident(String),

    // ── Operators ────────────────────────────────────────────
    #[token("+=")]
    PlusAssign,
    #[token("-=")]
    MinusAssign,
    #[token("*=")]
    StarAssign,
    #[token("/=")]
    SlashAssign,
    #[token("%=")]
    PercentAssign,
    #[token("&=")]
    AmpAssign,
    #[token("|=")]
    PipeAssign,
    #[token("^=")]
    CaretAssign,
    #[token("<<=")]
    ShlAssign,
    #[token(">>=")]
    ShrAssign,
    #[token("++")]
    PlusPlus,
    #[token("--")]
    MinusMinus,
    #[token("&&")]
    AmpAmp,
    #[token("||")]
    PipePipe,
    #[token("==")]
    EqEq,
    #[token("!=")]
    BangEq,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,
    #[token("<<")]
    Shl,
    #[token(">>")]
    Shr,
    #[token("->")]
    Arrow,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("&")]
    Amp,
    #[token("|")]
    Pipe,
    #[token("^")]
    Caret,
    #[token("~")]
    Tilde,
    #[token("!")]
    Bang,
    #[token("=")]
    Eq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("?")]
    Question,
    #[token(":")]
    Colon,

    // ── Delimiters ───────────────────────────────────────────
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(";")]
    Semi,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,

    // ── Preprocessor & comments ──────────────────────────────
    #[regex(r#"#include\s*[<"][^>"\n]+[>"]"#, |lex| lex.slice().to_string())]
    Include(String),
    #[regex(r"#define\s+[^\n]+", |lex| lex.slice().to_string())]
    Define(String),
    #[regex(r"#(pragma|ifdef|ifndef|endif|if|elif|else|undef|error|warning)[^\n]*", |lex| lex.slice().to_string())]
    Preprocessor(String),
    #[regex(r"//[^\n]*")]
    LineComment,
    #[regex(r"/\*([^*]|\*[^/])*\*/")]
    BlockComment,
}

/// A token with its span information
#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub span: std::ops::Range<usize>,
    pub text: String,
}

/// Tokenize CUDA source code, stripping comments and returning spanned tokens
pub fn tokenize(source: &str) -> Vec<SpannedToken> {
    let mut tokens = Vec::new();
    let lex = Token::lexer(source);
    for (result, span) in lex.spanned() {
        match result {
            Ok(tok) => {
                // Skip comments
                if matches!(tok, Token::LineComment | Token::BlockComment) {
                    continue;
                }
                tokens.push(SpannedToken {
                    token: tok,
                    span: span.clone(),
                    text: source[span].to_string(),
                });
            }
            Err(_) => {
                // Skip unrecognized bytes
            }
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenize() {
        let src = "__global__ void vectorAdd(float* a, int n) { }";
        let tokens = tokenize(src);
        assert!(tokens.iter().any(|t| matches!(&t.token, Token::Global)));
        assert!(tokens.iter().any(|t| matches!(&t.token, Token::Void)));
    }

    #[test]
    fn test_comments_stripped() {
        let src = "int x; // comment\n/* block */ float y;";
        let tokens = tokenize(src);
        assert!(!tokens.iter().any(|t| matches!(&t.token, Token::LineComment)));
        assert!(!tokens.iter().any(|t| matches!(&t.token, Token::BlockComment)));
    }
}
