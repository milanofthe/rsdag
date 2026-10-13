//! No number drifts back into the code: every tuning value, threshold,
//! size limit and contract value of the workspace has a name in a
//! constants file (`src/constants.rs`, or `src/constants/` of a crate).
//!
//! The test reads every source file of the crates (not their tests,
//! benches and examples) with a small lexer of its own and reports each
//! numeric literal outside the constants files, unless
//!
//! - its value is `0`, `1` or `2` (`0.0`, `0.5`, `1.0`, `2.0`), suffixed
//!   or negated alike;
//! - it sits where the language wants a number of the code's shape: a
//!   tuple field (`x.3`), an index (`v[3]`, `v[..3]`), the length of an
//!   array type (`[f64; 4]`), a pattern (`3 => ...`, `matches!(c, 1 | 2)`);
//! - it is in test code (`#[cfg(test)]`, `#[test]`);
//! - its item is on [`EXEMPT`], with the reason: instruction encodings, ABI
//!   tables, published coefficients, hash constants, tag bits;
//! - its line carries a `// lit: <reason>` comment, or the line above is
//!   one.
//!
//! The contract values ([`CONTRACT`]) are held tighter: outside the
//! constants files they appear nowhere, exempt code included, so a backend
//! cannot carry a copy of them.
//!
//! `constants_guard.baseline`, next to this file, lists what is left from
//! before the guard as `path :: item :: literal`; it may only shrink. A
//! literal not on it fails the test with its line; an entry no longer found
//! fails it too, and `CONSTANTS_GUARD_BLESS=1` drops such entries.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Items whose literals are definitions rather than knobs, by file and
/// item path (`Type::method`, a `mod`, a `const`; `*` the whole file),
/// with the items of such a file that are held to the rule all the same.
const EXEMPT: &[(&str, &[&str], &[&str], &str)] = &[
    (
        "rsdag-jit/src/x86_64.rs",
        &["*"],
        &["X64::exp_inline", "X64::ln_inline"],
        "x86-64 instruction encodings, registers and ABI frame sizes",
    ),
    (
        "rsdag-jit/src/aarch64.rs",
        &["*"],
        &[],
        "AArch64 instruction encodings, registers and ABI",
    ),
    (
        "rsdag-jit/src/native.rs",
        &[
            "Mapping::new::MEM_COMMIT_RESERVE",
            "Mapping::new::PAGE_READWRITE",
            "Mapping::new::PAGE_EXECUTE_READ",
            "Mapping::drop::MEM_RELEASE",
        ],
        &[],
        "Win32's memory allocation and protection flags",
    ),
    (
        "rsdag-jit/src/host.rs",
        &["reduce_code"],
        &[],
        "the op codes of the host's reduce routine, an ABI with the emitted code",
    ),
    (
        "rsdag/src/semantics.rs",
        &[
            "MAGIC", "LN2_HI", "LN2_LO", "EP1", "EP2", "EP3", "EP4", "EP5", "LG1", "LG2", "LG3",
            "LG4", "LG5", "LG6", "LG7",
        ],
        &[],
        "fdlibm's published exp and ln coefficients",
    ),
    (
        "rsdag/src/semantics.rs",
        &["digamma", "trigamma"],
        &[],
        "the recurrence threshold and Bernoulli coefficients of the asymptotic series",
    ),
    (
        "rsdag/src/semantics.rs",
        &["splitmix64", "unit", "rand_stream"],
        &[],
        "splitmix64's constants, and the 53 bits of a double's mantissa",
    ),
    (
        "rsdag/src/node.rs",
        &["shape"],
        &[],
        "splitmix64's constants of the structural hash",
    ),
    (
        "rsdag/src/tape/mod.rs",
        &[
            "INPUT",
            "Fold::NEG",
            "Fold::KIND",
            "Fold::SELF",
            "Fold::SELF_SHIFT",
        ],
        &[],
        "tag bits: an operand's input bit, a fold code's fields",
    ),
    (
        "rsdag/src/graph/mod.rs",
        &["BOUND"],
        &[],
        "the tag bit of a bound symbol",
    ),
    (
        "rsdag/src/role.rs",
        &["ParamRole::place"],
        &[],
        "the rank of each role's group in a signature, an enumeration",
    ),
    (
        "rsdag/src/synth.rs",
        &["*"],
        &[],
        "the synthetic program generator (test support behind feature synth)",
    ),
];

/// The contract values: the semantics every backend reproduces to the bit
/// (`rsdag::constants::contract`). No literal of one of these values
/// appears outside the constants files.
const CONTRACT: &[(&str, Contract)] = &[
    ("EXP_LIMIT", Contract::Float(80.0)),
    ("LN_FLOOR", Contract::Float(1e-30)),
    ("EXP_UNDERFLOW", Contract::Float(-745.2)),
    ("LN_HIGH_FRAC_MASK", Contract::Int(0x000f_ffff)),
    ("LN_SQRT2_CARRY", Contract::Int(0x9_5f64)),
    ("LN_HIGH_UNIT", Contract::Int(0x10_0000)),
    ("LN_HIGH_ONE", Contract::Int(0x3ff0_0000)),
    ("F64_EXP_BIAS", Contract::Int(1023)),
];

/// A contract value: a float matches float literals, an integer integer
/// ones, either sign.
#[derive(Clone, Copy)]
enum Contract {
    Float(f64),
    Int(u64),
}

impl Contract {
    fn matches(self, text: &str) -> bool {
        let v = value(text);
        match self {
            Contract::Float(c) => is_float(text) && (c == v || c == -v),
            Contract::Int(c) => !is_float(text) && c as f64 == v,
        }
    }
}

/// Whether a literal is a float: a fraction, an exponent or a float suffix.
fn is_float(text: &str) -> bool {
    let hex = text.starts_with("0x") || text.starts_with("0b") || text.starts_with("0o");
    !hex && (text.contains(['.', 'e', 'E']) || text.ends_with("f64") || text.ends_with("f32"))
}

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn baseline_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/constants_guard.baseline")
}

/// Every source file of every crate, as `crate/src/...`.
fn sources() -> Vec<(String, PathBuf)> {
    fn walk(dir: &Path, rel: &str, out: &mut Vec<(String, PathBuf)>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .map(|e| e.unwrap().path())
            .collect();
        entries.sort();
        for p in entries {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            let r = format!("{rel}/{name}");
            if p.is_dir() {
                walk(&p, &r, out);
            } else if name.ends_with(".rs") && name != "tests.rs" && !name.ends_with("_tests.rs") {
                out.push((r, p));
            }
        }
    }
    let mut out = Vec::new();
    let mut crates: Vec<_> = std::fs::read_dir(crates_dir())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("src").is_dir())
        .collect();
    crates.sort();
    for c in crates {
        let name = c.file_name().unwrap().to_string_lossy().to_string();
        walk(&c.join("src"), &format!("{name}/src"), &mut out);
    }
    out
}

fn is_constants_file(rel: &str) -> bool {
    rel.ends_with("/constants.rs") || rel.contains("/constants/")
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    /// A numeric literal: its text, and whether it is a tuple field.
    Num(String, bool),
    Punct(&'static str),
    /// A string, a character or a lifetime: nothing to look into.
    Other,
}

#[derive(Clone, Debug)]
struct Token {
    tok: Tok,
    line: usize,
}

const PUNCTS: &[&str] = &[
    "..=", "...", "::", "..", "=>", "->", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=",
    "/=", "%=", "^=", "&=", "|=", "<<", ">>", "(", ")", "[", "]", "{", "}", ";", ",", ".", "#",
    "!", "<", ">", ":", "=", "+", "-", "*", "/", "%", "^", "&", "|", "?", "@", "$", "~",
];

/// The tokens of `src`, and the lines whose comment says `lit:`.
fn lex(src: &str) -> (Vec<Token>, Vec<usize>) {
    let b = src.as_bytes();
    let (mut i, mut line) = (0, 1);
    let mut out: Vec<Token> = Vec::new();
    let mut lit_lines = Vec::new();
    let ident_char = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    while i < b.len() {
        let c = b[i];
        if c == b'\n' {
            line += 1;
            i += 1;
        } else if c.is_ascii_whitespace() {
            i += 1;
        } else if src[i..].starts_with("//") {
            let end = src[i..].find('\n').map_or(b.len(), |e| i + e);
            if src[i..end]
                .trim_start_matches('/')
                .trim_start()
                .starts_with("lit:")
                && !src[i..end].trim_start_matches('/').trim()["lit:".len()..]
                    .trim()
                    .is_empty()
            {
                // On a line of its own, the comment covers the next one.
                let own_line = src[..i].rsplit('\n').next().unwrap_or("").trim().is_empty();
                lit_lines.push(if own_line { line + 1 } else { line });
            }
            i = end;
        } else if src[i..].starts_with("/*") {
            let mut depth = 0;
            while i < b.len() {
                if src[i..].starts_with("/*") {
                    depth += 1;
                    i += 2;
                } else if src[i..].starts_with("*/") {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    if b[i] == b'\n' {
                        line += 1;
                    }
                    i += 1;
                }
            }
        } else if let Some(n) = raw_string_start(&src[i..]) {
            // r"..", r#".."#, br"..", cr".."
            let hashes = &src[i + n - 1 - count_hashes(&src[i..i + n - 1])..i + n - 1];
            let close = format!("\"{hashes}");
            let end = src[i + n..]
                .find(&close)
                .map_or(b.len(), |e| i + n + e + close.len());
            line += src[i..end].matches('\n').count();
            i = end;
            out.push(Token {
                tok: Tok::Other,
                line,
            });
        } else if c == b'"' || ((c == b'b' || c == b'c') && b.get(i + 1) == Some(&b'"')) {
            i += if c == b'"' { 1 } else { 2 };
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' {
                    i += 1;
                }
                if i < b.len() && b[i] == b'\n' {
                    line += 1;
                }
                i += 1;
            }
            i += 1;
            out.push(Token {
                tok: Tok::Other,
                line,
            });
        } else if c == b'\'' {
            // A character ('a', '\n', '\u{..}') or a lifetime ('a, 'static).
            if b.get(i + 1) == Some(&b'\\') {
                i += 2;
                while i < b.len() && b[i] != b'\'' {
                    i += 1;
                }
                i += 1;
            } else {
                let ch_len = src[i + 1..].chars().next().map_or(1, char::len_utf8);
                if b.get(i + 1 + ch_len) == Some(&b'\'') {
                    i += 2 + ch_len;
                } else {
                    i += 1;
                    while i < b.len() && ident_char(b[i]) {
                        i += 1;
                    }
                }
            }
            out.push(Token {
                tok: Tok::Other,
                line,
            });
        } else if c.is_ascii_digit() {
            let after_dot = matches!(
                out.last(),
                Some(Token {
                    tok: Tok::Punct("."),
                    ..
                })
            );
            let start = i;
            if after_dot {
                // A tuple field: digits only (`x.0.1` is two fields).
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            } else if c == b'0' && matches!(b.get(i + 1), Some(b'x' | b'o' | b'b')) {
                i += 2;
                while i < b.len() && (b[i].is_ascii_hexdigit() || b[i] == b'_') {
                    i += 1;
                }
                while i < b.len() && ident_char(b[i]) {
                    i += 1;
                }
            } else {
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') {
                    i += 1;
                }
                // A fraction: `.` then a digit, or `.` then nothing of a
                // range, a field or a method (`2.` is a float).
                if b.get(i) == Some(&b'.')
                    && b.get(i + 1) != Some(&b'.')
                    && !b
                        .get(i + 1)
                        .is_some_and(|&d| d.is_ascii_alphabetic() || d == b'_')
                {
                    i += 1;
                    while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') {
                        i += 1;
                    }
                }
                if matches!(b.get(i), Some(b'e' | b'E'))
                    && b.get(i + 1)
                        .is_some_and(|&d| d.is_ascii_digit() || d == b'+' || d == b'-')
                {
                    i += 2;
                    while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') {
                        i += 1;
                    }
                }
                while i < b.len() && ident_char(b[i]) {
                    i += 1;
                }
            }
            out.push(Token {
                tok: Tok::Num(src[start..i].to_string(), after_dot),
                line,
            });
        } else if ident_char(c) || c >= 0x80 {
            let start = i;
            if src[i..].starts_with("r#") {
                i += 2;
            }
            while i < b.len() && (ident_char(b[i]) || b[i] >= 0x80) {
                i += 1;
            }
            out.push(Token {
                tok: Tok::Ident(src[start..i].to_string()),
                line,
            });
        } else {
            let p = PUNCTS
                .iter()
                .find(|p| src[i..].starts_with(**p))
                .copied()
                .unwrap_or("~");
            i += p.len().max(1);
            out.push(Token {
                tok: Tok::Punct(p),
                line,
            });
        }
    }
    (out, lit_lines)
}

/// The `#`s of a raw string's prefix.
fn count_hashes(prefix: &str) -> usize {
    prefix.bytes().rev().take_while(|&c| c == b'#').count()
}

/// The length of a raw string's opening (`r#"` is 3), if one starts here.
fn raw_string_start(s: &str) -> Option<usize> {
    let rest = s
        .strip_prefix("br")
        .or_else(|| s.strip_prefix("cr"))
        .or_else(|| s.strip_prefix('r'))?;
    let hashes = rest.bytes().take_while(|&c| c == b'#').count();
    (rest.as_bytes().get(hashes) == Some(&b'"')).then(|| s.len() - rest.len() + hashes + 1)
}

/// The value of a numeric literal, its suffix and separators dropped.
fn value(text: &str) -> f64 {
    let t = text.replace('_', "");
    let (radix, digits) = match t.get(..2) {
        Some("0x") => (16, &t[2..]),
        Some("0o") => (8, &t[2..]),
        Some("0b") => (2, &t[2..]),
        _ => (10, &t[..]),
    };
    if radix != 10 {
        let end = [
            "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128",
        ]
        .iter()
        .chain(&["usize"])
        .find_map(|s| digits.strip_suffix(s))
        .unwrap_or(digits);
        return u128::from_str_radix(end, radix).expect("an integer") as f64;
    }
    let end = digits.find(|c: char| c.is_ascii_alphabetic() && c != 'e' && c != 'E');
    let mut core = &digits[..end.unwrap_or(digits.len())];
    // `1e5f64`: the suffix starts at the `f`, after the exponent.
    if let Some(f) = digits.find(['f', 'u', 'i']) {
        core = &digits[..f];
    }
    core.trim_end_matches('.')
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("a number: {text}"))
}

fn allowed_value(v: f64) -> bool {
    [0.0, 0.5, 1.0, 2.0].contains(&v)
}

const KEYWORDS: &[&str] = &[
    "as", "break", "const", "continue", "crate", "else", "enum", "extern", "fn", "for", "if",
    "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return", "static",
    "struct", "trait", "type", "unsafe", "use", "where", "while", "dyn", "yield",
];

const PRIMITIVES: &[&str] = &[
    "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128",
    "usize", "bool", "char", "str", "mut", "const", "dyn", "Self",
];

/// Whether the tokens of `[ .. ;` (the part before the length) name a type
/// (`[f64; 4]`, `[Vec<u32>; 3]`) rather than a value (`[x; 4]`).
fn is_type(toks: &[Token]) -> bool {
    let mut depth = 0i32;
    for (k, t) in toks.iter().enumerate() {
        match &t.tok {
            Tok::Num(..) if depth == 0 => return false,
            Tok::Ident(id) => {
                if toks
                    .get(k + 1)
                    .is_some_and(|n| n.tok == Tok::Punct("(") || n.tok == Tok::Punct("!"))
                {
                    return false;
                }
                let path = toks.get(k + 1).is_some_and(|n| n.tok == Tok::Punct("::"));
                let first = id.chars().next().unwrap();
                let upper_const = id.len() > 1 && !id.chars().any(|c| c.is_ascii_lowercase());
                let ok = PRIMITIVES.contains(&id.as_str())
                    || path
                    || (first.is_ascii_uppercase() && !upper_const);
                if !ok {
                    return false;
                }
            }
            Tok::Punct("[") => depth += 1,
            Tok::Punct("]") => depth -= 1,
            Tok::Punct(p) => {
                if !["::", "<", ">", "&", "*", ",", "(", ")", ";", ">>"].contains(p) {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
}

/// A literal found where it does not belong.
#[derive(Debug)]
struct Hit {
    line: usize,
    item: String,
    text: String,
    contract: Option<&'static str>,
}

/// The scope a token is in: an item (fn, const, impl, mod ...) and the
/// bracket depth of its keyword.
struct Scope {
    name: String,
    depth: i32,
    body: bool,
}

/// What a bracket was opened for.
#[derive(Clone, Copy, PartialEq)]
enum Bracket {
    Index,
    ArrayType,
    Other,
}

fn scan(rel: &str, src: &str) -> (Vec<Hit>, usize) {
    let (toks, lit_lines) = lex(src);
    let mut hits = Vec::new();
    let mut hatches = 0;
    let mut scopes: Vec<Scope> = Vec::new();
    let mut brackets: Vec<(Bracket, usize)> = Vec::new();
    // Match bodies and `matches!` arguments: (depth inside, in a pattern).
    let mut matches: Vec<(i32, bool, bool)> = Vec::new();
    let mut pending_match: Option<i32> = None;
    let mut depth = 0i32;
    let mut i = 0;
    let is = |k: usize, p: &str| toks.get(k).is_some_and(|t| t.tok == Tok::Punct(leak(p)));
    while i < toks.len() {
        let t = &toks[i];
        let prev = i.checked_sub(1).map(|k| &toks[k].tok);
        // Attributes: skipped, test items with them.
        if t.tok == Tok::Punct("#") && (is(i + 1, "[") || (is(i + 1, "!") && is(i + 2, "["))) {
            let open = if is(i + 1, "[") { i + 1 } else { i + 2 };
            let close = matching(&toks, open);
            let inner = &toks[open + 1..close];
            let name = |k: usize| match inner.get(k).map(|t| &t.tok) {
                Some(Tok::Ident(s)) => s.as_str(),
                _ => "",
            };
            if name(0) == "pyo3" {
                for t in inner {
                    if let Tok::Num(text, _) = &t.tok {
                        if !allowed_value(value(text)) {
                            hits.push(Hit {
                                line: t.line,
                                item: "pyo3 signature".into(),
                                text: text.clone(),
                                contract: None,
                            });
                        }
                    }
                }
            }
            let test_only = name(0) == "test"
                || (name(0) == "cfg"
                    && (name(2) == "test" || (name(2) == "all" && name(4) == "test")));
            i = close + 1;
            if test_only {
                i = item_end(&toks, i);
            }
            continue;
        }
        match &t.tok {
            Tok::Punct(p @ ("(" | "[" | "{")) => {
                if *p == "{" {
                    if let Some(s) = scopes.last_mut() {
                        if !s.body && s.depth == depth {
                            s.body = true;
                        }
                    }
                    if pending_match == Some(depth) {
                        matches.push((depth + 1, true, false));
                        pending_match = None;
                    }
                }
                if *p == "[" {
                    let index = match prev {
                        Some(Tok::Ident(id)) => !KEYWORDS.contains(&id.as_str()),
                        Some(Tok::Punct(q)) => [")", "]", "?"].contains(q),
                        _ => false,
                    };
                    let semi = top_level_semi(&toks, i);
                    let kind = if index {
                        Bracket::Index
                    } else if semi.is_some_and(|s| is_type(&toks[i + 1..s])) {
                        Bracket::ArrayType
                    } else {
                        Bracket::Other
                    };
                    brackets.push((kind, i));
                }
                if *p == "("
                    && matches!(prev, Some(Tok::Punct("!")))
                    && matches!(toks.get(i.wrapping_sub(2)).map(|t| &t.tok), Some(Tok::Ident(m)) if m == "matches")
                {
                    matches.push((depth + 1, false, true));
                }
                depth += 1;
            }
            Tok::Punct(p @ (")" | "]" | "}")) => {
                depth -= 1;
                if *p == "]" {
                    brackets.pop();
                }
                while matches.last().is_some_and(|m| m.0 > depth) {
                    matches.pop();
                }
                // A block that ends an arm (not one an `else` or a method
                // call continues): the next arm's pattern follows.
                let continued = match toks.get(i + 1).map(|t| &t.tok) {
                    Some(Tok::Ident(w)) => w == "else",
                    Some(Tok::Punct(q)) => [".", "?"].contains(q),
                    _ => false,
                };
                if let Some(m) = matches.last_mut() {
                    if *p == "}" && m.0 == depth && !m.2 && !continued {
                        m.1 = true;
                    }
                }
                while scopes
                    .last()
                    .is_some_and(|s| s.depth > depth || (s.depth == depth && s.body && *p == "}"))
                {
                    scopes.pop();
                }
            }
            Tok::Punct(";") => {
                while scopes.last().is_some_and(|s| s.depth == depth && !s.body) {
                    scopes.pop();
                }
            }
            Tok::Punct(",") => {
                if let Some(m) = matches.last_mut() {
                    if m.0 == depth {
                        m.1 = true;
                    }
                }
            }
            Tok::Punct("=>") => {
                if let Some(m) = matches.last_mut() {
                    if m.0 == depth && !m.2 {
                        m.1 = false;
                    }
                }
            }
            Tok::Ident(id) => {
                if id == "match" {
                    pending_match = Some(depth);
                }
                if id == "if" {
                    if let Some(m) = matches.last_mut() {
                        if m.0 == depth && m.1 && !m.2 {
                            m.1 = false;
                        }
                    }
                }
                if let Some(name) = item_name(&toks, i) {
                    scopes.push(Scope {
                        name,
                        depth,
                        body: false,
                    });
                }
            }
            Tok::Num(text, field) => {
                let v = value(text);
                let item = scopes
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::");
                let contract = CONTRACT
                    .iter()
                    .find(|(_, c)| c.matches(text))
                    .map(|(n, _)| *n);
                let structural = *field
                    || in_pattern(&matches, depth)
                    || match brackets.last() {
                        Some(&(Bracket::Index, open)) => {
                            let before = &toks[i - 1].tok;
                            let after = toks.get(i + 1).map(|t| &t.tok);
                            let lo = i - 1 == open
                                || (matches!(before, Tok::Punct(".." | "..=")) && i - 2 == open);
                            let hi = matches!(after, Some(Tok::Punct("]")))
                                || matches!(after, Some(Tok::Punct(".." | "..=")));
                            (i - 1 == open && hi) || (lo && matches!(after, Some(Tok::Punct("]"))))
                        }
                        Some(&(Bracket::ArrayType, _)) => {
                            matches!(prev, Some(Tok::Punct(";")))
                                && matches!(toks.get(i + 1).map(|t| &t.tok), Some(Tok::Punct("]")))
                        }
                        _ => false,
                    };
                let exempt = exempt(rel, &item);
                let hatch = lit_lines.contains(&t.line);
                if contract.is_some() && !is_constants_file(rel) {
                    hits.push(Hit {
                        line: t.line,
                        item,
                        text: text.clone(),
                        contract,
                    });
                } else if !allowed_value(v) && !structural && !exempt && !is_constants_file(rel) {
                    if hatch {
                        hatches += 1;
                    } else {
                        hits.push(Hit {
                            line: t.line,
                            item,
                            text: text.clone(),
                            contract: None,
                        });
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    (hits, hatches)
}

fn leak(p: &str) -> &'static str {
    PUNCTS
        .iter()
        .find(|q| **q == p)
        .copied()
        .expect("a known punctuation")
}

fn in_pattern(matches: &[(i32, bool, bool)], depth: i32) -> bool {
    matches.last().is_some_and(|m| m.1 && depth >= m.0)
}

fn exempt(rel: &str, item: &str) -> bool {
    let under = |pat: &str| pat == "*" || item == pat || item.starts_with(&format!("{pat}::"));
    EXEMPT.iter().any(|(file, items, except, _)| {
        rel.ends_with(file) && items.iter().any(|p| under(p)) && !except.iter().any(|p| under(p))
    })
}

/// The index of the bracket closing the one at `open`.
fn matching(toks: &[Token], open: usize) -> usize {
    let mut depth = 0;
    for (k, t) in toks.iter().enumerate().skip(open) {
        match t.tok {
            Tok::Punct("(" | "[" | "{") => depth += 1,
            Tok::Punct(")" | "]" | "}") => {
                depth -= 1;
                if depth == 0 {
                    return k;
                }
            }
            _ => {}
        }
    }
    toks.len() - 1
}

/// The `;` at the top level of the bracket opened at `open`, if any.
fn top_level_semi(toks: &[Token], open: usize) -> Option<usize> {
    let close = matching(toks, open);
    let mut depth = 0;
    for k in open + 1..close {
        match toks[k].tok {
            Tok::Punct("(" | "[" | "{") => depth += 1,
            Tok::Punct(")" | "]" | "}") => depth -= 1,
            Tok::Punct(";") if depth == 0 => return Some(k),
            _ => {}
        }
    }
    None
}

/// The first token after the item starting at `from` (its attributes,
/// then up to its `;` or its body's closing brace).
fn item_end(toks: &[Token], mut from: usize) -> usize {
    while toks.get(from).is_some_and(|t| t.tok == Tok::Punct("#")) {
        let open = if toks[from + 1].tok == Tok::Punct("[") {
            from + 1
        } else {
            from + 2
        };
        from = matching(toks, open) + 1;
    }
    let mut depth = 0;
    let mut k = from;
    while k < toks.len() {
        match toks[k].tok {
            Tok::Punct("(" | "[") => depth += 1,
            Tok::Punct(")" | "]") => depth -= 1,
            Tok::Punct("{") if depth == 0 => return matching(toks, k) + 1,
            Tok::Punct(";") if depth == 0 => return k + 1,
            _ => {}
        }
        k += 1;
    }
    k
}

/// The name of the item whose keyword is at `k`, if one is: `fn f`,
/// `const X`, `static X`, `mod m`, `trait T`, `struct S`, `enum E`, `impl
/// .. T` (the implementing type), `macro_rules! m`.
fn item_name(toks: &[Token], k: usize) -> Option<String> {
    let Tok::Ident(kw) = &toks[k].tok else {
        return None;
    };
    let ident = |j: usize| match toks.get(j).map(|t| &t.tok) {
        Some(Tok::Ident(s)) => Some(s.clone()),
        _ => None,
    };
    let prev = k.checked_sub(1).map(|j| &toks[j].tok);
    let item_position = match prev {
        None => true,
        Some(Tok::Punct(p)) => [";", "}", "{", "]", ")"].contains(p),
        Some(Tok::Ident(p)) => {
            ["pub", "unsafe", "extern", "async", "default", "const"].contains(&p.as_str())
        }
        Some(Tok::Other) => true,
        _ => false,
    };
    match kw.as_str() {
        "fn" | "mod" | "trait" | "struct" | "enum" | "union" => {
            let name = ident(k + 1)?;
            (item_position || kw == "fn").then_some(name)
        }
        "const" | "static" => {
            if matches!(prev, Some(Tok::Punct("<" | ",")) | Some(Tok::Punct("*"))) {
                return None;
            }
            let j = if ident(k + 1).as_deref() == Some("mut") {
                k + 2
            } else {
                k + 1
            };
            let name = ident(j)?;
            (toks.get(j + 1)?.tok == Tok::Punct(":")).then_some(name)
        }
        "macro_rules" => {
            (toks.get(k + 1)?.tok == Tok::Punct("!")).then_some(())?;
            ident(k + 2)
        }
        "impl" if item_position => {
            // The last name at the top level before the body or `where`.
            let (mut angle, mut name) = (0, None);
            for t in &toks[k + 1..] {
                match &t.tok {
                    Tok::Punct("<") => angle += 1,
                    Tok::Punct(">") => angle -= 1,
                    Tok::Punct(">>") => angle -= 2,
                    Tok::Punct("{") | Tok::Punct(";") => break,
                    Tok::Ident(w) if w == "where" => break,
                    Tok::Ident(s) if angle == 0 && s != "for" && s != "dyn" => {
                        name = Some(s.clone())
                    }
                    _ => {}
                }
            }
            name
        }
        _ => None,
    }
}

/// Every hit of the workspace, as `path :: item :: literal` with its line.
fn hits() -> (Vec<(String, usize, Hit)>, BTreeMap<String, usize>) {
    let mut all = Vec::new();
    let mut hatches = BTreeMap::new();
    for (rel, path) in sources() {
        let src = std::fs::read_to_string(&path).unwrap();
        let (hits, n) = scan(&rel, &src);
        if n > 0 {
            hatches.insert(rel.clone(), n);
        }
        for h in hits {
            let key = format!(
                "{rel} :: {} :: {}",
                if h.item.is_empty() { "-" } else { &h.item },
                h.text
            );
            all.push((key, h.line, h));
        }
    }
    (all, hatches)
}

#[test]
fn every_number_has_a_name() {
    let (all, hatches) = hits();
    for (file, n) in &hatches {
        println!("{file}: {n} literal(s) under `// lit:`");
    }
    let text = std::fs::read_to_string(baseline_path()).unwrap_or_default();
    let mut left: BTreeMap<String, usize> = BTreeMap::new();
    for l in text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        *left.entry(l.to_string()).or_default() += 1;
    }
    let mut kept: BTreeMap<String, usize> = BTreeMap::new();
    let mut fresh = Vec::new();
    for (key, line, h) in &all {
        let in_baseline = left.get(key).is_some_and(|&n| n > 0);
        if in_baseline {
            *left.get_mut(key).unwrap() -= 1;
            *kept.entry(key.clone()).or_default() += 1;
        } else {
            let (path, rest) = key.split_once(" :: ").unwrap();
            let why = match h.contract {
                Some(c) => format!("the contract value of `{c}`: use the constant"),
                None => "is not named (move it to the crate's constants file, or mark it `// lit: <reason>`)".into(),
            };
            fresh.push(format!(
                "crates/{path}:{line}: literal {} in {} {why}",
                h.text,
                rest.rsplit_once(" :: ").unwrap().0
            ));
        }
    }
    let stale: Vec<_> = left.iter().filter(|(_, &n)| n > 0).collect();
    if !stale.is_empty() && std::env::var_os("CONSTANTS_GUARD_BLESS").is_some() {
        let mut out = String::new();
        for l in text
            .lines()
            .filter(|l| l.trim().is_empty() || l.starts_with('#'))
        {
            out += l;
            out += "\n";
        }
        for (key, n) in &kept {
            for _ in 0..*n {
                out += key;
                out += "\n";
            }
        }
        std::fs::write(baseline_path(), out).unwrap();
    } else if !stale.is_empty() {
        let list: Vec<String> = stale.iter().map(|(k, n)| format!("  {k} (x{n})")).collect();
        panic!(
            "constants_guard.baseline lists literals that are gone; delete them (or run with CONSTANTS_GUARD_BLESS=1):\n{}",
            list.join("\n")
        );
    }
    assert!(
        fresh.is_empty(),
        "{} unnamed literal(s):\n{}",
        fresh.len(),
        fresh.join("\n")
    );
}

/// The lexer and the rules on a sample of everything they tell apart.
#[test]
fn the_guard_reads_rust() {
    let src = r##"
        // 77 in a comment, /* 78 */ and "79" and r#"80"# and '9' are no numbers.
        /* nested /* 81 */ */
        #[repr(align(64))]
        fn f(x: [f64; 3], t: (u8, u8)) -> u32 {
            let a = x[3] + x[..3].len() as f64 + t.1 as f64;
            let b = 0.5 * 2.0 - 1.0 + 0usize as f64 + -1i32 as f64;
            let c = match t.0 { 3 => 1, 4..=6 => 2, n if n > 40 => 0, _ => 0 };
            let d = matches!(t.0, 7 | 8);
            let e = [0u8; 16];
            let s = 'a: loop { break 'a 5; };
            let g = 1e3; // lit: a sample
            // lit: a sample on the line above
            let h = 1e4;
            42
        }
        #[cfg(test)]
        mod tests { const X: u32 = 43; }
        impl Foo<u32> for Bar { const K: usize = 44; fn m() { 45; } }
        const C: f64 = -745.2;
    "##;
    let (hits, hatches) = scan("x/src/lib.rs", src);
    let found: Vec<(String, String)> = hits
        .iter()
        .map(|h| (h.item.clone(), h.text.clone()))
        .collect();
    let want = [
        ("f", "40"),
        ("f", "16"),
        ("f", "5"),
        ("f", "42"),
        ("Bar::K", "44"),
        ("Bar::m", "45"),
        ("C", "745.2"),
    ];
    let want: Vec<(String, String)> = want
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    assert_eq!(found, want);
    assert_eq!(hatches, 2);
    assert_eq!(hits.last().unwrap().contract, Some("EXP_UNDERFLOW"));
}
