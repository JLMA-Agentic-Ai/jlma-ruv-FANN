# ADR-001: Bounded Deserialization for All I/O Paths

## Status: Accepted

## Context
The deep review identified CRITICAL deserialization vulnerabilities (F-01, F-02, F-03, F-10):
- `read_binary()`, `read_json()`, `from_bytes()` have no size limits
- `decompress_bytes()` is vulnerable to decompression bombs
- Crafted payloads can OOM-crash the process

## Decision
1. All deserialization functions MUST accept a max size parameter
2. `bincode::DefaultOptions::new().with_limit(MAX_SIZE)` for binary
3. `reader.take(max_bytes)` for JSON and streaming readers
4. Decompression with chunked reads and size tracking
5. `Network::from_bytes()` must validate structure post-deserialization
6. Default max size: 256MB for model files, 1GB for training data

## Consequences
- Breaking API change: `read_binary`, `read_json` gain a `max_size` parameter
- Existing callers must update
- Prevents OOM from untrusted model files
