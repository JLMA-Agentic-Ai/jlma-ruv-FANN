# ADR-004: Unified Error Type Hierarchy

## Status: Accepted

## Context
5+ error types coexist: `NetworkError`, `TrainingError`, `CascadeError`, `RuvFannError`,
`ComputeError`, `IoError`, plus `&'static str` and `String` returns. Consumers must
understand which error type each method returns.

## Decision
1. `RuvFannError` becomes the single public error type
2. Remove standalone `NetworkError`, `TrainingError` from public API
3. Remove unused `ErrorContext`, `RecoveryStrategy`, `RecoveryContext`
4. All public methods return `Result<_, RuvFannError>`
5. Replace `&'static str` returns in `Layer::set_inputs()`, `Neuron::set_connection_weight()`

## Consequences
- Simpler, consistent error handling for consumers
- Removes dead code (~200 lines in errors.rs)
- Breaking change for callers matching on specific error types
