# ADR-002: Make Struct Fields Private with Accessor Methods

## Status: Accepted

## Context
All fields on `Network`, `Layer`, `Neuron`, `Connection` are `pub`, exposing internal
representation and preventing invariant enforcement or future refactoring.

## Decision
1. Make all struct fields `pub(crate)` instead of `pub`
2. Add accessor methods: `layers()`, `neurons()`, `connections()`, `value()`, `weight()`
3. Add builder/setter methods where mutation is needed
4. `Neuron.sum` and `Neuron.value` (transient computation) become fully private

## Consequences
- Breaking API change for direct field access
- Enables future refactoring (e.g., separating topology from runtime state)
- Preserves internal invariants
