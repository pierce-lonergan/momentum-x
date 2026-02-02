# ADR-006: CI/CD Pipeline and Code Quality Gates

**Status**: ACCEPTED
**Date**: 2026-02-02
**Nodes**: ci.github_actions, dx.pre_commit
**Graph Link**: docs/memory/graph_state.json → ci.*

---

## Context

Momentum-X is open-sourced for community contribution. Without automated quality gates,
PRs can introduce regressions, type errors, or secret leaks. Every successful open-source
project has CI as table stakes.

## Drivers

1. **Regression prevention**: No merge without 127+ tests passing.
2. **Secret leak prevention**: Pre-commit hook catches `.env`, API keys, private keys.
3. **Code consistency**: Automated formatting and linting reduces review burden.
4. **Contributor confidence**: Green CI badge signals a well-maintained project.
5. **Multi-Python**: Must verify on Python 3.11 and 3.12.

## Decisions

### 1. GitHub Actions CI Pipeline

Triggered on push to `main` and all PRs. Matrix: Python 3.11 + 3.12.
Steps: install → lint → type-check → test. No API keys needed for CI.

### 2. Pre-commit Hooks

Local git hooks via `pre-commit` framework:
- ruff (format + lint)
- check-added-large-files (>500KB)
- detect-private-key
- detect-aws-credentials (catches API keys in common formats)
- check-merge-conflict
- end-of-file-fixer

### 3. Secret Scanning

Custom pre-commit hook regex: `[A-Za-z0-9]{20,}` in `.env`-like patterns.
GitHub Secret Scanning enabled on repository (free for public repos).

## Consequences

**Positive**: Zero-regression merges, secret leak prevention, contributor trust.
**Negative**: CI adds ~2 min to PR feedback loop. Acceptable.
**Risk**: Flaky tests could block merges. Mitigated by deterministic test suite (no network calls in unit tests).

## References

- GitHub Actions documentation
- pre-commit.com framework
- ADR-005 (Open-Source DX)
