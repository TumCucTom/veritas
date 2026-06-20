# Codeplain Submission Sources

This directory keeps the Codeplain source artifacts for the Veritas submission.
Do not delete the `.plain` files, local templates, config files, or runner
scripts after rendering. They are the source of truth for what Codeplain was
asked to regenerate.

Generated Codeplain outputs are intentionally ignored in this directory:
`plain_modules/`, `conformance_tests/`, `build/`, and `build_conformance_tests/`.
Render outputs should be reviewed before any generated code is copied into the
main Veritas source tree.

## Projects

| Project | Purpose | Primary target |
|---|---|---|
| `contract-clients/` | Regenerate the shared Veritas API types, validators, and typed client around the v1 contract. | `contract/`, `web/src/lib/`, `edge-sdk/src/` |
| `bank-connectors/` | Regenerate Tier 1 connector adapters and conformance fixtures for bank onboarding. | `node/node/connectors/` |

## Usage

Install Codeplain and provide an API key:

```sh
pip install codeplain
export CODEPLAIN_API_KEY="..."
```

Preview without API-side rendering:

```sh
cd codeplain/contract-clients
codeplain veritas_contract_clients.plain --dry-run

cd ../bank-connectors
codeplain veritas_bank_connectors.plain --dry-run
```

Render headlessly using the checked-in config:

```sh
cd codeplain/contract-clients
codeplain veritas_contract_clients.plain --headless

cd ../bank-connectors
codeplain veritas_bank_connectors.plain --headless
```

## Integration Rule

Generated files are not automatically authoritative. Keep the hand-owned
privacy, cryptography, aggregation, and model primitives in `core/`, `node/`,
and `controlplane/` as the trusted implementation. Use Codeplain output for the
contract-shaped edges: types, clients, connector adapters, fixtures, and
conformance tests.
