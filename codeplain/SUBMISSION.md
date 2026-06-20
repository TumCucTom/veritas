# Codeplain Hackathon Submission Notes

Veritas uses Codeplain as the spec-driven regeneration layer for the parts of the
system that should stay contract-shaped and reviewable: public API clients,
runtime validators, typed fixtures, bank connector adapters, and conformance
tests. The hand-owned privacy, cryptography, aggregation, and model primitives
remain in the main source tree; Codeplain owns the repeatable edges around them.

## Judging Map

### spec-driven development setup

- `contract-clients/veritas_contract_clients.plain` specifies the v1 Veritas API
  contract client, runtime validators, deterministic fixtures, package
  scaffolding, and exact integration targets in `contract/`, `web/src/lib/`, and
  `edge-sdk/src/`.
- `bank-connectors/veritas_bank_connectors.plain` specifies Tier 1 bank
  connector regeneration for CSV exports, ISO 20022 payment messages, warehouse
  adapter interfaces, synthetic fixtures, and privacy conformance checks.
- Each project commits its `.plain` spec, imported local template, `config.yaml`,
  and runner scripts so judges can inspect or rerun the same source artifacts.
- `tests/test_codeplain_submission.py` is the source-artifact guard:

```sh
python3 -m unittest tests.test_codeplain_submission
```

### presentation

The project story is simple: banks cannot pool raw customer records, but they can
share fraud intelligence through a governed federated system. Codeplain is used
where that story needs repeatability: generating the boring-but-critical edges
that prove new banks and clients can be onboarded from written specs.

Demo flow:

1. Show the Veritas fraud-intelligence UI and contract.
2. Open `codeplain/contract-clients/veritas_contract_clients.plain` to show the
   API client and fixture spec.
3. Open `codeplain/bank-connectors/veritas_bank_connectors.plain` to show how a
   new bank connector can be generated from a plain-language onboarding spec.
4. Run the guard command above to prove the submitted `.plain` files, configs,
   templates, and runners are present and reviewable.

### innovation and creativity

Most hackathon AI demos generate application code once. Veritas uses Codeplain as
a regeneration contract for regulated collaboration: the specs describe how to
add banks, preserve zero raw-record sharing, keep API clients aligned with the
contract, and generate conformance evidence without live customer data.

### charm

The pitch line: Veritas lets banks compare notes on fraud without passing the
notes around. Codeplain turns that promise into inspectable specs instead of a
pile of prompts.

## Render Evidence

The current branch includes two Codeplain projects:

| Project | Functional specs | Primary generated surface | Status |
|---|---:|---|---|
| `contract-clients` | 6 | TypeScript contract client, validators, fixtures, integration notes | Previously rendered locally; current hardened spec dry-runs clean and now requires package scaffolding and real Veritas paths |
| `bank-connectors` | 5 | Python connector package, synthetic fixtures, conformance tests | Previously rendered locally; current hardened spec dry-runs clean and now forbids stale `fetch_records()` warehouse output |

Generated folders and logs are ignored so the public repository keeps the
submission source clean. Rerender with:

```sh
cd codeplain/contract-clients
codeplain veritas_contract_clients.plain --headless

cd ../bank-connectors
codeplain veritas_bank_connectors.plain --headless
```

Before copying any generated output into the main source tree, review the diff
and run the relevant unit/conformance runner from that project directory.
