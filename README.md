# Veritas

**Federated fraud intelligence for UK banking.** One fraud-detection model trained across
many banks' books so they catch cross-bank scam/mule campaigns in hours instead of days —
without any customer record ever leaving a bank.

## Where things are

| Path | What it is |
|------|------------|
| `core/` | Python federated-learning engine — model stack (logistic, MLP, GRU, GraphSAGE GNN, embeddings, federated GBDT + stacked ensemble), FedAvg, Multi-Krum, secure aggregation, differential privacy (RDP accountant), VSA zero-knowledge proofs; trains as a real `flock_sdk` model. |
| `controlplane/` | Tier-2 control plane — enrolment/auth (EdDSA JWTs), federation rounds, model registry, node attestation, signed tamper-evident transparency log. |
| `node/` | Bank-node runtimes — identity, attestation, connectors/feature maps, federation client. |
| `web/` | Next.js demo console — the live siloed-vs-federated race + the **"Under the hood"** gallery of 8 real-model visualizations. |
| `mock/` | Standalone Express + SSE server that drives the demo dashboard (same contract as the real stack). |
| `tools/` | Offline generators (`gen_*_viz.py`) that precompute the model visualizations into `web/public/viz/*.json`. |
| `edge-sdk/` · `contract/` · `deploy/` | TypeScript edge SDK · shared API contract · local deployment harness. |
| `docs/` | `demo-script.md` (2-min walkthrough), pitch/deck, architecture, `PRODUCTION.md` (real-vs-stub). |

## Run the demo

**Quick path** (the visual demo — mock drives the dashboard):

```bash
cd mock && npm i && npm start        # :8001  (race data)
cd web  && npm i && npm run dev       # :3000  (console — point NEXT_PUBLIC_API_BASE at :8001)
```

Open **http://localhost:3000** and click **Auto-run** (it starts the scam campaign and the race diverges). The "Under the hood" panels are precomputed static JSON — no backend needed.

**Full federation** (real distributed stack for the governance/attestation panels):

```bash
VERITAS_DEV=1 VERITAS_DP_SIGMA=0.1 VERITAS_BLIND_NODE=3 bash deploy/run_local.sh up 5
# approve the 5 members, then run web with NEXT_PUBLIC_CONTROL_PLANE=http://localhost:9000
```

`web/.env.local`: `NEXT_PUBLIC_API_BASE` (mock `:8001`), `NEXT_PUBLIC_CONTROL_PLANE` (`:9000`), `NEXT_PUBLIC_TENANT_ID` (`tenant-node0`), and optional `MINIMAX_API_KEY` for the AI explainer. Teardown: `bash deploy/run_local.sh down`.

---

![Slide 01](docs/slides/slide-01.jpg)

![Slide 02](docs/slides/slide-02.jpg)

![Slide 03](docs/slides/slide-03.jpg)

![Slide 04](docs/slides/slide-04.jpg)

![Slide 05](docs/slides/slide-05.jpg)

![Slide 06](docs/slides/slide-06.jpg)

![Slide 07](docs/slides/slide-07.jpg)

![Slide 08](docs/slides/slide-08.jpg)

![Slide 09](docs/slides/slide-09.jpg)

![Slide 10](docs/slides/slide-10.jpg)

![Slide 11](docs/slides/slide-11.jpg)

![Slide 12](docs/slides/slide-12.jpg)

![Slide 13](docs/slides/slide-13.jpg)

![Slide 14](docs/slides/slide-14.jpg)

![Slide 15](docs/slides/slide-15.jpg)

![Slide 16](docs/slides/slide-16.jpg)

![Slide 17](docs/slides/slide-17.jpg)

![Slide 18](docs/slides/slide-18.jpg)

![Slide 19](docs/slides/slide-19.jpg)

![Slide 20](docs/slides/slide-20.jpg)

![Slide 21](docs/slides/slide-21.jpg)

![Slide 22](docs/slides/slide-22.jpg)

![Slide 23](docs/slides/slide-23.jpg)

![Slide 24](docs/slides/slide-24.jpg)
