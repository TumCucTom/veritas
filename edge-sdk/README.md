# Veritas Tier 0 — Edge SDK

On-device, cross-device federated fraud intelligence (TypeScript reference SDK).
Detects the **victim side** of APP fraud locally. **Raw events never leave the
device** — only a differentially-private weight delta is transmitted.

Mirrors the FL math in `core/veritas_core/{model,dp}.py`: logistic
`sigmoid(aug(X) @ w)`, full-batch gradient training, dim = `FEATURE_DIM + 1` (11),
DP = clip-to-L2 + Gaussian noise.

## Quick start

```ts
import { Veritas } from "@veritas/edge-sdk";

const veritas = Veritas.start({ nodeUrl: "https://bank-node.local", key: VERITAS_KEY });

const { risk, reason } = veritas.observePayment({
  payeeId, amount, isNewPayee,
  remoteAccessAppActive, inboundCallActive, sessionAnomaly,
});

await veritas.syncModel();        // GET  /edge/v1/model  (pull bank edge model)
await veritas.contributeUpdate(); // POST /edge/v1/updates (DP-protected delta only)
```

## Public API

- `Veritas.start({ nodeUrl?, key?, deviceToken?, transport?, dp?, seed?, seedEvents?, fraudRate? })`
- `veritas.observePayment(obs, label?) => { risk, reason, action, indicators }`
- `veritas.trainLocalModel(epochs?) => { recall, numExamples }`
- `veritas.syncModel() => { version, dim }`
- `veritas.contributeUpdate({ epochs? }) => { sent, numExamples, updateNorm, localRecall, response }`
- Getters: `dim`, `version`, `getWeights()`, `bufferSize()`

`transport` is injectable (`Transport` interface) so tests/demos run without a
live node; default is `FetchTransport` over the node's `/edge/v1/*`.

## Wire contract — `POST /edge/v1/updates`

```json
{ "deviceToken": "<ephemeral, not a customer id>", "update": [11 floats], "numExamples": 502 }
```

`GET /edge/v1/model -> { version, dim: 11, weights: [11 floats] }`.

## Build / test / demo

```bash
npm install
npx tsc --noEmit     # typecheck (clean)
npx vitest run       # 10 tests, all green
npx tsx demo/demo.ts # end-to-end against a fake node
```
