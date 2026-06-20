/**
 * Tiny end-to-end demo of the Veritas Tier 0 Edge SDK against a FAKE node.
 * Run with:  npx tsx demo/demo.ts   (from edge-sdk/)
 *
 * It: starts the SDK, syncs the bank edge model, trains on-device, scores a
 * fraud vs a benign payment, and contributes a DP-protected update — printing
 * exactly what crosses the wire (no raw events).
 */
import { Veritas } from "../src/index.js";
import { MODEL_DIM, initWeights } from "../src/index.js";
import type {
  EdgeModelResponse,
  EdgeUpdateRequest,
  EdgeUpdateResponse,
  Transport,
} from "../src/index.js";

class DemoNode implements Transport {
  received: EdgeUpdateRequest[] = [];
  async getModel(): Promise<EdgeModelResponse> {
    return { version: 3, dim: MODEL_DIM, weights: initWeights(MODEL_DIM - 1) };
  }
  async postUpdate(body: EdgeUpdateRequest): Promise<EdgeUpdateResponse> {
    this.received.push(body);
    return { accepted: true };
  }
}

async function main(): Promise<void> {
  const node = new DemoNode();
  const veritas = Veritas.start({
    transport: node,
    key: "demo-bank-key",
    deviceToken: "ephemeral-enrolment-token-abc",
    seed: 2026,
    seedEvents: 500,
    fraudRate: 0.12,
  });

  console.log("1. start()  — buffered", veritas.bufferSize(), "synthetic on-device events");

  const synced = await veritas.syncModel();
  console.log(`2. syncModel() — pulled bank edge model v${synced.version} (dim ${synced.dim})`);

  const trained = veritas.trainLocalModel();
  console.log(
    `3. trainLocal() — local recall ${trained.recall.toFixed(3)} over ${trained.numExamples} events (raw events stay on-device)`,
  );

  const fraud = veritas.observePayment({
    payeeId: "safe-account-7741",
    amount: 8500,
    isNewPayee: true,
    remoteAccessAppActive: true,
    inboundCallActive: true,
    sessionAnomaly: 0.92,
  });
  console.log("\n4a. observePayment(FRAUD):");
  console.log("    risk   =", fraud.risk.toFixed(3), "action =", fraud.action);
  console.log("    reason =", fraud.reason);
  console.log("    flags  =", fraud.indicators.join(", "));

  const benign = veritas.observePayment({
    payeeId: "monthly-rent",
    amount: 720,
    isNewPayee: false,
    sessionAnomaly: 0.04,
    accountAgeDays: 2200,
  });
  console.log("\n4b. observePayment(BENIGN):");
  console.log("    risk   =", benign.risk.toFixed(3), "action =", benign.action);
  console.log("    reason =", benign.reason);

  const contrib = await veritas.contributeUpdate();
  console.log("\n5. contributeUpdate():");
  console.log(`    sent=${contrib.sent}  numExamples=${contrib.numExamples}  updateNorm=${contrib.updateNorm.toFixed(4)}`);

  const payload = node.received[0]!;
  console.log("\n   POST /edge/v1/updates payload (the ONLY thing that left the device):");
  console.log("   {");
  console.log(`     deviceToken: ${JSON.stringify(payload.deviceToken)},`);
  console.log(`     numExamples: ${payload.numExamples},`);
  console.log(`     update: [${payload.update.map((x) => x.toFixed(3)).join(", ")}]`);
  console.log("   }");
  console.log("\n   Note: no payeeId, amount, features, or labels are present — only a DP-noised weight delta.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
