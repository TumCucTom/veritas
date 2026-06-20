import OpenAI from "openai";

const FALLBACK =
  "New account with rapid high-value transfers fanning out to many recipients — a classic mule/safe-account pattern matching the active campaign.";

interface ExplainRequest {
  transaction?: unknown;
  indicators?: unknown;
  confidence?: unknown;
  label?: unknown;
}

function asIndicators(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}

function asLabel(v: unknown): string {
  return typeof v === "string" && v.length > 0 ? v : "flagged";
}

function asConfidence(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export async function POST(req: Request) {
  let body: ExplainRequest = {};
  try {
    body = (await req.json()) as ExplainRequest;
  } catch {
    body = {};
  }

  const transaction = body.transaction ?? {};
  const indicators = asIndicators(body.indicators);
  const confidence = asConfidence(body.confidence);
  const label = asLabel(body.label);

  // "grounded in" is always the model-derived indicator set — surfaced so the
  // summary is visibly tied to /predict output, never free-floating LLM prose.
  const groundedIn = indicators.length > 0 ? indicators.join(", ") : "model risk indicators";
  const confidencePct = confidence != null ? `${Math.round(confidence * 100)}%` : "n/a";

  const apiKey = process.env.MINIMAX_API_KEY;
  if (!apiKey) {
    return Response.json({ explanation: FALLBACK, source: "fallback", groundedIn });
  }

  const client = new OpenAI({
    apiKey,
    baseURL: process.env.MINIMAX_BASE_URL ?? "https://api.minimax.io/v1",
  });

  // The model is a NARRATOR, not a detector. It may ONLY restate, in plain
  // English, the indicators / label / confidence produced by the real model's
  // /predict output. It must not invent new facts, numbers, or risk factors.
  const prompt = `You are a UK bank fraud analyst writing a one-or-two sentence plain-English summary of an ALREADY-MADE model decision.

The fraud model produced this decision (this is the ONLY ground truth — do not add to it):
- Label: ${label}
- Confidence: ${confidencePct}
- Indicators the model fired on: ${groundedIn}
- Raw transaction features: ${JSON.stringify(transaction)}

Rules:
- ONLY explain using the indicators, label, and confidence above. Do NOT introduce any new fact, number, account detail, amount, or risk factor that is not present above.
- Do not contradict the label or confidence. Do not invent specific figures.
- Write 1-2 short sentences, concrete and analyst-grade, restating WHY those indicators add up to the "${label}" decision.`;

  try {
    const r = await client.chat.completions.create({
      model: process.env.MINIMAX_MODEL ?? "MiniMax-Text-01",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 120,
    });
    const explanation = r.choices[0]?.message?.content ?? "";
    return Response.json({
      explanation: explanation || FALLBACK,
      source: explanation ? "minimax" : "fallback",
      groundedIn,
    });
  } catch {
    return Response.json({ explanation: FALLBACK, source: "fallback", groundedIn });
  }
}
