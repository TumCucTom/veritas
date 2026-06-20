import OpenAI from "openai";

const FALLBACK =
  "New account with rapid high-value transfers fanning out to many recipients — a classic mule/safe-account pattern matching the active campaign.";

export async function POST(req: Request) {
  let transaction: unknown = {};
  try {
    const body = await req.json();
    transaction = body?.transaction ?? {};
  } catch {
    transaction = {};
  }

  const apiKey = process.env.MINIMAX_API_KEY;
  if (!apiKey) {
    return Response.json({ explanation: FALLBACK, source: "fallback" });
  }

  const client = new OpenAI({
    apiKey,
    baseURL: process.env.MINIMAX_BASE_URL ?? "https://api.minimax.io/v1",
  });
  const prompt = `You are a UK bank fraud analyst. In 2 short sentences, explain why this flagged transaction looks like a mule/APP-fraud account. Be concrete (account age, velocity, fan-out). Transaction: ${JSON.stringify(transaction)}`;

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
    });
  } catch {
    return Response.json({ explanation: FALLBACK, source: "fallback" });
  }
}
