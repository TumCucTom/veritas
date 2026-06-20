"use client";
import { useEffect } from "react";
import { api } from "./api";
import type { VeritasEvent } from "./types";

export function useEvents(onEvent: (e: VeritasEvent) => void) {
  useEffect(() => {
    let es: EventSource | null = null;
    try {
      es = new EventSource(api.eventsUrl);
    } catch {
      // EventSource unsupported or URL invalid — degrade gracefully.
      return;
    }
    const h =
      (t: VeritasEvent["type"]) =>
      (m: MessageEvent): void => {
        try {
          onEvent({ type: t, data: JSON.parse(m.data) } as VeritasEvent);
        } catch {
          // Ignore malformed event payloads rather than crash the stream.
        }
      };
    const handlers = (
      ["round_complete", "client_updated", "fraud_propagated", "attack_detected"] as const
    ).map((t) => {
      const fn = h(t);
      es!.addEventListener(t, fn);
      return [t, fn] as const;
    });
    return () => {
      for (const [t, fn] of handlers) es?.removeEventListener(t, fn);
      es?.close();
    };
  }, [onEvent]);
}
