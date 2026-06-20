"use client";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useReducer,
  type ReactNode,
} from "react";
import { api } from "./api";
import { useEvents } from "./useEvents";
import type { State, VeritasEvent } from "./types";

export interface LastAttack {
  bankId: string;
  rejected: boolean;
  at: number;
}

export interface ConnectionStatus {
  status: "connecting" | "live" | "offline";
}

interface VeritasState {
  state: State | null;
  lastAttack: LastAttack | null;
  connection: ConnectionStatus["status"];
}

type Action =
  | { type: "replace"; state: State }
  | { type: "attack"; bankId: string; rejected: boolean }
  | { type: "clearAttack" }
  | { type: "connection"; status: ConnectionStatus["status"] };

const initialState: VeritasState = {
  state: null,
  lastAttack: null,
  connection: "connecting",
};

function reducer(prev: VeritasState, action: Action): VeritasState {
  switch (action.type) {
    case "replace":
      return { ...prev, state: action.state, connection: "live" };
    case "attack":
      return {
        ...prev,
        lastAttack: { bankId: action.bankId, rejected: action.rejected, at: Date.now() },
      };
    case "clearAttack":
      return { ...prev, lastAttack: null };
    case "connection":
      return { ...prev, connection: action.status };
    default:
      return prev;
  }
}

interface VeritasContextValue extends VeritasState {
  refresh: () => Promise<void>;
}

const VeritasContext = createContext<VeritasContextValue | null>(null);

export function VeritasProvider({ children }: { children: ReactNode }) {
  const [vstate, dispatch] = useReducer(reducer, initialState);

  const refresh = useCallback(async (): Promise<void> => {
    try {
      const next = await api.state();
      dispatch({ type: "replace", state: next });
    } catch {
      dispatch({ type: "connection", status: "offline" });
    }
  }, []);

  // Initialise from the snapshot endpoint on mount.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const onEvent = useCallback((e: VeritasEvent): void => {
    if (e.type === "round_complete") {
      dispatch({ type: "replace", state: e.data });
    } else if (e.type === "attack_detected") {
      dispatch({ type: "attack", bankId: e.data.bankId, rejected: e.data.rejected });
    }
  }, []);

  useEvents(onEvent);

  // Auto-dismiss the transient attack banner.
  useEffect(() => {
    if (!vstate.lastAttack) return;
    const id = setTimeout(() => dispatch({ type: "clearAttack" }), 4200);
    return () => clearTimeout(id);
  }, [vstate.lastAttack]);

  return (
    <VeritasContext.Provider value={{ ...vstate, refresh }}>
      {children}
    </VeritasContext.Provider>
  );
}

export function useVeritas(): VeritasContextValue {
  const ctx = useContext(VeritasContext);
  if (!ctx) throw new Error("useVeritas must be used within a VeritasProvider");
  return ctx;
}
