import { useCallback, useEffect, useReducer, useRef } from "react";
import { ApiFailure, predict, type PredictOptions } from "../api/client";
import type { PredictResponse } from "../api/types";

export type RunOptions = Pick<PredictOptions, "part" | "areaM2" | "region">;

export type State =
  | { status: "idle" }
  | { status: "uploading"; progress: number }
  | { status: "inferring" }
  | { status: "done"; result: PredictResponse }
  | { status: "error"; message: string; code: number | null };

export type Action =
  | { type: "start" }
  | { type: "progress"; fraction: number }
  | { type: "uploaded" }
  | { type: "success"; result: PredictResponse }
  | { type: "failure"; message: string; code: number | null }
  | { type: "reset" };

export function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "start":
      return { status: "uploading", progress: 0 };
    case "progress":
      return state.status === "uploading" ? { status: "uploading", progress: action.fraction } : state;
    case "uploaded":
      return state.status === "uploading" ? { status: "inferring" } : state;
    case "success":
      return { status: "done", result: action.result };
    case "failure":
      return { status: "error", message: action.message, code: action.code };
    case "reset":
      return { status: "idle" };
  }
}

/** idle -> uploading -> inferring -> done | error. A new run aborts the one in flight. */
export function usePrediction() {
  const [state, dispatch] = useReducer(reducer, { status: "idle" });
  const controller = useRef<AbortController | null>(null);

  const run = useCallback(async (file: Blob, options: RunOptions = {}) => {
    controller.current?.abort();
    const ctl = new AbortController();
    controller.current = ctl;
    dispatch({ type: "start" });
    try {
      const result = await predict(file, {
        ...options,
        signal: ctl.signal,
        onUploadProgress: (f) => {
          if (ctl.signal.aborted) return;
          dispatch({ type: "progress", fraction: f });
          if (f >= 1) dispatch({ type: "uploaded" });
        },
      });
      if (!ctl.signal.aborted) dispatch({ type: "success", result });
    } catch (err) {
      if (ctl.signal.aborted) return; // superseded or cancelled: the newer run owns the state
      const f = err instanceof ApiFailure ? err : null;
      dispatch({
        type: "failure",
        message: f?.message ?? "Something went wrong. Please try again.",
        code: f?.status ?? null,
      });
    }
  }, []);

  const reset = useCallback(() => {
    controller.current?.abort();
    dispatch({ type: "reset" });
  }, []);

  useEffect(() => () => controller.current?.abort(), []);

  return { state, run, reset };
}
