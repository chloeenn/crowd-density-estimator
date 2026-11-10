import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiFailure } from "../api/client";
import type { PredictResponse } from "../api/types";
import { reducer, usePrediction } from "./usePrediction";

vi.mock("../api/client", async (orig) => ({ ...(await orig<object>()), predict: vi.fn() }));
import { predict } from "../api/client";

const result = { count: 10 } as PredictResponse;

describe("reducer", () => {
  it("walks idle -> uploading -> inferring -> done", () => {
    let s = reducer({ status: "idle" }, { type: "start" });
    expect(s).toEqual({ status: "uploading", progress: 0 });
    s = reducer(s, { type: "progress", fraction: 0.5 });
    expect(s).toEqual({ status: "uploading", progress: 0.5 });
    s = reducer(s, { type: "uploaded" });
    expect(s.status).toBe("inferring");
    expect(reducer(s, { type: "success", result }).status).toBe("done");
  });
  it("ignores late progress events once inferring", () => {
    expect(reducer({ status: "inferring" }, { type: "progress", fraction: 1 }).status).toBe("inferring");
  });
});

describe("usePrediction", () => {
  beforeEach(() => {
    vi.mocked(predict).mockReset();
  });

  it("aborts a superseded request and keeps only the newest result", async () => {
    const signals: AbortSignal[] = [];
    vi.mocked(predict).mockImplementation((...args: unknown[]) => {
      const o = args[1] as { signal: AbortSignal } | undefined;
      signals.push(o!.signal);
      return new Promise((resolve, reject) => {
        o!.signal.addEventListener("abort", () => reject(new Error("cancelled")));
        if (signals.length === 2) resolve(result);
      });
    });
    const { result: hook } = renderHook(() => usePrediction());
    act(() => void hook.current.run(new Blob(["a"])));
    act(() => void hook.current.run(new Blob(["b"])));
    expect(signals[0].aborted).toBe(true);
    await waitFor(() => expect(hook.current.state.status).toBe("done"));
  });

  it("reports a failure with its status", async () => {
    vi.mocked(predict).mockImplementation(async () => {
      throw new ApiFailure(415, "Only JPEG, PNG and WebP images are supported.");
    });
    const { result: hook } = renderHook(() => usePrediction());
    await act(() => hook.current.run(new Blob(["a"])));
    expect(hook.current.state).toEqual({ status: "error", message: "Only JPEG, PNG and WebP images are supported.", code: 415 });
  });
});
