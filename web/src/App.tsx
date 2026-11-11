import { useCallback, useEffect, useRef, useState } from "react";
import { getMetadata } from "./api/client";
import type { Metadata, Part, PredictResponse } from "./api/types";
import { Dropzone } from "./components/Dropzone";
import { ErrorBanner } from "./components/ErrorBanner";
import { HeatmapViewer } from "./components/HeatmapViewer";
import { ModelSwitch } from "./components/ModelSwitch";
import { Progress } from "./components/Progress";
import { ReferenceStrip } from "./components/ReferenceStrip";
import { ResultCard } from "./components/ResultCard";
import { SampleGallery, type Sample } from "./components/SampleGallery";
import { ScaleLegend } from "./components/ScaleLegend";
import { usePrediction } from "./hooks/usePrediction";
import { MODEL_LABEL } from "./status";
import { LevelBadge } from "./components/LevelBadge";
import { approxCount } from "./status";

interface Source {
  blob: Blob;
  url: string;
  name: string;
}

const DEFAULT_TYPES = ["image/jpeg", "image/png", "image/webp"];

/** True at the two-column breakpoint (Tailwind md), so the layout can restructure rather than just reflow. */
function useWide() {
  const q = "(min-width: 768px)";
  const [wide, setWide] = useState(() => window.matchMedia?.(q).matches ?? true);
  useEffect(() => {
    const m = window.matchMedia?.(q);
    if (!m) return;
    const on = () => setWide(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return wide;
}

export default function App() {
  const { state, run, reset } = usePrediction();
  const [meta, setMeta] = useState<Metadata | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [part, setPart] = useState<Part | undefined>();
  const [source, setSource] = useState<Source | null>(null);
  const [last, setLast] = useState<PredictResponse | null>(null);
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const resultTop = useRef<HTMLDivElement>(null);
  const wide = useWide();

  const loadMeta = useCallback((signal?: AbortSignal) => {
    getMetadata(signal)
      .then((m) => {
        setMeta(m);
        setPart(m.default_part);
      })
      .catch((e) => {
        if (!signal?.aborted) setMetaError(e.message);
      });
  }, []);

  useEffect(() => {
    const ctl = new AbortController();
    loadMeta(ctl.signal);
    return () => ctl.abort();
  }, [loadMeta]);

  // Object URLs hold the image in memory until revoked.
  useEffect(() => () => { if (source) URL.revokeObjectURL(source.url); }, [source]);

  const submit = useCallback(
    (blob: Blob, name: string, p: Part | undefined) => {
      setActionError(null);
      setLast(null); // a new photo has no previous result to show while it runs
      setSource({ blob, name, url: URL.createObjectURL(blob) });
      void run(blob, p);
    },
    [run],
  );

  const onFile = useCallback((f: File) => submit(f, f.name, part), [submit, part]);

  const onSample = useCallback(
    async (sample: Sample) => {
      // A sample is a scene for one model: choosing it selects that model, so the first result isn't a wrong-model one.
      const p = meta?.models.some((m) => m.part === sample.part) ? sample.part : part;
      setPart(p);
      try {
        const res = await fetch(sample.src);
        if (!res.ok) throw new Error(String(res.status));
        submit(await res.blob(), sample.label, p);
      } catch {
        setActionError("That sample image could not be loaded. Try another one, or upload your own photo.");
      }
    },
    [submit, part, meta],
  );

  const onPart = (p: Part) => {
    setPart(p);
    if (source) void run(source.blob, p); // re-run the same image on the other model
  };

  const retry = () => {
    if (source) void run(source.blob, part);
  };

  const clear = () => {
    reset();
    setSource(null);
    setLast(null);
    setActionError(null);
  };

  const busy = state.status === "uploading" || state.status === "inferring";
  const error = state.status === "error" ? state.message : (actionError ?? metaError);
  const done = state.status === "done" ? state.result : null;
  // Keep the previous result on screen, dimmed, while the same photo re-runs (model switch, retry), so nothing
  // vanishes and the two scene types can be compared on one image.
  if (done && last !== done) setLast(done);
  const shown = done ?? (busy ? last : null);
  const stale = !done && busy && last !== null;

  // The inputs collapse once a photo is chosen, so move keyboard and screen-reader focus to the result when it lands.
  useEffect(() => {
    if (state.status !== "done") return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    resultHeading.current?.focus({ preventScroll: true });
    // Scroll to the photo, not the heading: on one column the heading sits below it and would hide the heatmap.
    resultTop.current?.scrollIntoView?.({ block: "start", behavior: reduce ? "auto" : "smooth" });
  }, [state]);

  const modelSwitch =
    meta && part ? (
      <ModelSwitch models={meta.models} value={part} onChange={onPart} disabled={busy} />
    ) : metaError ? (
      <p className="text-sm text-muted">Scene type selection is unavailable right now, so the default model will be used.</p>
    ) : null;

  return (
    <div className="min-h-screen bg-page text-primary">
      <main className="mx-auto w-full max-w-5xl space-y-10 px-6 py-10 md:space-y-14 md:py-14">
        <header>
          <div className="space-y-4">
            <h1 className="text-5xl leading-[1.05] md:text-6xl">Crowd Estimator</h1>
            <p className="max-w-prose text-lg text-muted">Upload a crowd photo to get an estimate of the number of people and how they're spread out.</p>
          </div>
        </header>

        {error && (
          <ErrorBanner
            message={error}
            onDismiss={() => (state.status === "error" ? reset() : actionError ? setActionError(null) : setMetaError(null))}
            onRetry={state.status === "error" && source ? retry : !actionError && metaError ? () => {
                  setMetaError(null);
                  loadMeta();
                } : undefined}
          />
        )}

        <Dropzone onFile={onFile} disabled={busy} compact source={!!source} maxMb={meta?.limits.max_upload_mb} accept={meta?.limits.allowed_content_types ?? DEFAULT_TYPES} />

        {!source && <SampleGallery onPick={onSample} disabled={busy} />}

        {!source && modelSwitch}

        {source && (
          <div className="grid gap-8 md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <div ref={resultTop} className="scroll-mt-4 space-y-4">
              {shown ? (
                <div aria-busy={stale} className="space-y-4">
                  <HeatmapViewer originalUrl={source.url} heatmap={shown.heatmap} name={source.name} stale={stale} />
                  {!wide && (
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-2" aria-hidden="true">
                      <span className="font-display text-3xl font-medium tabular-nums">~{approxCount(shown.count)}</span>
                      <span className="text-sm text-muted">people</span>
                      <LevelBadge level={shown.level} />
                    </div>
                  )}
                  {wide ? (
                    <ScaleLegend />
                  ) : (
                    <details>
                      <summary className="pill pill-ghost cursor-pointer list-none">What do the colours mean?</summary>
                      <div className="pt-3">
                        <ScaleLegend />
                      </div>
                    </details>
                  )}
                </div>
              ) : (
                <div className="overflow-hidden rounded-2xl">
                  <img src={source.url} alt={`Uploaded photo: ${source.name}`} className={`block w-full ${busy ? "opacity-60" : ""}`} />
                </div>
              )}
              {state.status === "uploading" && <Progress fraction={state.progress} label={`Uploading… ${Math.round(state.progress * 100)}%`} />}
              {state.status === "inferring" && <Progress fraction={null} label="Analysing the image… this can take several seconds." />}
            </div>
            <div className="space-y-6">
              {modelSwitch}
              {shown && (
                <div aria-busy={stale}>
                  <ResultCard
                    result={shown}
                    headingRef={resultHeading}
                    stale={stale}
                    partLabel={part ? MODEL_LABEL[part].name : undefined}
                  />
                </div>
              )}
              <button type="button" onClick={clear} disabled={busy} className="pill pill-ghost disabled:opacity-40">
                Clear photo and result
              </button>
            </div>
          </div>
        )}

        <ReferenceStrip thresholds={meta?.thresholds} level={done?.level} />
      </main>
    </div>
  );
}
