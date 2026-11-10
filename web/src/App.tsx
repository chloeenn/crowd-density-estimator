import { useCallback, useEffect, useState } from "react";
import { getMetadata } from "./api/client";
import type { Metadata, Part } from "./api/types";
import { Dropzone } from "./components/Dropzone";
import { ErrorBanner } from "./components/ErrorBanner";
import { HeatmapViewer } from "./components/HeatmapViewer";
import { ModelSwitch } from "./components/ModelSwitch";
import { ResultCard } from "./components/ResultCard";
import { SampleGallery } from "./components/SampleGallery";
import { ScaleLegend } from "./components/ScaleLegend";
import { usePrediction } from "./hooks/usePrediction";

interface Source {
  blob: Blob;
  url: string;
  name: string;
}

const DEFAULT_TYPES = ["image/jpeg", "image/png", "image/webp"];

export default function App() {
  const { state, run, reset } = usePrediction();
  const [meta, setMeta] = useState<Metadata | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [part, setPart] = useState<Part | undefined>();
  const [source, setSource] = useState<Source | null>(null);

  useEffect(() => {
    const ctl = new AbortController();
    getMetadata(ctl.signal)
      .then((m) => {
        setMeta(m);
        setPart(m.default_part);
      })
      .catch((e) => {
        if (!ctl.signal.aborted) setMetaError(e.message);
      });
    return () => ctl.abort();
  }, []);

  // Object URLs hold the image in memory until revoked.
  useEffect(() => () => { if (source) URL.revokeObjectURL(source.url); }, [source]);

  const submit = useCallback(
    (blob: Blob, name: string, p: Part | undefined) => {
      setSource({ blob, name, url: URL.createObjectURL(blob) });
      void run(blob, p);
    },
    [run],
  );

  const onFile = useCallback((f: File) => submit(f, f.name, part), [submit, part]);

  const onSample = useCallback(
    async (src: string, label: string) => {
      try {
        const blob = await (await fetch(src)).blob();
        submit(blob, label, part);
      } catch {
        submit(new Blob(), label, part); // surfaces the normal error path
      }
    },
    [submit, part],
  );

  const onPart = (p: Part) => {
    setPart(p);
    if (source) void run(source.blob, p); // re-run the same image on the other model
  };

  const busy = state.status === "uploading" || state.status === "inferring";
  const error = state.status === "error" ? state.message : metaError;

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <main className="mx-auto max-w-3xl space-y-6 px-4 py-8">
        <header>
          <h1 className="text-2xl font-bold">Crowd Estimator</h1>
          <p className="text-slate-700">Upload a crowd photo to estimate how many people are in it and where they are concentrated.</p>
        </header>

        {error && (
          <ErrorBanner
            message={error}
            onDismiss={() => (state.status === "error" ? reset() : setMetaError(null))}
          />
        )}

        <Dropzone onFile={onFile} disabled={busy} accept={meta?.limits.allowed_content_types ?? DEFAULT_TYPES} />

        {meta && part && <ModelSwitch models={meta.models} value={part} onChange={onPart} />}

        <SampleGallery onPick={onSample} disabled={busy} />

        <div aria-live="polite" className="min-h-6 text-sm text-slate-700">
          {state.status === "uploading" && `Uploading… ${Math.round(state.progress * 100)}%`}
          {state.status === "inferring" && "Analysing the image… this can take several seconds."}
        </div>

        {state.status === "done" && source && (
          <div className="space-y-4">
            <ResultCard result={state.result} caveat={meta?.level_caveat} />
            <HeatmapViewer originalUrl={source.url} heatmap={state.result.heatmap} name={source.name} />
            <ScaleLegend result={state.result} />
          </div>
        )}
      </main>
    </div>
  );
}
