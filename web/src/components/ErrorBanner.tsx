/** Errors are neutral and bordered, never the "Critical density" red, so a failure can't be mistaken for a verdict. */
import { useEffect, useRef } from "react";

export function ErrorBanner({ message, onDismiss, onRetry }: { message: string; onDismiss: () => void; onRetry?: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.focus({ preventScroll: true });
    ref.current?.scrollIntoView?.({ block: "nearest" });
  }, [message]);
  const btn = "pill pill-ghost";
  return (
    <div ref={ref} tabIndex={-1} role="alert" className="flex flex-wrap items-center justify-between gap-x-4 gap-y-3 rounded-2xl border border-edge bg-surface px-6 py-4 text-sm text-primary">
      <p>{message}</p>
      <div className="flex shrink-0">
        {onRetry && (
          <button type="button" onClick={onRetry} className={btn}>
            Try again
          </button>
        )}
        <button type="button" onClick={onDismiss} className={btn}>
          Dismiss
        </button>
      </div>
    </div>
  );
}
