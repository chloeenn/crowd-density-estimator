export function ErrorBanner({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div role="alert" className="flex items-start justify-between gap-3 rounded-lg border border-red-700 bg-red-50 p-4 text-red-900">
      <p>
        <span aria-hidden="true">⚠ </span>
        {message}
      </p>
      <button type="button" onClick={onDismiss} className="shrink-0 rounded px-2 font-medium underline focus:outline-none focus-visible:ring-4 focus-visible:ring-red-400">
        Dismiss
      </button>
    </div>
  );
}
