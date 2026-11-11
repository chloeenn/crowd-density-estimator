/** Upload is determinate; analysis has no measurable progress, so it slides. Both sit directly under the photo
 * so the wait is visible where the eye already is. */
export function Progress({ fraction, label }: { fraction: number | null; label: string }) {
  return (
    <div className="space-y-3" role="status">
      <div
        className="relative h-1.5 w-full overflow-hidden rounded-full bg-line"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={fraction === null ? undefined : Math.round(fraction * 100)}
      >
        {fraction === null ? (
          <div className="absolute inset-y-0 left-0 w-2/5 animate-slide rounded-full bg-ramp-1" />
        ) : (
          <div className="h-full rounded-full bg-ramp-1 transition-[width] duration-300 ease-out" style={{ width: `${Math.round(fraction * 100)}%` }} />
        )}
      </div>
      <p className="text-sm text-muted">{label}</p>
    </div>
  );
}
