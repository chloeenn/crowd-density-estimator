import axios, { AxiosError } from "axios";
import type { Metadata, Part, Point, PredictResponse } from "./types";

const http = axios.create({ baseURL: `${import.meta.env.VITE_API_URL ?? ""}/api` });

export class ApiFailure extends Error {
  constructor(
    public readonly status: number | null,
    message: string,
    public readonly retryAfter?: number,
  ) {
    super(message);
  }
}

/** Plain-language text for each failure the user can act on. */
export function friendlyMessage(status: number | null, detail?: string): string {
  switch (status) {
    case 400:
      return "That file could not be read as an image. It may be damaged or incomplete. Try another photo.";
    case 413:
      return "That image is too large. Try a smaller file or a lower-resolution photo.";
    case 415:
      return "Only JPEG, PNG and WebP images are supported.";
    case 422:
      return detail ?? "The request was not valid. Try a different image or model.";
    case 503:
      return "The service is busy or still starting. Wait a few seconds and try again.";
    case null:
      return "Could not reach the server. Check your connection and that the API is running.";
    default:
      return "Something went wrong on the server. Please try again.";
  }
}

function toFailure(err: unknown): ApiFailure {
  const e = err as AxiosError<{ detail?: string }>;
  const status = e.response?.status ?? null;
  const retry = Number(e.response?.headers?.["retry-after"]);
  return new ApiFailure(status, friendlyMessage(status, e.response?.data?.detail), retry || undefined);
}

export async function getMetadata(signal?: AbortSignal): Promise<Metadata> {
  try {
    return (await http.get<Metadata>("/metadata", { signal })).data;
  } catch (err) {
    if (axios.isCancel(err)) throw err;
    throw toFailure(err);
  }
}

export interface PredictOptions {
  /** Force one model. Leave out to let the server choose by scene. */
  part?: Part;
  /** Ground area in m². With it the result includes people per m². */
  areaM2?: number;
  /** Polygon (x, y as 0-1 fractions of the photo) that the area belongs to. */
  region?: Point[];
  /** Also ask for labelled object boxes (only when the server has a detector). */
  detect?: boolean;
  signal?: AbortSignal;
  onUploadProgress?: (fraction: number) => void;
}

export async function predict(file: Blob, opts: PredictOptions = {}): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file, file instanceof File ? file.name : "image");
  try {
    const res = await http.post<PredictResponse>("/predict", form, {
      params: {
        ...(opts.part ? { part: opts.part } : {}),
        ...(opts.areaM2 ? { area_m2: opts.areaM2 } : {}),
        ...(opts.detect ? { detect: true } : {}),
        ...(opts.areaM2 && opts.region ? { region: JSON.stringify(opts.region) } : {}),
      },
      signal: opts.signal,
      onUploadProgress: (e) => e.total && opts.onUploadProgress?.(e.loaded / e.total),
    });
    return res.data;
  } catch (err) {
    if (axios.isCancel(err)) throw err;
    throw toFailure(err);
  }
}

export { axios };
