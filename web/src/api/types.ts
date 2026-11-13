export type Level = "Low" | "Medium" | "High" | "Critical";
export type Part = "A" | "B";

export interface Thresholds {
  low: number;
  medium: number;
  high: number;
  critical: number;
}

export type Scene = Part | "auto";
export type Point = [number, number];

export interface DensityBands {
  medium: number;
  high: number;
  critical: number;
}

export interface DensityResult {
  people_per_m2: number;
  level: Level;
  area_m2: number;
  count: number;
  in_region: boolean;
  bands: DensityBands;
  units: string;
}

export interface PredictResponse {
  count: number;
  level: Level;
  thresholds: Thresholds;
  density: DensityResult | null;
  routed: boolean;
  model: { version: string; part: string; git_sha: string };
  heatmap: string;
  inference_ms: number;
  detections?: Detection[] | null;
  /** The photo with labelled boxes, as a data URI. Present only when detection was requested. */
  detections_image?: string | null;
}

export interface Detection {
  label: string;
  confidence: number;
  box: [number, number, number, number];
}

export interface ModelDetail {
  version: string;
  part: Part;
  git_sha: string;
  run_id: string;
  epoch: number;
  val_mae: number | null;
  val_rmse: number | null;
}

export interface Metadata {
  default_part: Part;
  models: ModelDetail[];
  thresholds: Thresholds;
  density_bands: DensityBands;
  routing_threshold: number | null;
  detector_enabled?: boolean;
  limits: { max_upload_mb: number; allowed_content_types: string[] };
}
