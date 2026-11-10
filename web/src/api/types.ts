export type Level = "Low" | "Medium" | "High" | "Critical";
export type Part = "A" | "B";

export interface Thresholds {
  low: number;
  medium: number;
  high: number;
  critical: number;
}

export interface PredictResponse {
  count: number;
  level: Level;
  local_concentration: number;
  max_concentration: number;
  concentration_basis: {
    cell_input_px: number;
    processed_width: number;
    processed_height: number;
    units: string;
  };
  thresholds: Thresholds;
  model: { version: string; part: string; git_sha: string };
  heatmap: string;
  inference_ms: number;
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
  level_caveat: string;
  limits: { max_upload_mb: number; allowed_content_types: string[] };
}
