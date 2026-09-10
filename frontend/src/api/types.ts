export type ClassKey = "glioma" | "meningioma" | "no_tumor" | "pituitary_tumor";

export const CLASS_LABELS: Record<ClassKey, string> = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  no_tumor: "No Tumor",
  pituitary_tumor: "Pituitary Tumor",
};

export const CLASS_ORDER: ClassKey[] = [
  "glioma",
  "meningioma",
  "no_tumor",
  "pituitary_tumor",
];

export interface ModelMeta {
  name: string;
  version: string;
  architecture: string;
  input_size: number;
  dataset_version: string;
}

export interface PredictionLinks {
  image: string | null;
  gradcam: string | null;
}

export interface PredictionResponse {
  analysis_id: string;
  prediction: string;
  label: string;
  confidence: number;
  low_confidence: boolean;
  probabilities: Record<string, number>;
  model: ModelMeta;
  processing_time_ms: number | null;
  created_at: string;
  report_available: boolean;
  links: PredictionLinks;
  note?: string | null;
  warnings?: string[];
  filename?: string | null;
}

export interface HistoryItem {
  id: string;
  created_at: string;
  filename: string;
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_name: string;
  model_version: string;
  low_confidence: boolean;
  processing_time_ms: number | null;
  report_available: boolean;
}

export interface HistoryListResponse {
  items: HistoryItem[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

export interface ModelInfo {
  id: number;
  name: string;
  version: string;
  architecture: string;
  file_path: string;
  input_size: number;
  dataset_version: string;
  training_datetime: string | null;
  metrics: Record<string, unknown>;
  status: string;
  description: string;
  is_active: boolean;
  created_at: string;
}

export interface ModelListResponse {
  models: ModelInfo[];
  active: ModelInfo | null;
}

export interface ReportResponse {
  id: string;
  analysis_id: string;
  size_bytes: number;
  generated_at: string;
  download_url: string | null;
}

export interface ReportListItem {
  id: string;
  analysis_id: string;
  created_at: string;
  size_bytes: number;
  filename: string | null;
  prediction: string | null;
  confidence: number | null;
  model_name: string | null;
  model_version: string | null;
  download_url: string;
}

export interface ReportListResponse {
  items: ReportListItem[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  environment: string;
  uptime_seconds: number;
  database: string;
  model: {
    name: string;
    version: string;
    architecture: string;
    status: string;
  } | null;
  admin_enabled: boolean;
}

export interface DistributionItem {
  prediction: string;
  count: number;
}

export interface AdminAnalytics {
  total_analyses: number;
  total_reports: number;
  prediction_distribution: DistributionItem[];
  recent_analyses: {
    id: string;
    created_at: string;
    filename: string;
    prediction: string;
    confidence: number;
    model_name: string;
    model_version: string;
  }[];
  model_versions: {
    id: number;
    name: string;
    version: string;
    architecture: string;
    input_size: number;
    dataset_version: string;
    status: string;
    is_active: boolean;
    metrics: Record<string, unknown>;
    file_path: string;
  }[];
  active_model: {
    id: number;
    name: string;
    version: string;
    architecture: string;
    input_size: number;
    dataset_version: string;
    status: string;
    is_active: boolean;
    metrics: Record<string, unknown>;
    file_path: string;
  } | null;
}

export function metricNumber(metrics: Record<string, unknown>, key: string): number | null {
  const value = metrics?.[key];
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

export const DISCLAIMER_TEXT =
  "This AI-generated result is intended for research and educational assistance only and does not constitute a medical diagnosis. Always consult a qualified healthcare professional for medical interpretation.";