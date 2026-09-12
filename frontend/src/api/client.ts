import type {
  AdminAnalytics,
  HealthResponse,
  HistoryListResponse,
  ModelInfo,
  ModelListResponse,
  PredictionResponse,
  ReportListResponse,
  ReportResponse,
} from "./types";

const DEFAULT_BASE = "/api/v1";
const BASE_KEY = "btai-api-base";
const ADMIN_KEY_STORAGE = "btai-admin-key";
const NO_BASE_KEYS = ["http", "https", "data", "blob"];

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

function resolveUrl(path: string): string {
  const base = getApiBase();
  if (NO_BASE_KEYS.some((p) => path.startsWith(p))) return path;
  return `${base}${path}`;
}

export function getApiBase(): string {
  const stored = localStorage.getItem(BASE_KEY);
  if (stored) return stored.replace(/\/+$/, "");
  return DEFAULT_BASE;
}

export function setApiBase(base: string): void {
  localStorage.setItem(BASE_KEY, base.replace(/\/+$/, ""));
}

export function getAdminKey(): string {
  return localStorage.getItem(ADMIN_KEY_STORAGE) ?? "";
}

export function setAdminKey(key: string): void {
  localStorage.setItem(ADMIN_KEY_STORAGE, key.trim());
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  admin = false,
): Promise<T> {
  const url = resolveUrl(path);
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (admin) {
    const key = getAdminKey();
    if (key) headers.set("X-Admin-Key", key);
  }
  if (init.body instanceof FormData === false && init.body) {
    headers.set("Content-Type", "application/json");
  }

  let response: Response;
  try {
    response = await fetch(url, { ...init, headers });
  } catch {
    throw new ApiError(0, "Cannot reach the analysis service. Check its address in Settings.");
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (body?.detail) detail = JSON.stringify(body.detail);
    } catch {
      /* keep default detail */
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/health"),

  analyze: (file: File, onProgress?: (fraction: number) => void) =>
    new Promise<PredictionResponse>((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", resolveUrl("/predictions"));
      xhr.responseType = "json";
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(xhr.response as PredictionResponse);
        } else {
          const detail =
            xhr.response && typeof xhr.response.detail === "string"
              ? xhr.response.detail
              : "Analysis failed. Please try a valid brain MRI image.";
          reject(new ApiError(xhr.status || 422, detail));
        }
      };
      xhr.onerror = () =>
        reject(
          new ApiError(0, "Cannot reach the analysis service. Check its address in Settings."),
        );
      xhr.upload.onprogress = (event) => {
        if (onProgress && event.lengthComputable) {
          onProgress(event.loaded / event.total);
        }
      };
      xhr.send(form);
    }),

  analysis: (id: string) => request<PredictionResponse>(`/predictions/${id}`),

  history: (params: {
    page?: number;
    page_size?: number;
    search?: string;
    prediction?: string;
  } = {}) => {
    const query = new URLSearchParams();
    if (params.page) query.set("page", String(params.page));
    if (params.page_size) query.set("page_size", String(params.page_size));
    if (params.search) query.set("search", params.search);
    if (params.prediction) query.set("prediction", params.prediction);
    const qs = query.toString();
    return request<HistoryListResponse>(`/history${qs ? `?${qs}` : ""}`);
  },

  models: () => request<ModelListResponse>("/models"),
  model: (id: number) => request<ModelInfo>(`/models/${id}`),
  activateModel: (id: number) =>
    request<ModelInfo>(`/models/${id}/activate`, { method: "POST" }, true),
  reloadModels: () =>
    request<{ status: string }>("/models/reload", { method: "POST" }, true),

  generateReport: (analysisId: string) =>
    request<ReportResponse>(
      "/reports",
      { method: "POST", body: JSON.stringify({ analysis_id: analysisId }) },
    ),

  reports: async (params: { page?: number; page_size?: number } = {}) => {
    const query = new URLSearchParams();
    if (params.page) query.set("page", String(params.page));
    if (params.page_size) query.set("page_size", String(params.page_size));
    const qs = query.toString();
    return request<ReportListResponse>(`/reports${qs ? `?${qs}` : ""}`);
  },

  downloadReport: async (url: string, filename: string) => {
    const absolute = resolveUrl(url);
    const response = await fetch(absolute);
    if (!response.ok) {
      throw new ApiError(response.status, "Report download failed.");
    }
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objectUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 2500);
  },

  analytics: () => request<AdminAnalytics>("/admin/analytics", {}, true),
};