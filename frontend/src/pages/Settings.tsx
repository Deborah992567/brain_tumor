import { useEffect, useState } from "react";
import { api, getAdminKey, setAdminKey, getApiBase, setApiBase } from "../api/client";
import type { HealthResponse } from "../api/types";
import { useTheme, type ThemePreference } from "../theme";
import { useToast } from "../components/Toasts";

const STORAGE_KEYS = ["btai-theme", "btai-api-base", "btai-admin-key"];

export function Settings() {
  const { theme, preference, setPreference } = useTheme();
  const { toast } = useToast();
  const [apiBase, setApiBaseState] = useState(getApiBase());
  const [apiBaseDirty, setApiBaseDirty] = useState(false);
  const [adminKey, setAdminKeyState] = useState(getAdminKey());
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const saveApiBase = () => {
    setApiBase(apiBase);
    setApiBaseDirty(false);
    toast("Service address saved. Reloading status…", "success");
    setHealth(null);
    api.health().then(setHealth).catch(() => setHealth(null));
  };

  const saveAdminKey = () => {
    setAdminKey(adminKey);
    toast("Admin key stored locally in this browser.", "success");
  };

  const clearSensitive = () => {
    if (!window.confirm("Clear stored admin key and service address from this browser?")) return;
    STORAGE_KEYS.forEach((key) => localStorage.removeItem(key));
    setApiBaseState("/api/v1");
    setAdminKeyState("");
    setHealth(null);
    window.location.reload();
  };

  return (
    <div className="page" style={{ maxWidth: 720 }}>
      <header className="page-header">
        <h1 className="page-title">Settings</h1>
        <p className="page-subtitle">
          Appearance, service connection and local credentials. Everything here is stored in
          this browser only.
        </p>
      </header>

      <section className="card section">
        <h2 className="card-title mb-3">Appearance</h2>
        <div className="grid grid-3">
          {(
            [
              ["light", "Light"],
              ["dark", "Dark"],
              ["system", "System"],
            ] as [ThemePreference, string][]
          ).map(([value, label]) => (
            <button
              type="button"
              key={value}
              className={`btn${preference === value ? " btn-primary" : ""}`}
              onClick={() => setPreference(value)}
            >
              {label}
              {preference === value && <span className="active-tag"> · active</span>}
            </button>
          ))}
        </div>
        <p className="text-xs text-faint mt-3">
          Current theme: <strong>{theme}</strong>. “System” follows your device preference.
        </p>
      </section>

      <section className="card section">
        <h2 className="card-title mb-3">Analysis service</h2>
        <label className="text-xs text-faint" htmlFor="api-base">
          API base URL
        </label>
        <div className="flex gap-3 mt-2">
          <input
            id="api-base"
            className="input flex-1"
            value={apiBase}
            onChange={(event) => {
              setApiBaseState(event.target.value);
              setApiBaseDirty(true);
            }}
            placeholder="/api/v1 or http://localhost:8000/api/v1"
          />
          <button type="button" className="btn btn-primary" onClick={saveApiBase} disabled={!apiBaseDirty}>
            Save
          </button>
        </div>
        <p className="text-xs text-faint mt-3">
          When the frontend runs behind the backend proxy, use <code>/api/v1</code>. Otherwise
          use the full backend address.
        </p>
        <div className="mt-3">
          {health ? (
            <span className={`badge ${health.status === "ok" ? "badge-success" : "badge-danger"}`}>
              {health.status === "ok" ? "Service online" : "Service unreachable"}
            </span>
          ) : (
            <span className="text-xs text-faint">Checking service status…</span>
          )}
          {health && (
            <span className="text-xs text-faint"> · {health.model?.name ?? "no active model"} · {health.environment}</span>
          )}
        </div>
      </section>

      <section className="card section">
        <h2 className="card-title mb-3">Administration</h2>
        <p className="text-sm text-muted mb-3">
          Required only for model activation, rescanning the registry and analytics. The key is
          never transmitted to any third party and lives only in this browser’s local storage.
        </p>
        <label className="text-xs text-faint" htmlFor="admin-key">
          Admin key (<code>X-Admin-Key</code> header)
        </label>
        <div className="flex gap-3 mt-2">
          <input
            id="admin-key"
            className="input flex-1"
            type="password"
            value={adminKey}
            onChange={(event) => setAdminKeyState(event.target.value)}
            placeholder="Leave empty to disable admin actions"
          />
          <button type="button" className="btn btn-primary" onClick={saveAdminKey}>
            Save
          </button>
        </div>
      </section>

      <section className="card section">
        <h2 className="card-title mb-3">Danger zone</h2>
        <p className="text-sm text-muted mb-3">
          Remove locally stored settings and admin key. Analyses and reports remain in the
          backend database.
        </p>
        <button type="button" className="btn" onClick={clearSensitive}>
          Clear browser data
        </button>
      </section>
    </div>
  );
}