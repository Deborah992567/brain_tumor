import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../api/client";
import type { HealthResponse } from "../api/types";
import { Icon, type IconName } from "./Icon";
import { Spinner } from "./States";
import { useTheme } from "../theme";

interface NavItem {
  to: string;
  label: string;
  icon: IconName;
}

const NAV: NavItem[] = [
  { to: "/", label: "Dashboard", icon: "dashboard" },
  { to: "/analysis/new", label: "New Analysis", icon: "upload" },
  { to: "/history", label: "Analysis History", icon: "history" },
  { to: "/reports", label: "Reports", icon: "report" },
  { to: "/models", label: "Models", icon: "model" },
  { to: "/research", label: "Research", icon: "research" },
  { to: "/settings", label: "Settings", icon: "settings" },
];

export function AppLayout() {
  return (
    <div className="app-shell">
      <Sidebar />
      <div className="app-main">
        <Topbar />
        <main className="app-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-mark" aria-hidden="true">
          BT
        </div>
        <div>
          <div className="brand-name">Brain Tumor AI</div>
          <div className="brand-subtitle">AI-Assisted MRI Analysis</div>
        </div>
      </div>
      <nav className="nav" aria-label="Main navigation">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
          >
            <Icon name={item.icon} className="nav-icon" />
            {item.label}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-footer">
        <span>Research &amp; educational use</span>
        <span>Not a medical diagnosis</span>
      </div>
    </aside>
  );
}

function Topbar() {
  const { theme, preference, setPreference } = useTheme();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    api
      .health()
      .then((h) => {
        if (!cancelled) setHealth(h);
      })
      .catch(() => {
        if (!cancelled) setHealth(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const nextPreference =
    preference === "dark" ? "light" : preference === "light" ? "system" : "dark";

  return (
    <header className="topbar">
      <div className="flex flex-1 flex-between">
        <h1 className="topbar-title">AI-Assisted Brain MRI Analysis</h1>
      </div>
      <div className="topbar-actions">
        <div className="flex gap-2" title={health?.status === "ok" ? "Service online" : "Service unreachable"}>
          {loading ? (
            <Spinner size={16} />
          ) : health?.status === "ok" ? (
            <span className="badge badge-success">Service online</span>
          ) : (
            <span className="badge badge-danger">Service offline</span>
          )}
        </div>
        <button
          type="button"
          className="theme-toggle"
          onClick={() => setPreference(nextPreference)}
          aria-label={`Switch theme (currently ${theme}). ${nextPreference}`}
        >
          <Icon name={theme === "dark" ? "moon" : "sun"} size={15} />
          {theme === "dark" ? "Dark" : "Light"}
        </button>
      </div>
    </header>
  );
}