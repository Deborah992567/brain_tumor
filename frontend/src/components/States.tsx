import type { ReactNode } from "react";
import { Icon } from "./Icon";

export function Spinner({ size = 22 }: { size?: number }) {
  return <span className="spinner" role="status" style={{ width: size, height: size }} aria-label="Loading" />;
}

export function PageLoader({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex flex-center" style={{ padding: "48px 0", gap: "12px" }}>
      <Spinner />
      <span className="text-muted text-sm">{label}</span>
    </div>
  );
}

export function SkeletonRows({ rows = 4 }: { rows?: number }) {
  return (
    <div aria-label="Loading content">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton skeleton-block" style={{ height: 96 }} />
      ))}
    </div>
  );
}

export function EmptyState({
  title,
  hint,
  action,
}: {
  title: string;
  hint: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <Icon name="image" size={40} className="text-faint" />
      <div className="empty-title">{title}</div>
      <p className="empty-hint">{hint}</p>
      {action}
    </div>
  );
}

export function ErrorState({ message, action }: { message: string; action?: ReactNode }) {
  return (
    <div className="error-state" role="alert">
      <Icon name="alert" size={40} className="text-faint" />
      <div className="empty-title">Something went wrong</div>
      <p className="empty-hint">{message}</p>
      {action}
    </div>
  );
}