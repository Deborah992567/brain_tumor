import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ToastProvider } from "../components/Toasts";
import { AnalysisView } from "../components/AnalysisView";
import type { PredictionResponse } from "../api/types";

const result: PredictionResponse = {
  analysis_id: "abc-123",
  prediction: "no_tumor",
  label: "No Tumor",
  confidence: 0.9972,
  low_confidence: false,
  probabilities: {
    glioma: 0.0004,
    meningioma: 0.0014,
    no_tumor: 0.9972,
    pituitary_tumor: 0.001,
  },
  model: {
    name: "Brain Tumor CNN",
    version: "1.0.0",
    architecture: "Custom CNN",
    input_size: 64,
    dataset_version: "Demo",
  },
  processing_time_ms: 172,
  created_at: "2026-01-01T00:00:00Z",
  report_available: false,
  links: {
    image: "/media/img.png",
    gradcam: "/media/gradcam.png",
  },
};

describe("AnalysisView", () => {
  it("renders the top prediction and all probabilities", () => {
    render(
      <ToastProvider>
        <AnalysisView result={result} />
      </ToastProvider>,
    );

    expect(screen.getAllByText("No Tumor").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Confidence 99.7%")).toBeInTheDocument();
    expect(screen.getAllByText("Glioma").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Meningioma").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Pituitary Tumor")).toBeInTheDocument();
  });

  it("labels the attention overlay and includes the disclaimer", () => {
    render(
      <ToastProvider>
        <AnalysisView result={result} />
      </ToastProvider>,
    );

    expect(screen.getByText("AI attention visualization")).toBeInTheDocument();
    expect(screen.getByText(/does not constitute a medical diagnosis/i)).toBeInTheDocument();
  });

  it("shows a low-confidence warning when the threshold is missed", () => {
    const low: PredictionResponse = { ...result, confidence: 0.52, low_confidence: true };
    render(
      <ToastProvider>
        <AnalysisView result={low} />
      </ToastProvider>,
    );
    expect(screen.getByText("Low confidence")).toBeInTheDocument();
  });
});