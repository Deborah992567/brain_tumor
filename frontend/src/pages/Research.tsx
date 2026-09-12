import { Icon } from "../components/Icon";
import { CLASS_LABELS } from "../api/types";

export function Research() {
  return (
    <div className="page" style={{ maxWidth: 920 }}>
      <header className="page-header">
        <h1 className="page-title">Research</h1>
        <p className="page-subtitle">
          How the system works, what the classes mean, and how to interpret the outputs.
        </p>
      </header>

      <section className="card section">
        <h2 className="card-title mb-3">Tumor classes</h2>
        <p className="text-sm text-muted mb-4">
          The active CNN produces a probability for each of four categories. The class with
          the highest probability is reported, together with the full distribution.
        </p>
        <div style={{ display: "grid", gap: 12 }}>
          {[
            {
              label: CLASS_LABELS.glioma,
              note: "The most common primary brain tumor class in adults, arising from glial support cells.",
            },
            {
              label: CLASS_LABELS.meningioma,
              note: "Usually a slow-growing tumor that develops from the membranes around the brain and spinal cord.",
            },
            {
              label: CLASS_LABELS.no_tumor,
              note: "No tumor region detected within the scan at the model’s operating threshold.",
            },
            {
              label: CLASS_LABELS.pituitary_tumor,
              note: "A growth in the pituitary gland at the base of the skull, often detectable on mid-line views.",
            },
          ].map((item) => (
            <div key={item.label} className="flex gap-3" style={{ alignItems: "flex-start" }}>
              <span className="badge" style={{ flexShrink: 0 }}>
                {item.label}
              </span>
              <span className="text-sm text-muted">{item.note}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="card section">
        <h2 className="card-title mb-3">Model probability and trust threshold</h2>
        <p className="text-sm text-muted">
          Each result carries a <strong>model probability</strong> — the softmax output for the
          predicted class. When the top probability falls below <strong>60%</strong>, the analysis
          is flagged <em>unsupported / classification unavailable</em> and a caution note is
          attached. The model’s probabilities are not calibrated: a high value does not imply
          medical certainty, and an out-of-distribution image (for example a tumor type outside
          the four supported classes) can still get a high nominal probability. Never treat any
          result as a diagnosis.
        </p>
      </section>

      <section className="card section">
        <h2 className="card-title mb-3">AI attention visualization (Grad-CAM)</h2>
        <p className="text-sm text-muted">
          Grad-CAM highlights the regions of the image that most influenced the model’s
          decision. The heatmap is overlaid on the original scan and labeled{" "}
          <strong>AI attention visualization</strong>. It shows “what the model looked at”, not
          a segmentation of a tumor. It is an explanatory aid, not an anatomical outline.
        </p>
      </section>

      <section className="card section">
        <h2 className="card-title mb-3">Methodology</h2>
        <ul className="list">
          <li>
            Images are preprocessed with a single shared pipeline (RGB, bilinear resize,
            <code> 1/255</code>) that is used identically by training and inference.
          </li>
          <li>
            The pipeline enforces file-level validation: extension, MIME type, size limit,
            near-blank detection, and decode checks with clear error messages.
          </li>
          <li>
            Every analysis is persisted with the exact model version, full probability vector,
            processing time and inputs, so results are auditable.
          </li>
          <li>
            All metrics shown on the Models page are training-time values; they are not
            re-validated against a held-out clinical set and may not generalize.
          </li>
        </ul>
      </section>

      <section className="card section">
        <h2 className="card-title mb-3">Limitations</h2>
        <ul className="list">
          <li>This system is for research and educational use only — not CE/FDA cleared.</li>
          <li>The bundled model is a small demonstration network with modest accuracy.</li>
          <li>Cross-architecture generalization has not been verified on external scanners.</li>
          <li>The training dataset is not distributed with this repository.</li>
        </ul>
        <div className="alert alert-info mt-4" role="note" style={{ alignItems: "flex-start" }}>
          <Icon name="alert" size={18} />
          <span className="text-sm">
            Reference dataset (not bundled): Kaggle “Brain Tumor MRI” 4-class collection
            (glioma, meningioma, no tumor, pituitary tumor).
          </span>
        </div>
      </section>
    </div>
  );
}