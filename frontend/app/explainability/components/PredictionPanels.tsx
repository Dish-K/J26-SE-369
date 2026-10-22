import styles from "../explainability.module.css";
import { CLASS_LABELS, EVIDENCE_LABELS, clock, percent } from "../format";
import type { Analysis, ClassName, TimelineWindow } from "../types";

export function PredictionSummary({ analysis }: { analysis: Analysis }) {
  const prediction = analysis.representative_prediction;
  const target = analysis.model.target_class;
  const order: ClassName[] = ["no_ai", "ide_ai", "external_ai"];

  return (
    <section className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Model prediction</h2>
        <span className={styles.overline}>representative window</span>
      </div>
      <div className={styles.statRow}>
        <div className={styles.stat}>
          <span className={styles.muted}>Predicted class</span>
          <span className={styles.bigNumber}>{CLASS_LABELS[prediction.predicted_category]}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.muted}>Model confidence</span>
          <span className={styles.bigNumber}>{percent(prediction.confidence, 2)}</span>
        </div>
      </div>
      <div>
        {order.map((name) => (
          <div className={styles.barRow} key={name}>
            <span>{CLASS_LABELS[name]}</span>
            <div className={styles.barTrack}>
              <div
                className={`${styles.barFill} ${name === target ? styles.barFillTarget : ""}`}
                style={{ width: `${prediction.class_probabilities[name] * 100}%` }}
              />
            </div>
            <span className={styles.barValue}>{percent(prediction.class_probabilities[name], 2)}</span>
          </div>
        ))}
      </div>
      <p className={styles.note}>
        Model {analysis.model.model_version} ({analysis.model.model_family}). A window counts as
        AI-positive only at or above the model&apos;s threshold of{" "}
        {percent(analysis.model.classification_threshold, 0)}. Model confidence is not the
        probability that the candidate used AI.
      </p>
    </section>
  );
}

export function EvidencePanel({ kinds }: { kinds: string[] }) {
  return (
    <section className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Observed behaviour</h2>
        <span className={styles.overline}>C2 evidence metadata</span>
      </div>
      <div className={styles.chips}>
        {kinds.map((kind) => (
          <span className={styles.chip} key={kind}>
            {EVIDENCE_LABELS[kind] ?? kind}
          </span>
        ))}
      </div>
      <p className={styles.note}>
        These behaviours were observed in this segment. They are descriptive evidence, not the
        reason for the prediction. The model&apos;s reasoning is shown in the SHAP and LIME panels
        below.
      </p>
    </section>
  );
}

function windowClass(window: TimelineWindow): string {
  return [
    styles.window,
    window.in_segment ? styles.windowSegment : "",
    window.is_representative ? styles.windowRepresentative : "",
  ].join(" ");
}

export function Timeline({ windows }: { windows: TimelineWindow[] }) {
  return (
    <section className={`${styles.panel} ${styles.panelWide}`}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Task timeline</h2>
        <span className={styles.overline}>every 30 s window, one every 10 s</span>
      </div>
      <div className={styles.timeline}>
        {windows.map((window) => (
          <div className={windowClass(window)} key={window.prediction_id}>
            <span className={styles.mono}>
              {clock(window.window_start)}–{clock(window.window_end)}
            </span>
            <span className={`${styles.windowClass} ${window.ai_positive ? styles.windowAi : ""}`}>
              {CLASS_LABELS[window.predicted_category]}
            </span>
            <span className={styles.muted}>{percent(window.confidence, 2)}</span>
            {window.ai_positive && <span className={styles.badge}>AI+</span>}
            {window.is_representative && (
              <span className={`${styles.badge} ${styles.badgeAccent}`}>★ explained</span>
            )}
          </div>
        ))}
      </div>
      <div className={styles.legend}>
        <span>Red border: windows merged into this flagged segment</span>
        <span>★ explained: the representative (highest-confidence) window</span>
      </div>
      <p className={styles.note}>
        The explanations below describe this exact representative prediction, not an average of
        the segment&apos;s windows.
      </p>
    </section>
  );
}
