import styles from "../explainability.module.css";
import { CLASS_LABELS, featureLabel, featureValue, score, signed } from "../format";
import type { ClassName, FeatureInfo, LimeResult, ShapResult } from "../types";

type FeatureMap = Map<string, FeatureInfo>;

function sentence(label: string, value: string, positive: boolean, target: ClassName): string {
  const verb = positive ? "increased" : "reduced";
  return `${label} (${value}) ${verb} the model's ${CLASS_LABELS[target]} score.`;
}

export function ShapPanel({ shap, features }: { shap: ShapResult; features: FeatureMap }) {
  const max = Math.max(...shap.features.map((f) => f.absolute_shap_value), 1e-12);
  const top = shap.features.slice(0, 3);

  return (
    <section className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>SHAP feature contributions</h2>
        <span className={styles.overline}>{shap.explainer_type}</span>
      </div>
      <ul className={styles.sentenceList}>
        {top.map((f) => {
          const info = features.get(f.name);
          return (
            <li className={styles.sentence} key={f.name}>
              {sentence(featureLabel(info, f.name), featureValue(info, f.value), f.shap_value >= 0,
                shap.target_class)}
            </li>
          );
        })}
      </ul>
      <div>
        {shap.features.map((f) => {
          const info = features.get(f.name);
          const width = `${(f.absolute_shap_value / max) * 50}%`;
          return (
            <div className={styles.barRow} key={f.name}>
              <span className={styles.featureName}>
                {featureLabel(info, f.name)}
                <span className={styles.featureValue}>{featureValue(info, f.value)}</span>
              </span>
              <div className={styles.shapTrack}>
                <div
                  className={f.shap_value >= 0 ? styles.shapToward : styles.shapAway}
                  style={{ width }}
                />
              </div>
              <span className={styles.barValue}>{signed(f.shap_value, 2)}</span>
            </div>
          );
        })}
      </div>
      <p className={styles.note}>
        Red pushes toward {CLASS_LABELS[shap.target_class]}, grey pushes away. Contributions are in
        the model&apos;s {shap.output_space} output space ({shap.output_description}). Check: base
        value {shap.base_value.toFixed(3)} + contributions = {shap.reconstructed_output.toFixed(3)},
        matching the model output {shap.model_output.toFixed(3)} (error{" "}
        {shap.reconstruction_error.toExponential(1)}).
      </p>
    </section>
  );
}

export function LimePanel({ lime, features }: { lime: LimeResult; features: FeatureMap }) {
  return (
    <section className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>LIME local rules</h2>
        <span className={styles.overline}>
          seed {lime.seed} · {lime.num_samples} samples
        </span>
      </div>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Rule</th>
            <th>Weight</th>
            <th>Effect</th>
          </tr>
        </thead>
        <tbody>
          {lime.features.slice(0, 6).map((f) => (
            <tr key={f.name}>
              <td>
                <span className={styles.mono}>{f.condition}</span>
                <div className={styles.featureValue}>
                  {featureLabel(features.get(f.name), f.name)}:{" "}
                  {featureValue(features.get(f.name), f.value)}
                </div>
              </td>
              <td className={styles.mono}>{signed(f.weight, 3)}</td>
              <td className={f.weight >= 0 ? styles.toward : styles.away}>
                {f.weight >= 0 ? "toward" : "away"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className={styles.note}>
        LIME fits a simple local model around this one window. Its local fit (R²) is{" "}
        {score(lime.score)}; a low fit means the simple model describes the classifier only
        loosely here, and it is shown rather than hidden.
      </p>
    </section>
  );
}
