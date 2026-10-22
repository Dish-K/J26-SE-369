"use client";

import { useEffect, useMemo, useState } from "react";

import { EvidencePanel, PredictionSummary, Timeline } from "./components/PredictionPanels";
import { LimePanel, ShapPanel } from "./components/AttributionPanels";
import { ReliabilityPanel } from "./components/ReliabilityPanel";
import styles from "./explainability.module.css";
import { shortId } from "./format";
import type { Analysis, FeatureInfo, FlagList } from "./types";

const API = "/api/explainability/demo";

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} from ${url}`);
  }
  return (await response.json()) as T;
}

function Provenance({ analysis }: { analysis: Analysis }) {
  const rows: [string, string][] = [
    ["Flag", analysis.flag.flag_id],
    ["Representative prediction", analysis.representative_prediction.prediction_id],
    ["Feature vector", analysis.feature_vector.feature_vector_id],
    ["Model version", analysis.model.model_version],
    ["Feature schema", analysis.model.feature_schema_version],
    ["XAI background", analysis.shap.background_version],
    ["Metric versions",
      `${analysis.agreement.version} · ${analysis.stability.version} · ${analysis.faithfulness.version}`],
  ];
  return (
    <section className={`${styles.panel} ${styles.panelWide}`}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Provenance</h2>
        <span className={styles.overline}>every result traces back to one C2 prediction</span>
      </div>
      <dl className={styles.provenance}>
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

export default function ExplainabilityDashboard() {
  const [flags, setFlags] = useState<FlagList | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [analyses, setAnalyses] = useState<Record<string, Analysis>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getJson<FlagList>(`${API}/flags`)
      .then((list) => {
        if (cancelled) return;
        setFlags(list);
        setSelected(list.flags[0]?.flag_id ?? null);
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(String(reason));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selected || analyses[selected]) return;
    let cancelled = false;
    getJson<Analysis>(`${API}/flags/${selected}/analysis`)
      .then((analysis) => {
        if (!cancelled) setAnalyses((previous) => ({ ...previous, [selected]: analysis }));
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(String(reason));
      });
    return () => {
      cancelled = true;
    };
  }, [selected, analyses]);

  const analysis = selected ? analyses[selected] : undefined;
  const features = useMemo(
    () => new Map<string, FeatureInfo>((analysis?.features ?? []).map((f) => [f.name, f])),
    [analysis],
  );

  if (error) {
    return (
      <div className={styles.error} role="alert">
        <strong>The explainability service is not reachable.</strong>
        <p className={styles.note}>
          Start the backend (uvicorn backend.app.main:app) and reload. Details: {error}
        </p>
      </div>
    );
  }

  if (!flags) {
    return <div className={styles.state}>Loading flagged segments…</div>;
  }

  return (
    <>
      <div className={styles.banner} role="note">
        <span className={styles.bannerLabel}>{flags.data_source.label}</span>
        <span className={styles.bannerNote}>{flags.data_source.note}</span>
      </div>

      <div className={styles.titleRow}>
        <div>
          <h1 className={styles.title}>Why did the model flag this segment?</h1>
          <p className={styles.subtitle}>
            C4 traces a flag back to the exact classifier prediction behind it, explains that
            prediction with SHAP and LIME, and measures how reliable those explanations are.
          </p>
        </div>
        <div className={styles.tabs} role="tablist" aria-label="Flagged segments">
          {flags.flags.map((flag) => (
            <button
              key={flag.flag_id}
              type="button"
              role="tab"
              aria-selected={flag.flag_id === selected}
              className={`${styles.tab} ${flag.flag_id === selected ? styles.tabActive : ""}`}
              onClick={() => setSelected(flag.flag_id)}
            >
              {flag.title} · {flag.window_count} window{flag.window_count === 1 ? "" : "s"}
            </button>
          ))}
        </div>
      </div>

      {!analysis ? (
        <div className={styles.state}>
          Explaining flag {selected ? shortId(selected) : ""} (SHAP, LIME, 20 LIME re-runs,
          perturbation tests)…
        </div>
      ) : (
        <div className={styles.grid}>
          <PredictionSummary analysis={analysis} />
          <EvidencePanel kinds={analysis.flag.evidence_kinds} />
          <Timeline windows={analysis.timeline} />
          <ShapPanel shap={analysis.shap} features={features} />
          <LimePanel lime={analysis.lime} features={features} />
          <ReliabilityPanel
            agreement={analysis.agreement}
            stability={analysis.stability}
            faithfulness={analysis.faithfulness}
            ecs={analysis.ecs}
            scenarios={analysis.ecs_sensitivity.scenarios}
          />
          <section className={`${styles.panel} ${styles.panelWide}`}>
            <div className={styles.panelHeader}>
              <h2 className={styles.panelTitle}>Next phases</h2>
            </div>
            <ul className={styles.sentenceList}>
              {Object.entries(analysis.planned).map(([key, text]) => (
                <li className={styles.sentence} key={key}>
                  {text}
                </li>
              ))}
            </ul>
          </section>
          <Provenance analysis={analysis} />
        </div>
      )}
    </>
  );
}
