import styles from "../explainability.module.css";
import { percent, score } from "../format";
import type {
  AgreementResult,
  EcsResult,
  EcsScenario,
  FaithfulnessResult,
  StabilitySummary,
} from "../types";

const SCENARIO_LABELS: Record<string, string> = {
  equal_weight: "Equal weights (baseline)",
  agreement_emphasis: "Agreement emphasis",
  stability_emphasis: "Stability emphasis",
  faithfulness_emphasis: "Faithfulness emphasis",
};

function Meter({ label, value, detail }: { label: string; value: number | null; detail: string }) {
  return (
    <div>
      <div className={styles.barRow}>
        <span>{label}</span>
        <div className={styles.barTrack}>
          <div className={styles.barFill} style={{ width: `${(value ?? 0) * 100}%` }} />
        </div>
        <span className={styles.barValue}>{score(value)}</span>
      </div>
      <p className={styles.note}>{detail}</p>
    </div>
  );
}

export function ReliabilityPanel({
  agreement,
  stability,
  faithfulness,
  ecs,
  scenarios,
}: {
  agreement: AgreementResult;
  stability: StabilitySummary;
  faithfulness: FaithfulnessResult;
  ecs: EcsResult;
  scenarios: EcsScenario[];
}) {
  const agreeK = agreement.at_k.find((r) => r.k === agreement.primary_top_k);
  const stableK = stability.at_k.find((r) => r.k === stability.primary_top_k);
  const faithK = faithfulness.at_k.find((r) => r.k === faithfulness.primary_top_k);

  return (
    <section className={`${styles.panel} ${styles.panelWide}`}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Explanation reliability</h2>
        <span className={styles.overline}>{ecs.version}</span>
      </div>

      <div className={styles.grid}>
        <div className={styles.stat}>
          <span className={styles.muted}>Explanation Confidence Score (ECS)</span>
          <span className={styles.bigNumber}>{score(ecs.score)}</span>
          <p className={styles.note}>
            A composite reliability indicator: the equal-weight mean of Agreement, Stability and
            Faithfulness. It is not a probability that the explanation is correct.
            {ecs.reason ? ` ${ecs.reason}` : ""}
          </p>
        </div>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Weighting scenario</th>
              <th>ECS</th>
            </tr>
          </thead>
          <tbody>
            {scenarios.map((s) => (
              <tr key={s.name}>
                <td>{SCENARIO_LABELS[s.name] ?? s.name}</td>
                <td className={styles.mono}>{score(s.score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Meter
        label="Agreement"
        value={agreement.score}
        detail={
          agreeK
            ? `Do SHAP and LIME agree? Top-${agreeK.k} feature overlap ${percent(agreeK.overlap_at_k, 0)}, ` +
              `ranking agreement ${score(agreement.rank_agreement)}, ` +
              `direction agreement ${percent(agreeK.direction_agreement, 0)}.`
            : "Do SHAP and LIME agree?"
        }
      />
      <Meter
        label="Stability"
        value={stability.score}
        detail={
          `Does LIME give the same answer when re-run? ${stability.successful_runs} of ` +
          `${stability.repetitions_requested} seeded runs, ${stability.pair_count} run pairs: ` +
          `Top-${stability.primary_top_k} set stability ${score(stableK?.set_stability ?? null)}, ` +
          `rank stability ${score(stability.rank_stability)}.`
        }
      />
      <Meter
        label="Faithfulness"
        value={faithfulness.score}
        detail={
          faithK
            ? `Do the top features really move the model? Replacing the top ${faithK.k} SHAP ` +
              `features drops the ${percent(faithfulness.p0, 2)} probability to ${percent(faithK.p_top, 1)}; ` +
              `replacing the bottom ${faithK.k} leaves it at ${percent(faithK.p_base, 1)}.`
            : "Do the top features really move the model?"
        }
      />
    </section>
  );
}
