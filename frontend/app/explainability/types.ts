// Response shapes of the C4 explainability API (backend/app/explainability/api_schemas.py).

export type ClassName = "no_ai" | "ide_ai" | "external_ai";
export type Direction = "toward_target" | "away_from_target" | "neutral";
export type MetricStatus = "computed" | "not_computable";

export interface DataSource {
  kind: string;
  label: string;
  note: string;
}

export interface FlagSummary {
  flag_id: string;
  title: string;
  task_id: string;
  category: ClassName;
  confidence: number;
  window_count: number;
  model_version: string;
}

export interface FlagList {
  data_source: DataSource;
  flags: FlagSummary[];
}

export interface Provenance {
  prediction_id: string;
  feature_vector_id: string;
  model_version: string;
  target_class: ClassName;
  target_class_index: number;
}

export interface ShapFeature {
  name: string;
  value: number | null;
  shap_value: number;
  absolute_shap_value: number;
  rank: number;
  direction: Direction;
}

export interface ShapResult extends Provenance {
  model_family: string;
  target_probability: number;
  strategy: string;
  explainer_type: string;
  output_space: string;
  output_description: string;
  background_version: string;
  base_value: number;
  model_output: number;
  reconstructed_output: number;
  reconstruction_error: number;
  reconstruction_tolerance: number;
  features: ShapFeature[];
}

export interface LimeFeature {
  name: string;
  feature_index: number;
  value: number | null;
  condition: string;
  weight: number;
  absolute_weight: number;
  rank: number;
  direction: Direction;
}

export interface LimeResult extends Provenance {
  target_probability: number;
  seed: number;
  num_samples: number;
  score: number | null;
  intercept: number | null;
  features: LimeFeature[];
}

export interface AgreementAtK {
  k: number;
  shap_top_features: string[];
  lime_top_features: string[];
  intersection: string[];
  overlap_at_k: number;
  jaccard: number;
  feature_set_agreement: number;
  direction_agreement: number;
  direction_status: string;
  status: MetricStatus;
  agreement_score: number | null;
}

export interface AgreementResult extends Provenance {
  version: string;
  spearman_raw: number | null;
  kendall_raw: number | null;
  rank_agreement: number | null;
  at_k: AgreementAtK[];
  primary_top_k: number;
  status: MetricStatus;
  score: number | null;
  reason: string | null;
}

export interface StabilityAtK {
  k: number;
  set_stability: number | null;
  min_pairwise_jaccard: number | null;
  stability_score: number | null;
}

export interface StabilitySummary extends Provenance {
  version: string;
  repetitions_requested: number;
  successful_runs: number;
  pair_count: number;
  rank_stability: number | null;
  at_k: StabilityAtK[];
  primary_top_k: number;
  status: MetricStatus;
  score: number | null;
  reason: string | null;
  lime_fit: { count: number; mean: number | null; minimum: number | null; maximum: number | null };
}

export interface FaithfulnessAtK {
  k: number;
  top_features: string[];
  baseline_features: string[];
  p_top: number;
  p_base: number;
  d_top: number;
  d_base: number;
  status: MetricStatus;
  score: number | null;
}

export interface FaithfulnessResult extends Provenance {
  version: string;
  ranking_source: string;
  perturbation_method: string;
  perturbation_samples: number;
  p0: number;
  at_k: FaithfulnessAtK[];
  primary_top_k: number;
  status: MetricStatus;
  score: number | null;
  reason: string | null;
}

export interface EcsResult extends Provenance {
  version: string;
  status: MetricStatus;
  agreement_score: number | null;
  stability_score: number | null;
  faithfulness_score: number | null;
  weights: Record<string, number>;
  score: number | null;
  missing_components: string[];
  reason: string | null;
}

export interface EcsScenario {
  name: string;
  weights: Record<string, number>;
  status: MetricStatus;
  score: number | null;
}

export interface FeatureInfo {
  name: string;
  value: number | null;
  label: string | null;
  unit: string | null;
  display_as: string | null;
}

export interface TimelineWindow {
  prediction_id: string;
  window_start: string;
  window_end: string;
  predicted_category: ClassName;
  confidence: number;
  ai_positive: boolean;
  in_segment: boolean;
  is_representative: boolean;
}

export interface Analysis {
  data_source: DataSource;
  flag: {
    flag_id: string;
    task_id: string;
    category: ClassName;
    confidence: number;
    evidence_kinds: string[];
    window_count: number;
    detected_start_line: number;
    detected_end_line: number;
    representative_prediction_id: string;
  };
  representative_prediction: Provenance & {
    predicted_category: ClassName;
    class_probabilities: Record<ClassName, number>;
    confidence: number;
  };
  feature_vector: { feature_vector_id: string; schema_version: string; run_id: string };
  model: {
    model_version: string;
    model_family: string;
    feature_schema_version: string;
    classification_threshold: number;
    class_names: ClassName[];
    target_class: ClassName;
  };
  features: FeatureInfo[];
  timeline: TimelineWindow[];
  shap: ShapResult;
  lime: LimeResult;
  agreement: AgreementResult;
  stability: StabilitySummary;
  faithfulness: FaithfulnessResult;
  ecs: EcsResult;
  ecs_sensitivity: { baseline_version: string; scenarios: EcsScenario[] };
  planned: Record<string, string>;
}
