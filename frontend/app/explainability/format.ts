import type { ClassName, FeatureInfo } from "./types";

export const CLASS_LABELS: Record<ClassName, string> = {
  no_ai: "No AI",
  ide_ai: "IDE AI",
  external_ai: "External AI",
};

// C2 evidence kinds describe what was observed. They are never the reason
// for the class, so they are phrased as observations.
export const EVIDENCE_LABELS: Record<string, string> = {
  paste: "Paste-related behaviour",
  suggestion_accepted: "Accepted editor suggestion",
  typing_burst: "Burst of fast typing",
  uniform_intervals: "Unusually uniform typing rhythm",
  no_typing_errors: "Very few typing corrections",
};

export function percent(value: number | null | undefined, digits = 1): string {
  return value == null ? "—" : `${(value * 100).toFixed(digits)}%`;
}

export function score(value: number | null | undefined, digits = 3): string {
  return value == null ? "not computable" : value.toFixed(digits);
}

export function signed(value: number, digits = 3): string {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

/** Display a raw feature value in the unit the shared feature registry gives it. */
export function featureValue(feature: FeatureInfo | undefined, value: number | null): string {
  if (value == null) return "not measured";
  switch (feature?.unit) {
    case "fraction":
      return percent(value, 0);
    case "ms":
      return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
    case "s":
      return `${value.toFixed(1)} s`;
    case "count":
    case "characters":
      return Number.isInteger(value) ? value.toString() : value.toFixed(1);
    default:
      return Number.isInteger(value) ? value.toString() : value.toFixed(2);
  }
}

export function featureLabel(feature: FeatureInfo | undefined, name: string): string {
  return feature?.label ?? name.replaceAll("_", " ");
}

export function clock(iso: string): string {
  return new Date(iso).toISOString().slice(11, 19);
}

export function shortId(id: string): string {
  return `${id.slice(0, 8)}…`;
}
