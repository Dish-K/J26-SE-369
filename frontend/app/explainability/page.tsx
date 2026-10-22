import type { Metadata } from "next";
import Link from "next/link";

import ExplainabilityDashboard from "./ExplainabilityDashboard";
import styles from "./explainability.module.css";

export const metadata: Metadata = {
  title: "Explainability",
  description: "Interviewer view of CodeTrace classifier explanations (C4).",
};

export default function ExplainabilityPage() {
  return (
    <div className="placeholder-page">
      <header className="placeholder-header page-gutter">
        <Link className="wordmark" href="/" aria-label="CodeTrace home">
          CodeTrace<span className="wordmark-period">.</span>
        </Link>
        <Link className="text-link" href="/">
          Back to home <span aria-hidden="true">↗</span>
        </Link>
      </header>
      <main className={`${styles.main} page-gutter`}>
        <ExplainabilityDashboard />
      </main>
    </div>
  );
}
