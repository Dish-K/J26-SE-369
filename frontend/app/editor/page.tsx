import type { Metadata } from "next";
import Link from "next/link";

import CodeEditor from "./CodeEditor";

export const metadata: Metadata = {
  title: "Editor",
  description: "CodeTrace editor page.",
};

export default function EditorPage() {
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
      <main className="placeholder-main page-gutter">
        <CodeEditor />
      </main>
    </div>
  );
}
