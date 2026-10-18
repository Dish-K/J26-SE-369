import type { editor } from "monaco-editor";

import packageJson from "@/package.json";

export const MONACO_VERSION = packageJson.dependencies["monaco-editor"];

// Text the editor inserts without a keystroke breaks the keydown-to-edit mapping.
const RESEARCH_V1: editor.IStandaloneEditorConstructionOptions = {
  quickSuggestions: false,
  suggestOnTriggerCharacters: false,
  wordBasedSuggestions: "off",
  parameterHints: { enabled: false },
  inlineSuggest: { enabled: false },
  acceptSuggestionOnEnter: "off",
  tabCompletion: "off",
  snippetSuggestions: "none",
  autoClosingBrackets: "never",
  autoClosingQuotes: "never",
  autoClosingComments: "never",
  autoClosingOvertype: "never",
  autoClosingDelete: "never",
  autoSurround: "never",
  autoIndent: "none",
  autoIndentOnPaste: false,
  formatOnType: false,
  formatOnPaste: false,
  dragAndDrop: false,
  dropIntoEditor: { enabled: false },
  pasteAs: { enabled: false },
  multiCursorLimit: 1,
  columnSelection: false,
  linkedEditing: false,
  tabSize: 4,
  insertSpaces: true,
  detectIndentation: false,
  wordWrap: "off",
  fontSize: 14,
};

export const EDITOR_PROFILES = { "research-v1": RESEARCH_V1 } as const;
export type EditorProfile = keyof typeof EDITOR_PROFILES;
export const ACTIVE_PROFILE: EditorProfile = "research-v1";

function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  const entries = Object.entries(value).sort(([a], [b]) => (a < b ? -1 : 1));
  return `{${entries.map(([key, inner]) => `${JSON.stringify(key)}:${canonical(inner)}`).join(",")}}`;
}

// crypto.subtle only exists on https pages and on localhost.
export async function configHash(profile: EditorProfile): Promise<string> {
  const text = canonical({ monaco: MONACO_VERSION, options: EDITOR_PROFILES[profile] });
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
