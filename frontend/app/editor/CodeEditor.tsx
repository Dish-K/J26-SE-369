"use client";

import Editor, { loader, type OnMount } from "@monaco-editor/react";
import type { editor } from "monaco-editor";
import { useEffect, useState } from "react";

import { attachCapture } from "./capture/capture";
import { ACTIVE_PROFILE, EDITOR_PROFILES, MONACO_VERSION } from "./capture/editorOptions";
import { type CaptureStatus, type TelemetryClient, telemetryClient } from "./capture/telemetryClient";

// Load the pinned Monaco, so the version in the fingerprint is the one that runs.
loader.config({ paths: { vs: `https://cdn.jsdelivr.net/npm/monaco-editor@${MONACO_VERSION}/min/vs` } });

const NOTICES: Partial<Record<CaptureStatus, string>> = {
  connecting: "Connecting...",
  reconnecting: "Reconnecting...",
  blocked: "This interview is already open in another tab. Return to that tab, or close it to continue here.",
  stopped: "The connection was lost. Reload the page to continue.",
};

export default function CodeEditor() {
  const [client, setClient] = useState<TelemetryClient | null>(null);
  const [status, setStatus] = useState<CaptureStatus>("connecting");
  const [instance, setInstance] = useState<editor.IStandaloneCodeEditor | null>(null);

  useEffect(() => {
    telemetryClient()
      .then(setClient)
      .catch((error: unknown) => console.error(error));
  }, []);

  useEffect(() => client?.onStatus(setStatus), [client]);

  useEffect(() => {
    if (!instance || !client) {
      return;
    }
    return attachCapture(instance, client);
  }, [instance, client]);

  const handleMount: OnMount = (mounted) => setInstance(mounted);
  const notice = NOTICES[status];

  return (
    <>
      {notice && (
        <p className="editor-notice" role="status">
          {notice}
          {status === "blocked" && (
            <button className="button button-accent" type="button" onClick={() => client?.retry()}>
              Try Again
            </button>
          )}
        </p>
      )}
      <Editor
        height="60vh"
        defaultLanguage="python"
        // Read-only unless capture is ready, so no keystroke goes unrecorded.
        options={{ ...EDITOR_PROFILES[ACTIVE_PROFILE], readOnly: status !== "ready" }}
        onMount={handleMount}
      />
    </>
  );
}
