"use client";

import Editor, { loader, type OnMount } from "@monaco-editor/react";
import type { editor } from "monaco-editor";
import { useEffect, useState } from "react";

import { attachCapture } from "./capture/capture";
import { ACTIVE_PROFILE, EDITOR_PROFILES, MONACO_VERSION } from "./capture/editorOptions";
import { type TelemetryClient, telemetryClient } from "./capture/telemetryClient";

// Load the pinned Monaco, so the version in the fingerprint is the one that runs.
loader.config({ paths: { vs: `https://cdn.jsdelivr.net/npm/monaco-editor@${MONACO_VERSION}/min/vs` } });

export default function CodeEditor() {
  const [client, setClient] = useState<TelemetryClient | null>(null);
  const [instance, setInstance] = useState<editor.IStandaloneCodeEditor | null>(null);

  useEffect(() => {
    telemetryClient()
      .then(setClient)
      .catch((error: unknown) => console.error(error));
  }, []);

  useEffect(() => {
    if (!instance || !client) {
      return;
    }
    return attachCapture(instance, client);
  }, [instance, client]);

  const handleMount: OnMount = (mounted) => setInstance(mounted);

  return (
    <Editor
      height="60vh"
      defaultLanguage="python"
      // Read-only until the session exists, so no keystroke goes unrecorded.
      options={{ ...EDITOR_PROFILES[ACTIVE_PROFILE], readOnly: client === null }}
      onMount={handleMount}
    />
  );
}
