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

type Submission = "open" | "confirming" | "submitting" | "submitted" | "failed";

const SUBMISSION_NOTICES: Partial<Record<Submission, string>> = {
  submitting: "Submitting...",
  submitted: "Submitted. Waiting for the next question.",
  failed: "Your answer was not submitted.",
};

export default function CodeEditor() {
  const [client, setClient] = useState<TelemetryClient | null>(null);
  const [status, setStatus] = useState<CaptureStatus>("connecting");
  const [instance, setInstance] = useState<editor.IStandaloneCodeEditor | null>(null);
  const [submission, setSubmission] = useState<Submission>("open");

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

  const submit = async () => {
    const model = instance?.getModel();
    if (!client || !model) {
      return;
    }
    setSubmission("submitting");
    const stored = await client.submit(model.getValue(), model.getVersionId());
    setSubmission(stored ? "submitted" : "failed");
  };

  const handleMount: OnMount = (mounted) => setInstance(mounted);
  const notice = NOTICES[status];
  const submissionNotice = SUBMISSION_NOTICES[submission];
  const editable = status === "ready" && (submission === "open" || submission === "confirming" || submission === "failed");

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
      {submissionNotice && (
        <p className="editor-notice" role="status">
          {submissionNotice}
        </p>
      )}
      <Editor
        height="60vh"
        defaultLanguage="python"
        // Read-only unless capture is ready and the answer is still open, so no keystroke goes unrecorded.
        options={{ ...EDITOR_PROFILES[ACTIVE_PROFILE], readOnly: !editable }}
        onMount={handleMount}
      />
      {submission === "confirming" ? (
        <div className="editor-actions">
          <p>Submit this answer? You cannot change it afterwards.</p>
          <button className="button button-accent" type="button" onClick={() => void submit()}>
            Yes, Submit
          </button>
          <button className="button button-plain" type="button" onClick={() => setSubmission("open")}>
            Cancel
          </button>
        </div>
      ) : (
        (submission === "open" || submission === "failed") && (
          <div className="editor-actions">
            <button
              className="button button-accent"
              type="button"
              disabled={!editable}
              onClick={() => setSubmission("confirming")}
            >
              Submit Answer
            </button>
          </div>
        )
      )}
    </>
  );
}
