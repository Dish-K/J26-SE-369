import type { IKeyboardEvent, editor } from "monaco-editor";

import { Kind, Modifier } from "./codes";
import { keyClass } from "./keyClass";
import type { TelemetryClient } from "./telemetryClient";

function modifiers(key: KeyboardEvent): number {
  return (
    (key.ctrlKey ? Modifier.ctrl : 0) |
    (key.altKey ? Modifier.alt : 0) |
    (key.shiftKey ? Modifier.shift : 0) |
    (key.metaKey ? Modifier.meta : 0)
  );
}

// Returns the function that removes both handlers again
export function attachCapture(target: editor.IStandaloneCodeEditor, client: TelemetryClient): () => void {
  const handler = (kind: number) => (event: IKeyboardEvent) => {
    // First line: any work done before this read becomes error in every timing feature
    const perfNow = performance.now();
    // Never read key.key: it is the typed character.
    const key = event.browserEvent;
    client.record({
      kind,
      perf_now: perfNow,
      key_code: key.code || null,
      key_class: keyClass(key.code),
      modifiers: modifiers(key),
      is_repeat: key.repeat,
      is_trusted: key.isTrusted,
    });
  };

  const down = target.onKeyDown(handler(Kind.keydown));
  const up = target.onKeyUp(handler(Kind.keyup));

  return () => {
    down.dispose();
    up.dispose();
  };
}
