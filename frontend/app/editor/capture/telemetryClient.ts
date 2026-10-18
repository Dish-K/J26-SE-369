import { ACTIVE_PROFILE, configHash } from "./editorOptions";

const WS_URL = process.env.NEXT_PUBLIC_TELEMETRY_WS_URL ?? "ws://127.0.0.1:8000/ws/telemetry";
const FLUSH_MS = 250;
const MAX_BATCH_EVENTS = 25;

export type CapturedEvent = {
  kind: number;
  perf_now: number;
  [column: string]: number | string | boolean | null;
};

type Reply = { ack?: { load_seq: number; up_to_batch_seq: number } };

type SentBatch = { batchSeq: number; text: string };

export class TelemetryClient {
  private readonly sessionId: string;
  private readonly taskId: string;
  private readonly socket: WebSocket;
  private readonly unconfirmed: SentBatch[] = [];
  private readonly timer: ReturnType<typeof setInterval>;
  private buffer: CapturedEvent[] = [];
  private nextBatchSeq = 0;
  private stopped = false;

  constructor(sessionId: string, taskId: string) {
    this.sessionId = sessionId;
    this.taskId = taskId;
    this.socket = new WebSocket(WS_URL);
    // Batches made before the socket opened, are waiting in unconfirmed.
    this.socket.addEventListener("open", () => {
      for (const batch of this.unconfirmed) {
        this.socket.send(batch.text);
      }
    });
    this.socket.addEventListener("message", (message) => {
      this.confirm(JSON.parse(message.data) as Reply);
    });
    this.socket.addEventListener("close", (close) => {
      this.stop(`connection closed (${close.code} ${close.reason})`);
    });
    this.timer = setInterval(() => this.flush(), FLUSH_MS);
  }

  record(event: CapturedEvent): void {
    if (this.stopped) {
      return;
    }
    this.buffer.push({ ...event, task_id: this.taskId });
    if (this.buffer.length >= MAX_BATCH_EVENTS) {
      this.flush();
    }
  }

  private flush(): void {
    if (this.buffer.length === 0) {
      return;
    }
    const events = this.buffer.map((event, position) => ({ ...event, seq_in_batch: position }));
    this.buffer = [];

    const batchSeq = this.nextBatchSeq++;
    const text = JSON.stringify({
      session_id: this.sessionId,
      load_seq: 0,
      batch_seq: batchSeq,
      events,
    });
    this.unconfirmed.push({ batchSeq, text });
    if (this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(text);
    }
  }

  private confirm(reply: Reply): void {
    const upTo = reply.ack?.up_to_batch_seq;
    if (upTo === undefined) {
      return;
    }
    while (this.unconfirmed.length > 0 && this.unconfirmed[0].batchSeq <= upTo) {
      this.unconfirmed.shift();
    }
  }

  private stop(reason: string): void {
    if (this.stopped) {
      return;
    }
    this.stopped = true;
    clearInterval(this.timer);
    console.error(
      `Telemetry capture stopped: ${reason}. ${this.unconfirmed.length} batch(es) not confirmed.`,
    );
  }
}

function browserContext() {
  // "|| null" turns an unknown size of 0 into "not reported", which the server accepts.
  return {
    user_agent: navigator.userAgent,
    platform: navigator.platform,
    screen_w: screen.width || null,
    screen_h: screen.height || null,
    viewport_w: window.innerWidth || null,
    viewport_h: window.innerHeight || null,
    device_pixel_ratio: window.devicePixelRatio || null,
    locale: navigator.language,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  };
}

async function startSession(): Promise<TelemetryClient> {
  const response = await fetch("/api/telemetry/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      editor_profile: ACTIVE_PROFILE,
      editor_config_hash: await configHash(ACTIVE_PROFILE),
      ...browserContext(),
    }),
  });
  if (!response.ok) {
    throw new Error(`Could not start a telemetry session: ${response.status}`);
  }
  const ids = (await response.json()) as { session_id: string; task_id: string };
  return new TelemetryClient(ids.session_id, ids.task_id);
}

let client: Promise<TelemetryClient> | undefined;

// One client per page load, however many times the editor mounts.
export function telemetryClient(): Promise<TelemetryClient> {
  client ??= startSession();
  return client;
}
