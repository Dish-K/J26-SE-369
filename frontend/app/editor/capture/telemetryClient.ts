import { ACTIVE_PROFILE, configHash } from "./editorOptions";

const WS_URL = process.env.NEXT_PUBLIC_TELEMETRY_WS_URL ?? "ws://127.0.0.1:8000/ws/telemetry";
const FLUSH_MS = 250;
const MAX_BATCH_EVENTS = 25;
// Must match HANDSHAKE_SAMPLES in backend/app/telemetry/clock.py, the server refuses any other count.
const HANDSHAKE_SAMPLES = 8;
// Must match SESSION_OPEN_ELSEWHERE in backend/app/telemetry/router.py.
const SESSION_OPEN_ELSEWHERE = 4409;
const RETRY_MS = 2_000;
const RETRY_FOR_MS = 45_000;
const SESSION_KEY = "codetrace.telemetry.session";
const CHANNEL_NAME = "codetrace.telemetry";

export type CaptureStatus = "connecting" | "ready" | "reconnecting" | "blocked" | "stopped";

export type CapturedEvent = {
  kind: number;
  perf_now: number;
  [column: string]: number | string | boolean | null;
};

type SessionIds = { session_id: string; task_id: string };

type Sample = { t0: number; t1: number; t2: number; t3: number };

type Reply = {
  pong?: { t0: number; t1: number; t2: number };
  ready?: { load_seq: number };
  ack?: { load_seq: number; up_to_batch_seq: number };
};

type SentBatch = { loadSeq: number; batchSeq: number; text: string };

type TabMessage = { type: "who-has" | "mine"; session_id: string };

export class TelemetryClient {
  private readonly ids: SessionIds;
  private readonly channel: BroadcastChannel | null;
  private readonly listeners = new Set<(status: CaptureStatus) => void>();
  private socket: WebSocket | null = null;
  private status: CaptureStatus = "connecting";
  private samples: Sample[] = [];
  private buffer: CapturedEvent[] = [];
  private unconfirmed: SentBatch[] = [];
  private loadSeq: number | null = null;
  private nextBatchSeq = 0;
  private openElsewhere = false;
  private retryUntil = 0;

  constructor(ids: SessionIds) {
    this.ids = ids;
    this.channel = typeof BroadcastChannel === "undefined" ? null : new BroadcastChannel(CHANNEL_NAME);
    this.channel?.addEventListener("message", (message) => {
      this.onTabMessage(message.data as TabMessage);
    });
    // Ask any other tab of this browser whether one of them already has this session.
    this.channel?.postMessage({ type: "who-has", session_id: ids.session_id } satisfies TabMessage);
    setInterval(() => this.flush(), FLUSH_MS);
    this.connect(true);
  }

  onStatus(listener: (status: CaptureStatus) => void): () => void {
    this.listeners.add(listener);
    listener(this.status);
    return () => {
      this.listeners.delete(listener);
    };
  }

  retry(): void {
    if (this.status !== "blocked") {
      return;
    }
    this.openElsewhere = false;
    this.connect(false);
  }

  record(event: CapturedEvent): void {
    // Only a ready connection has a load_seq to send with the event.
    if (this.status !== "ready") {
      return;
    }
    this.buffer.push({ ...event, task_id: this.ids.task_id });
    if (this.buffer.length >= MAX_BATCH_EVENTS) {
      this.flush();
    }
  }

  private connect(autoRetry: boolean): void {
    if (this.status !== "reconnecting") {
      this.setStatus("connecting");
    }
    this.samples = [];
    const socket = new WebSocket(WS_URL);
    this.socket = socket;

    socket.addEventListener("open", () => {
      socket.send(JSON.stringify({ hello: { session_id: this.ids.session_id } }));
      this.ping();
    });
    socket.addEventListener("message", (message) => {
      // First line: anything done before this read adds error to the sample.
      const t3 = performance.now();
      this.onReply(JSON.parse(message.data) as Reply, t3);
    });
    socket.addEventListener("close", (close) => {
      if (socket === this.socket) {
        this.onClose(close, autoRetry);
      }
    });
  }

  private ping(): void {
    this.socket?.send(JSON.stringify({ ping: { t0: performance.now() } }));
  }

  private onReply(reply: Reply, t3: number): void {
    if (reply.pong) {
      this.samples.push({ ...reply.pong, t3 });
      if (this.samples.length < HANDSHAKE_SAMPLES) {
        this.ping();
      } else {
        this.socket?.send(JSON.stringify({ clock: { samples: this.samples } }));
      }
    } else if (reply.ready) {
      this.loadSeq = reply.ready.load_seq;
      this.nextBatchSeq = 0;
      this.retryUntil = 0;
      this.setStatus("ready");
    } else if (reply.ack) {
      const { load_seq, up_to_batch_seq } = reply.ack;
      this.unconfirmed = this.unconfirmed.filter(
        (batch) => batch.loadSeq !== load_seq || batch.batchSeq > up_to_batch_seq,
      );
    }
  }

  private onClose(close: CloseEvent, autoRetry: boolean): void {
    this.loadSeq = null;

    if (close.code !== SESSION_OPEN_ELSEWHERE) {
      this.setStatus("stopped");
      console.error(
        `Telemetry capture stopped: connection closed (${close.code} ${close.reason}). ` +
          `${this.unconfirmed.length} batch(es) not confirmed.`,
      );
      return;
    }

    if (this.openElsewhere || !autoRetry) {
      this.setStatus("blocked");
      return;
    }

    // No other tab answered, so this is most likely our own page's old connection after a reload.
    if (this.retryUntil === 0) {
      this.retryUntil = Date.now() + RETRY_FOR_MS;
    }
    if (Date.now() >= this.retryUntil) {
      this.setStatus("blocked");
      return;
    }
    this.setStatus("reconnecting");
    setTimeout(() => {
      if (!this.openElsewhere) {
        this.connect(true);
      }
    }, RETRY_MS);
  }

  private onTabMessage(message: TabMessage): void {
    if (message.session_id !== this.ids.session_id) {
      return;
    }
    if (message.type === "who-has" && this.status === "ready") {
      this.channel?.postMessage({ type: "mine", session_id: this.ids.session_id } satisfies TabMessage);
    } else if (message.type === "mine" && this.status !== "ready") {
      this.openElsewhere = true;
      this.setStatus("blocked");
    }
  }

  private flush(): void {
    if (this.status !== "ready" || this.loadSeq === null || this.buffer.length === 0) {
      return;
    }
    const events = this.buffer.map((event, position) => ({ ...event, seq_in_batch: position }));
    this.buffer = [];

    const batchSeq = this.nextBatchSeq++;
    const text = JSON.stringify({
      session_id: this.ids.session_id,
      load_seq: this.loadSeq,
      batch_seq: batchSeq,
      events,
    });
    this.unconfirmed.push({ loadSeq: this.loadSeq, batchSeq, text });
    this.socket?.send(text);
  }

  private setStatus(status: CaptureStatus): void {
    this.status = status;
    for (const listener of this.listeners) {
      listener(status);
    }
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

async function startSession(): Promise<SessionIds> {
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
  return (await response.json()) as SessionIds;
}

function savedIds(): SessionIds | null {
  try {
    const saved = sessionStorage.getItem(SESSION_KEY);
    return saved ? (JSON.parse(saved) as SessionIds) : null;
  } catch {
    return null;
  }
}

async function sessionIds(): Promise<SessionIds> {
  const saved = savedIds();
  if (saved) {
    return saved;
  }
  const ids = await startSession();
  try {
    // Kept across a reload in this tab, and gone when the tab closes.
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(ids));
  } catch {
    // Storage is blocked in some private modes; a reload then starts a new session.
  }
  return ids;
}

let client: Promise<TelemetryClient> | undefined;

// One client per page load, however many times the editor mounts (Decision 38).
export function telemetryClient(): Promise<TelemetryClient> {
  client ??= sessionIds().then((ids) => new TelemetryClient(ids));
  return client;
}
