// Typed calls to the sanitise HTTP API (docs/phase3_plan.md section 6.4).
// Types come from schema.d.ts, generated from the API's OpenAPI document: `npm run types`.

import type { components } from "./schema";

type S = components["schemas"];
export type RagAnswer = S["RagAnswer"];
export type CompareResponse = S["CompareResponse"];
export type AttackDefinition = S["AttackDefinition"];
export type AttackKind = AttackDefinition["kind"];
export type AttackRun = S["AttackRun"];
export type AttackRound = S["AttackRound"];
export type RoundLeakage = S["RoundLeakage"];
export type SessionTrace = S["SessionTrace"];
export type DocumentSummary = S["DocumentSummary"];
export type DocumentView = S["DocumentView"];
export type RoutingDecision = S["RoutingDecision"];

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

async function call<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_URL + path, {
      method,
      headers: body === undefined ? undefined : { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "UNREACHABLE", `Cannot reach the API at ${API_URL}`);
  }
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    const err = data?.error;
    throw new ApiError(
      res.status,
      err?.code ?? "HTTP_" + res.status,
      err?.detail ?? (typeof data?.detail === "string" ? data.detail : res.statusText),
    );
  }
  return res.json() as Promise<T>;
}

const documentCache = new Map<string, Promise<DocumentView>>();

export const api = {
  health: () => call<S["HealthResponse"]>("GET", "/v1/health"),
  createSession: () => call<{ session_id: string }>("POST", "/v1/sessions", {}),
  session: (id: string) => call<SessionTrace>("GET", `/v1/sessions/${id}`),
  compare: (query: string, session_id: string) =>
    call<CompareResponse>("POST", "/v1/compare", { query, session_id }),
  attacks: () => call<AttackDefinition[]>("GET", "/v1/attacks"),
  documents: (limit = 500, offset = 0) =>
    call<DocumentSummary[]>("GET", `/v1/documents?limit=${limit}&offset=${offset}`),
  document(id: string): Promise<DocumentView> {
    let view = documentCache.get(id);
    if (!view) {
      view = call<DocumentView>("GET", `/v1/documents/${id}`);
      view.catch(() => documentCache.delete(id));
      documentCache.set(id, view);
    }
    return view;
  },
};

type StreamHandlers = {
  round: (r: AttackRound) => void;
  done: (run: AttackRun) => void;
  fail: (message: string) => void;
};

/** Runs an attack over Server-Sent Events; each round arrives as soon as it is answered.
 * Returns a function that stops listening. */
export function streamAttack(id: string, on: StreamHandlers): () => void {
  const source = new EventSource(`${API_URL}/v1/attacks/${encodeURIComponent(id)}/stream`);
  const data = (e: Event) => JSON.parse((e as MessageEvent).data);
  source.addEventListener("round", (e) => on.round(data(e)));
  source.addEventListener("done", (e) => {
    source.close();
    on.done(data(e));
  });
  source.addEventListener("error", (e) => {
    source.close(); // without this the browser reconnects and reruns the attack
    const body = (e as MessageEvent).data;
    on.fail(body ? (JSON.parse(body).error?.detail ?? "The attack failed") : `Lost the API at ${API_URL}`);
  });
  return () => source.close();
}
