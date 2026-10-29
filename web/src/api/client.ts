// Typed calls to the sanitise HTTP API (docs/phase3_plan.md section 6.4).
// Types come from schema.d.ts, generated from the API's OpenAPI document: `npm run types`.

import type { components } from "./schema";

type S = components["schemas"];
export type RagAnswer = S["RagAnswer"];
export type CompareResponse = S["CompareResponse"];
export type AttackDefinition = S["AttackDefinition"];
export type AttackKind = AttackDefinition["kind"];
export type DocumentSummary = S["DocumentSummary"];
export type DocumentView = S["DocumentView"];
export type RoutingDecision = S["RoutingDecision"];
export type SanitisedRecord = S["SanitisedRecord"];
export type MonitorVerdict = S["MonitorVerdict"];
export type PIICategory = S["PIICategory"];
export type Pipeline = RagAnswer["pipeline"];
export type DetectorResult = S["DetectorResult"];
export type SensitiveSpan = S["SensitiveSpan"];
export type FactTuple = S["FactTuple"];
export type ExtractorOutput = S["ExtractorOutput"];
export type RewriterOutput = S["RewriterOutput"];

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
      // the header skips ngrok's warning page for free accounts; other hosts ignore it
      headers: {
        "ngrok-skip-browser-warning": "1",
        ...(body === undefined ? {} : { "content-type": "application/json" }),
      },
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
  // the three agents one at a time, so each one's output can be shown as it finishes
  detect: (text: string) => call<DetectorResult>("POST", "/v1/detect", { text }),
  // spans: Agent 1's result for this text, so Agent 1 does not run again with a different one
  extract: (doc_id: string, text: string, spans?: SensitiveSpan[]) =>
    call<ExtractorOutput>("POST", "/v1/extract", { doc_id, text, spans }),
  rewrite: (doc_id: string, spans: SensitiveSpan[], facts: FactTuple[]) =>
    call<RewriterOutput>("POST", "/v1/rewrite", { doc_id, spans, facts }),
  createSession: () => call<{ session_id: string }>("POST", "/v1/sessions", {}),
  query: (query: string, pipeline: Pipeline, session_id: string) =>
    call<RagAnswer>("POST", "/v1/query", { query, pipeline, session_id }),
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
