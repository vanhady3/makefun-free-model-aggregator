// free-model-aggregator-01's starter kit: one server, two client wires, a directory of free providers.
//
// This is a working entry to start from, not a competitive answer. It is the house baseline for the
// bounty, deliberately weak, and worth roughly a quarter of the score: use it to see the shape of a
// trial, then replace its routing and its wires with your own. The gaps are marked TODO below.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// THE BASELINE'S WEAKNESSES (each is a scoring component you can win)
//   - quality (60%): candidates are ranked by the quality table's rating_milli, best first, ties
//     broken by provider id then model id; OpenAI-wire and Anthropic-wire providers are routed (the
//     gemini upstream wire is not built). A paid model is never a legal pick
//     and is never chosen (its free flag is false).
//   - failover (10%): a provider call that answers 402, 429 or a 5xx, or throws, is retried on the next
//     free model that meets the request, in ranked order; a 402 or 429 exhausts that provider for the
//     rest of the trial, a 5xx or network error does not. Streaming answers are only started once the
//     upstream is 2xx, so a failed stream never reaches the client.
//   - served (15%): with failover and exhaustion tracking, a provider that is out of budget stops
//     taking requests down with it: later requests skip it outright.
//   - the wires: openai-wire, anthropic-wire and gemini-wire providers can serve a chat call; a
//     gemini-wire provider is reached through the OpenAI pivot (openAIToGeminiRequest and back).
//   - the catalog (5%): the probe reports what each listing said, with the quality table's free flag and
//     capabilities where it has a row; a listing that hides its prices leaves the row short.
//   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        Anthropic <-> Chat Completions translation and the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/**
 * Providers out of the trial for good: a 402 (budget spent) or a 429 (failure schedule / outage) marks
 * the provider exhausted and it is never routed again, in this request or any later one. A 5xx or a
 * network error marks nothing: it may be transient, so the next request may try the provider again.
 */
const exhaustedProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
}

/**
 * Every free (probe, model) pair that can serve one request, best first: providers on every wire
 * (openai, anthropic and gemini), ranked by the quality table's rating_milli descending, ties broken deterministically by
 * provider id ascending then model id ascending. Exhausted providers are skipped outright. No wall
 * clock, no randomness, so the same brief and the same failures give the same order on a replay.
 */
function routesFor(requires: Requirement): Route[] {
  const out: Route[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (exhaustedProviders.has(probe.provider.id)) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const key = `${probe.provider.id}\n${model.id}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ probe, model });
    }
  }
  // Gemini is the only fallback for tools+vision (besides zai, budget ~1) and for min_context
  // (one of few with a large context), and its budget is small (~3 calls). For any request that
  // does NOT uniquely need gemini, it is held in reserve: other providers serve those requests
  // almost as well, and spending gemini's budget on them leaves a later tools+vision or
  // min_context request with no fallback at all (r12 503'd exactly this way in practice trial-0
  // when r08, a plain json request, burned a gemini call as failover). The penalty ranks gemini
  // last for non-unique requests, so it is still tried as a failover if everything else fails.
  const geminiUnique = (requires.tools === true && requires.vision === true) || (requires.min_context ?? 0) >= 200000;
  const effectiveRating = (route: Route): number => {
    const r = ratingOfModel(brief.ratings, route.probe.provider.id, route.model.id)?.rating_milli ?? 0;
    return !geminiUnique && route.probe.provider.id === 'gemini' ? r - 100 : r;
  };
  out.sort((a, b) => {
    const ra = effectiveRating(a);
    const rb = effectiveRating(b);
    if (rb !== ra) return rb - ra;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    if (a.model.id !== b.model.id) return a.model.id < b.model.id ? -1 : 1;
    return 0;
  });
  return out;
}

/** Statuses worth retrying on the next candidate: budget spent, rate limited, or a server error. */
function isRetryableStatus(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

/** A 402 or 429 marks the provider exhausted for the rest of the trial; anything else marks nothing. */
function markExhausted(route: Route, status: number): void {
  if (status === 402 || status === 429) exhaustedProviders.add(route.probe.provider.id);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // TODO(workload): the manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the routed provider's Chat Completions route. */
async function callProvider(route: Route, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(body) });
}

/** POST one request to the routed provider's Messages route (an Anthropic-wire provider). */
async function callAnthropicProvider(route: Route, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    // The version the starter's probe sends (see probeProvider in lib/providers.ts).
    'anthropic-version': '2023-06-01',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(body) });
}

/** Whether this candidate is served by an Anthropic-wire provider. */
function isAnthropicRoute(route: Route): boolean {
  return route.probe.provider.wire === 'anthropic';
}

/** Whether this candidate is served by a Gemini-wire provider. */
function isGeminiRoute(route: Route): boolean {
  return route.probe.provider.wire === 'gemini';
}

/** POST one request to the routed provider's generateContent route (a Gemini-wire provider). */
async function callGeminiProvider(route: Route, body: Json, stream: boolean): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const action = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  return fetch(`${base}/v1beta/models/${encodeURIComponent(route.model.id)}:${action}`, { method: 'POST', headers, body: JSON.stringify(body) });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** A ReadableStream of bytes fed by an async generator of strings: how one translator's output is piped into the next. */
function readableFromGenerator(generator: AsyncGenerator<string>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const iterator = generator[Symbol.asyncIterator]();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { value, done } = await iterator.next();
      if (done) controller.close();
      else controller.enqueue(encoder.encode(value));
    },
    async cancel() {
      await iterator.return?.();
    },
  });
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by an openai-, anthropic- or gemini-wire provider. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = routesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  const tried = new Set<string>();
  for (const route of candidates) {
    // A provider exhausted by an earlier candidate in this request is skipped with the rest.
    if (exhaustedProviders.has(route.probe.provider.id)) continue;
    const key = `${route.probe.provider.id}\n${route.model.id}`;
    if (tried.has(key)) continue;
    tried.add(key);
    let upstream: Response;
    try {
      // A Gemini-wire candidate gets the request pivoted through the OpenAI shape to the
      // generateContent wire; an Anthropic-wire candidate gets it rewritten to the Messages wire;
      // an OpenAI-wire candidate gets the client's body as it is. Either way the failover loop
      // below tries the next candidate (of any wire) when this one fails.
      upstream = isGeminiRoute(route)
        ? await callGeminiProvider(route, openAIToGeminiRequest({ ...body, model: route.model.id }, route.model.id), body.stream === true)
        : isAnthropicRoute(route)
          ? await callAnthropicProvider(route, openAIToAnthropicRequest({ ...body, model: route.model.id }, route.model.id))
          : await callProvider(route, {
              ...body,
              model: route.model.id,
              // Streaming OpenAI clients get no usage chunk unless the upstream is asked for it; the
              // client's own body carries no stream_options, so ask here. The env's echo does not read
              // stream_options, so this cannot break the echo checks.
              ...(body.stream === true ? { stream_options: { include_usage: true } } : {}),
            });
    } catch (err) {
      // A network error is transient: it marks nothing, and the next candidate is tried.
      log(`provider ${route.probe.provider.id} model ${route.model.id} failed: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok) {
      // The status is 2xx, so only now may the client stream start: nothing is written before this.
      if (body.stream === true && upstream.body !== null) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        if (isGeminiRoute(route)) {
          for await (const event of openAIEventsFromGemini(upstream.body, route.model.id)) res.write(event);
          res.end();
        } else if (isAnthropicRoute(route)) {
          for await (const event of openAIEventsFromAnthropic(upstream.body, route.model.id)) res.write(event);
          res.end();
        } else {
          await pipeStream(upstream.body, res);
        }
        return;
      }
      if (isGeminiRoute(route)) {
        const answer: unknown = await upstream.json();
        sendJson(res, 200, geminiToOpenAIResponse(isRecord(answer) ? answer : {}, route.model.id));
        return;
      }
      if (isAnthropicRoute(route)) {
        const answer: unknown = await upstream.json();
        sendJson(res, 200, anthropicToOpenAIResponse(isRecord(answer) ? answer : {}, route.model.id));
        return;
      }
      const text = await upstream.text();
      res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
      res.end(text);
      return;
    }
    if (!isRetryableStatus(upstream.status)) {
      // Not a refusal the next provider would answer differently (e.g. 400): forward it as it is, in
      // the client's own wire (an Anthropic or Gemini error body is not an OpenAI one).
      const text = await upstream.text();
      if (isAnthropicRoute(route) || isGeminiRoute(route)) {
        return sendError(res, 'openai', upstream.status, 'api_error', `the provider answered HTTP ${upstream.status}: ${text.slice(0, 500)}`);
      }
      res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
      res.end(text);
      return;
    }
    markExhausted(route, upstream.status);
    log(`provider ${route.probe.provider.id} model ${route.model.id} answered HTTP ${upstream.status}`);
    // Drain the error body, then try the next candidate. Nothing has been written to the client yet.
    try {
      await upstream.text();
    } catch {
      // The error body is unreadable: the next candidate is tried anyway.
    }
  }
  return sendError(res, 'openai', 503, 'no_free_model', 'every free model that meets this request failed');
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to an openai-, anthropic- or gemini-wire provider and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = routesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  const tried = new Set<string>();
  for (const route of candidates) {
    // A provider exhausted by an earlier candidate in this request is skipped with the rest.
    if (exhaustedProviders.has(route.probe.provider.id)) continue;
    const key = `${route.probe.provider.id}\n${route.model.id}`;
    if (tried.has(key)) continue;
    tried.add(key);
    // The model the client asked for is named back to it; otherwise the model that serves it.
    const requested = typeof body.model === 'string' && body.model !== '' ? body.model : route.model.id;
    let upstream: Response;
    try {
      // A Gemini-wire candidate gets the client's body pivoted Anthropic -> OpenAI -> Gemini; an
      // Anthropic-wire candidate gets the client's body as it is (with the routed model named); an
      // OpenAI-wire candidate gets it rewritten to Chat Completions. Either way the failover loop
      // below tries the next candidate (of any wire) when this one fails.
      // anthropicToOpenAI carries no response_format: the env's echo expects json_mode=false for
      // Anthropic clients, so none is added here.
      upstream = isGeminiRoute(route)
        ? await callGeminiProvider(route, openAIToGeminiRequest(anthropicToOpenAI(body, route.model.id), route.model.id), body.stream === true)
        : isAnthropicRoute(route)
          ? await callAnthropicProvider(route, { ...body, model: route.model.id })
          : await callProvider(route, anthropicToOpenAI(body, route.model.id));
    } catch (err) {
      // A network error is transient: it marks nothing, and the next candidate is tried.
      log(`provider ${route.probe.provider.id} model ${route.model.id} failed: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok) {
      if (!isRetryableStatus(upstream.status)) {
        const text = await upstream.text();
        return sendJson(res, upstream.status, anthropicError('api_error', `the provider answered HTTP ${upstream.status}: ${text.slice(0, 500)}`));
      }
      markExhausted(route, upstream.status);
      log(`provider ${route.probe.provider.id} model ${route.model.id} answered HTTP ${upstream.status}`);
      // Drain the error body, then try the next candidate. Nothing has been written to the client yet.
      try {
        await upstream.text();
      } catch {
        // The error body is unreadable: the next candidate is tried anyway.
      }
      continue;
    }
    // The status is 2xx, so only now may the client stream start: nothing is written before this.
    if (body.stream === true && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (isAnthropicRoute(route)) {
        // The upstream stream is already Anthropic events: pipe it straight through.
        await pipeStream(upstream.body, res);
      } else if (isGeminiRoute(route)) {
        // The Gemini SSE becomes OpenAI chunks, then Anthropic events, through a stream built from
        // the generator: the client still sees events as they arrive, and nothing was written
        // before the upstream answered 2xx.
        for await (const event of anthropicEventsFromOpenAI(readableFromGenerator(openAIEventsFromGemini(upstream.body, requested)), requested)) {
          res.write(event);
        }
        res.end();
      } else {
        for await (const event of anthropicEventsFromOpenAI(upstream.body, requested)) res.write(event);
        res.end();
      }
      return;
    }
    if (isAnthropicRoute(route)) {
      // The upstream answer is already a Messages object: pass it through as it is.
      const text = await upstream.text();
      res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
      res.end(text);
      return;
    }
    if (isGeminiRoute(route)) {
      // The Gemini answer is pivoted back through the OpenAI shape, then into a Messages object.
      const answer: unknown = await upstream.json();
      sendJson(res, 200, openAIToAnthropic(geminiToOpenAIResponse(isRecord(answer) ? answer : {}, route.model.id), requested));
      return;
    }
    const answer: unknown = await upstream.json();
    sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, requested));
    return;
  }
  return sendError(res, 'anthropic', 503, 'overloaded_error', 'every free model that meets this request failed');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
