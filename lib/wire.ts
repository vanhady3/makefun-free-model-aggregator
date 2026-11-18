// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does; an
//     Anthropic-wire provider gets the request rewritten to the Messages wire and its answer translated
//     back (see openAIToAnthropicRequest, anthropicToOpenAIResponse, openAIEventsFromAnthropic);
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini upstream wire is built below (openAIToGeminiRequest, geminiToOpenAIResponse,
// openAIEventsFromGemini): an OpenAI-shaped request is pivoted through the Gemini generateContent
// shape, and a Gemini answer (JSON or SSE) is pivoted back through the OpenAI shape, reusing the
// translators above rather than adding an Anthropic <-> Gemini one.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Anthropic's stop_reason as OpenAI's finish_reason (the reverse of stopReasonOf). */
function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function anthropicPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: the mirror of
 * anthropicToOpenAI. System messages become the system string, user text+image_url parts become
 * Anthropic content blocks, assistant tool_calls become tool_use blocks, tool messages become
 * tool_result blocks, tools and tool_choice are rewritten, and the output cap, temperature, top_p, stop
 * sequences and streaming pass through. No response_format is added: the Anthropic wire cannot carry it
 * (the env's echo expects json_mode=false for Anthropic).
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    // A user message: text and image parts become Anthropic content blocks.
    if (typeof raw.content === 'string') {
      if (raw.content !== '') messages.push({ role: 'user', content: [{ type: 'text', text: raw.content }] });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const parts: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      const block = anthropicPartOf(part);
      if (block !== null) parts.push(block);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice.type === 'required') out.tool_choice = { type: 'any' };
    else if (choice.type === 'none') out.tool_choice = { type: 'none' };
    else if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
      out.tool_choice = { type: 'tool', name: choice.function.name };
    }
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object: the mirror of openAIToAnthropic. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : typeof body.content === 'string' ? [{ type: 'text', text: body.content }] : [];
  const texts: string[] = [];
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' && block.id !== '' ? block.id : 'toolu_unknown',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const promptTokens = numberOrZero(usage.input_tokens);
  const completionTokens = numberOrZero(usage.output_tokens);
  return {
    id: `chatcmpl_${typeof body.id === 'string' && body.id !== '' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: texts.length > 0 ? texts.join('\n') : null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: finishReasonOf(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
  };
}

/**
 * An Anthropic SSE stream as OpenAI Chat Completions SSE chunks: the mirror of anthropicEventsFromOpenAI.
 * message_start opens with a role chunk; text deltas become content deltas; tool_use starts and
 * input_json_deltas become tool_calls deltas; message_delta holds the stop reason and output usage until
 * message_stop (or the stream's end), when the final chunk carries the finish_reason and the usage, then
 * [DONE]. `model` is the model the client asked for, named back to it.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The finish reason a message_delta named, held until message_stop (or the stream's end). */
  let pendingFinish: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  const toolIndexes = new Map<number, number>();

  const chunk = (delta: Json, finish: string | null, usage: Json | null): string => {
    const out: Json = {
      id: 'chatcmpl_stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    };
    if (usage !== null) out.usage = usage;
    return sseEvent(null, out);
  };
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [chunk({ role: 'assistant', content: '' }, null, null)];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const usage: Json = { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens };
    return [chunk({}, pendingFinish ?? 'stop', usage), 'data: [DONE]\n\n'];
  };

  /** One SSE frame's lines as OpenAI chunks, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    let event: string | null = null;
    let payload = '';
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('event:')) event = trimmed.slice(6).trim();
      else if (trimmed.startsWith('data:')) payload = trimmed.slice(5).trim();
    }
    if (payload === '' || payload === '[DONE]') {
      if (event === 'message_stop' || payload === '[DONE]') out.push(...startEvents(), ...finishEvents());
      return out;
    }
    let data: unknown;
    try {
      data = JSON.parse(payload);
    } catch {
      return out;
    }
    if (!isRecord(data)) return out;
    switch (event) {
      case 'message_start': {
        const message = isRecord(data.message) ? data.message : {};
        const usage = isRecord(message.usage) ? message.usage : {};
        inputTokens = numberOrZero(usage.input_tokens);
        out.push(...startEvents());
        break;
      }
      case 'content_block_start': {
        out.push(...startEvents());
        const index = typeof data.index === 'number' ? data.index : toolIndexes.size;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use') {
          toolIndexes.set(index, index);
          out.push(
            chunk(
              {
                tool_calls: [
                  {
                    index,
                    id: typeof block.id === 'string' && block.id !== '' ? block.id : `toolu_${index}`,
                    type: 'function',
                    function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' },
                  },
                ],
              },
              null,
              null,
            ),
          );
        }
        break;
      }
      case 'content_block_delta': {
        out.push(...startEvents());
        const index = typeof data.index === 'number' ? data.index : 0;
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
          out.push(chunk({ content: delta.text }, null, null));
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
          out.push(chunk({ tool_calls: [{ index, function: { arguments: delta.partial_json } }] }, null, null));
        }
        break;
      }
      case 'content_block_stop': {
        break;
      }
      case 'message_delta': {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (typeof delta.stop_reason === 'string') pendingFinish = finishReasonOf(delta.stop_reason);
        const usage = isRecord(data.usage) ? data.usage : {};
        outputTokens = numberOrZero(usage.output_tokens);
        break;
      }
      case 'message_stop': {
        out.push(...startEvents(), ...finishEvents());
        break;
      }
      default: {
        break;
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents()) yield event;
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/* ------------------------------------------------------------------ */
/* The Gemini upstream wire, pivoted through the OpenAI shape.        */
/*                                                                     */
/* A Gemini-wire provider is reached at POST                          */
/*   <base>/v1beta/models/<model>:generateContent (JSON) or            */
/*   <base>/v1beta/models/<model>:streamGenerateContent?alt=sse       */
/* and answers a generateContent object (candidates + usageMetadata)  */
/* or a streamGenerateContent SSE stream of the same objects. Neither  */
/* shape is translated directly to or from Anthropic: the OpenAI      */
/* request and answer are the pivot, so the Anthropic client reuses    */
/* anthropicToOpenAI / openAIToAnthropic around these three functions. */
/* ------------------------------------------------------------------ */

/** A Gemini finishReason as an OpenAI finish_reason; a function call wins over STOP. */
function geminiFinishReasonOf(reason: unknown, hasToolCall: boolean): string {
  if (hasToolCall) return 'tool_calls';
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request. The model is not carried
 * in the body: it names the URL path instead, so `model` is accepted for the shared translator
 * shape and not written out. System messages become systemInstruction; user text and image_url parts
 * become contents parts (a data URL becomes inlineData with the `data:` prefix stripped, a plain
 * http(s) URL becomes fileData); assistant tool_calls become functionCall parts; tool messages
 * become functionResponse parts (the name is the one the matching assistant tool_call carried);
 * tools become one functionDeclarations entry; tool_choice "none" omits the tools, anything else
 * passes them through (the env reads only the declarations, so no toolConfig is sent: a "required"
 * choice is carried by the declarations alone and the model decides); max_tokens, temperature,
 * top_p, stop and a JSON response_format become generationConfig fields. Only fields the OpenAI
 * body carried are written: nothing is invented.
 */
export function openAIToGeminiRequest(body: Json, model: string): Json {
  void model;
  const systemParts: Json[] = [];
  const contents: Json[] = [];
  /** The function name each assistant tool_call id carried, for the tool message that answers it. */
  const toolNamesById = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push({ text });
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        if (typeof call.id === 'string' && call.id !== '' && name !== '') toolNamesById.set(call.id, name);
        parts.push({ functionCall: { name, args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const name =
        typeof raw.tool_call_id === 'string' ? (toolNamesById.get(raw.tool_call_id) ?? raw.tool_call_id) : '';
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      contents.push({ role: 'user', parts: [{ functionResponse: { name, response: { result: text } } }] });
      continue;
    }
    // A user message: text and image parts become Gemini parts.
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string' && part.text !== '') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) {
            parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
          } else if (/^https?:\/\//.test(url)) {
            parts.push({ fileData: { fileUri: url } });
          }
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const choice = body.tool_choice;
  const choiceIsNone = choice === 'none' || (isRecord(choice) && choice.type === 'none');
  if (!choiceIsNone && Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens =
    typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (typeof body.stop === 'string' && body.stop !== '') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop)) {
    const stops = body.stop.filter((s): s is string => typeof s === 'string');
    if (stops.length > 0) config.stopSequences = stops;
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * A Gemini generateContent JSON answer as an OpenAI Chat Completions object. The first candidate's
 * text parts become the message content and its functionCall parts become tool_calls (id
 * `call_<name>_<index>`); a function call makes the finish_reason "tool_calls", otherwise STOP
 * becomes "stop", MAX_TOKENS "length" and SAFETY "content_filter"; usageMetadata becomes usage;
 * the id names the answer's modelVersion where it has one, else the model given. No candidates is
 * still a valid empty completion with finish_reason "stop".
 */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const parts = isRecord(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
  const texts: string[] = [];
  const toolCalls: Json[] = [];
  let callIndex = 0;
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    else if (isRecord(part.functionCall)) {
      const name = typeof part.functionCall.name === 'string' ? part.functionCall.name : '';
      toolCalls.push({
        id: `call_${name}_${callIndex}`,
        type: 'function',
        function: { name, arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) },
      });
      callIndex += 1;
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const promptTokens = numberOrZero(usage.promptTokenCount);
  const completionTokens = numberOrZero(usage.candidatesTokenCount);
  const totalTokens =
    typeof usage.totalTokenCount === 'number' && Number.isFinite(usage.totalTokenCount)
      ? usage.totalTokenCount
      : promptTokens + completionTokens;
  const version = typeof body.modelVersion === 'string' && body.modelVersion !== '' ? body.modelVersion : model;
  return {
    id: `chatcmpl_${version}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: texts.length > 0 ? texts.join('') : null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: geminiFinishReasonOf(candidate.finishReason, toolCalls.length > 0),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: totalTokens },
  };
}

/**
 * A Gemini streamGenerateContent SSE stream as OpenAI Chat Completions SSE chunks: the mirror of
 * openAIEventsFromAnthropic. The first frame opens with a role chunk; text parts become content
 * deltas; functionCall parts become tool_calls deltas (id, type and name on first sight of that
 * call, arguments JSON as the delta); a frame carrying a finishReason, or the stream's end, emits
 * the final chunk with the mapped finish_reason and the last usageMetadata, then [DONE] (the Gemini
 * stream itself sends no [DONE]). Partial frames split across reads are buffered until complete.
 * `model` is the model the client asked for, named back to it.
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The finish reason a frame's finishReason named, held until the final chunk is emitted. */
  let pendingFinish: string | null = null;
  let sawToolCall = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let totalTokens = 0;
  /** Function-call indexes already announced with an id and a name. */
  const seenCalls = new Set<number>();

  const chunk = (delta: Json, finish: string | null, usage: Json | null): string => {
    const out: Json = {
      id: 'chatcmpl_stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    };
    if (usage !== null) out.usage = usage;
    return sseEvent(null, out);
  };
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [chunk({ role: 'assistant', content: '' }, null, null)];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const usage: Json = { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: totalTokens };
    return [chunk({}, pendingFinish ?? (sawToolCall ? 'tool_calls' : 'stop'), usage), 'data: [DONE]\n\n'];
  };

  /** One SSE frame's lines as OpenAI chunks, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents());
        continue;
      }
      let data: unknown;
      try {
        data = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      if (isRecord(data.usageMetadata)) {
        inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
        outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
        totalTokens =
          typeof data.usageMetadata.totalTokenCount === 'number' && Number.isFinite(data.usageMetadata.totalTokenCount)
            ? data.usageMetadata.totalTokenCount
            : inputTokens + outputTokens;
      }
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      const candidate = isRecord(candidates[0]) ? candidates[0] : {};
      const parts = isRecord(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
      let callIndex = 0;
      for (const part of parts) {
        if (!isRecord(part)) continue;
        if (typeof part.text === 'string' && part.text !== '') {
          out.push(...startEvents());
          out.push(chunk({ content: part.text }, null, null));
        } else if (isRecord(part.functionCall)) {
          out.push(...startEvents());
          sawToolCall = true;
          const name = typeof part.functionCall.name === 'string' ? part.functionCall.name : '';
          const args = JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {});
          if (!seenCalls.has(callIndex)) {
            seenCalls.add(callIndex);
            out.push(
              chunk(
                { tool_calls: [{ index: callIndex, id: `call_${name}_${callIndex}`, type: 'function', function: { name, arguments: args } }] },
                null,
                null,
              ),
            );
          } else {
            out.push(chunk({ tool_calls: [{ index: callIndex, function: { arguments: args } }] }, null, null));
          }
          callIndex += 1;
        }
      }
      if (typeof candidate.finishReason === 'string') {
        pendingFinish = geminiFinishReasonOf(candidate.finishReason, sawToolCall);
        out.push(...startEvents(), ...finishEvents());
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents()) yield event;
  }
}
