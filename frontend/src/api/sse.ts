/**
 * Shared SSE adapter: id-aware parsing, Last-Event-ID resume, exponential backoff reconnect.
 * Use for Chat, Scholar, Ingest, Deep Research long-task streams.
 */

export type SSEResumableOptions = {
  /** Build URL for the next request; receive lastEventId for query/header resume. */
  getUrl: (lastEventId: string) => string;
  /** Optional extra headers (e.g. Authorization). Last-Event-ID is added by the adapter when resuming. */
  getHeaders?: (lastEventId: string) => Record<string, string>;
  /** Event types that end the stream; on receipt the generator returns. */
  terminalEvents: string[];
  signal?: AbortSignal;
  maxRetries?: number;
  baseMs?: number;
  maxMs?: number;
};

export type SSEEvent = { event: string; data: unknown; id: string };

const DEFAULT_MAX_RETRIES = 5;
const DEFAULT_BASE_MS = 1000;
const DEFAULT_MAX_MS = 30000;

function parseSSEBlock(block: string): { event: string; id: string; data: unknown } | null {
  let eventType = 'message';
  let eventId = '';
  const dataLines: string[] = [];
  let hasField = false;
  for (const line of block.split('\n')) {
    if (line.startsWith(':')) continue;
    if (line.startsWith('event: ')) {
      eventType = line.slice(7).trim();
      hasField = true;
    } else if (line.startsWith('id: ')) {
      eventId = line.slice(4).trim();
      hasField = true;
    } else if (line.startsWith('data: ')) {
      dataLines.push(line.slice(6));
      hasField = true;
    } else if (line.trim()) hasField = true;
  }
  if (!hasField) return null;
  const dataStr = dataLines.join('\n');
  let data: unknown = {};
  if (dataStr) {
    try {
      data = JSON.parse(dataStr);
    } catch {
      data = { raw: dataStr };
    }
  }
  return { event: eventType, id: eventId, data };
}

/**
 * Stream SSE with resume: parses id, yields events, reconnects with Last-Event-ID on drop.
 * Stops when a terminal event is received or maxRetries is exceeded.
 */
export async function* streamSSEResumable(
  options: SSEResumableOptions,
): AsyncGenerator<SSEEvent> {
  const {
    getUrl,
    getHeaders,
    terminalEvents,
    signal,
    maxRetries = DEFAULT_MAX_RETRIES,
    baseMs = DEFAULT_BASE_MS,
    maxMs = DEFAULT_MAX_MS,
  } = options;
  let lastEventId = '';
  let attempt = 0;
  let receivedTerminal = false;

  while (!receivedTerminal) {
    try {
      const url = getUrl(lastEventId);
      const headers: Record<string, string> = {
        ...(getHeaders ? getHeaders(lastEventId) : {}),
        ...(lastEventId ? { 'Last-Event-ID': lastEventId } : {}),
      };

      const response = await fetch(url, { method: 'GET', headers, signal });
      if (!response.ok) {
        const errBody = await response.text().catch(() => '');
        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errBody}`);
      }
      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';
      let gotAnyData = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        gotAnyData = true;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split(/\n\n/);
        buffer = blocks.pop() || '';
        for (const block of blocks) {
          const parsed = parseSSEBlock(block);
          if (!parsed) continue;
          if (parsed.id) lastEventId = parsed.id;
          yield { event: parsed.event, data: parsed.data, id: parsed.id };
          if (terminalEvents.includes(parsed.event)) {
            receivedTerminal = true;
            return;
          }
        }
      }
      if (buffer.trim()) {
        const parsed = parseSSEBlock(buffer);
        if (parsed) {
          if (parsed.id) lastEventId = parsed.id;
          yield { event: parsed.event, data: parsed.data, id: parsed.id };
          if (terminalEvents.includes(parsed.event)) {
            receivedTerminal = true;
            return;
          }
        }
      }

      if (!receivedTerminal) {
        attempt++;
        if (attempt > maxRetries) {
          throw new Error(`SSE connection closed unexpectedly after ${maxRetries} retries`);
        }
        const delay = gotAnyData
          ? Math.min(baseMs * Math.pow(2, attempt) + Math.random() * baseMs, maxMs)
          : Math.min(500, baseMs);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
    } catch (err) {
      if (signal?.aborted) throw err;
      attempt++;
      if (attempt > maxRetries || receivedTerminal) throw err;
      const delay = Math.min(baseMs * Math.pow(2, attempt) + Math.random() * baseMs, maxMs);
      console.warn(`[streamSSEResumable] connection error (attempt=${attempt}), retrying in ${Math.round(delay)}ms:`, err);
      await new Promise((r) => setTimeout(r, delay));
    }
  }
}

export const SCHOLAR_TERMINAL_EVENTS = ['done', 'error', 'cancelled', 'timeout'] as const;
export const INGEST_TERMINAL_EVENTS = ['done', 'error', 'cancelled'] as const;
// DR stream ends with job_status (terminal state reached) or error (job not found).
// Heartbeat events use the distinct name 'heartbeat', so job_status is always terminal.
export const DR_TERMINAL_EVENTS = ['job_status', 'error'] as const;
