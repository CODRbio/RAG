import axios, { type AxiosInstance, type InternalAxiosRequestConfig, type AxiosRequestConfig } from 'axios';

// 基础 URL：开发时通过 Vite proxy 转发，生产时直接访问后端
const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

/** 是否为连接级错误（后端重启/reload 时会出现），可安全重试 */
function isRetryableNetworkError(e: unknown): boolean {
  const err = e as { code?: string; response?: unknown };
  return (
    !err.response &&
    (err.code === 'ECONNREFUSED' || err.code === 'ETIMEDOUT' || err.code === 'ERR_NETWORK')
  );
}

/**
 * GET 请求，在连接失败时自动重试（用于轮询接口，避免后端 --reload 时短暂不可用导致报错）。
 */
export async function getWithRetry<T>(
  url: string,
  config?: AxiosRequestConfig,
  maxRetries = 2,
  delayMs = 2000
): Promise<{ data: T }> {
  let lastErr: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await client.get<T>(url, config);
      return { data: res.data };
    } catch (e) {
      lastErr = e;
      if (attempt < maxRetries && isRetryableNetworkError(e)) {
        await new Promise((r) => setTimeout(r, delayMs));
        continue;
      }
      throw e;
    }
  }
  throw lastErr;
}

const client: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器：自动附加 token
client.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// 响应拦截器：处理 401
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      // 可以在这里触发全局登出逻辑
      window.dispatchEvent(new CustomEvent('auth:logout'));
    }
    return Promise.reject(error);
  }
);

export default client;

// SSE 流式请求辅助
export async function* streamChat(
  url: string,
  body: object,
  signal?: AbortSignal
): AsyncGenerator<{ event: string; data: unknown }> {
  const token = localStorage.getItem('token');
  const response = await fetch(`${BASE_URL}${url}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    let currentEvent = '';
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        const dataStr = line.slice(6);
        try {
          const data = JSON.parse(dataStr);
          yield { event: currentEvent, data };
        } catch {
          yield { event: currentEvent, data: dataStr };
        }
      }
    }
  }
}
