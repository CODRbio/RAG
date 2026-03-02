import axios, { type AxiosInstance, type InternalAxiosRequestConfig, type AxiosRequestConfig } from 'axios';

// 基础 URL：开发时通过 Vite proxy 转发，生产时直接访问后端
const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

/**
 * 判断是否为可安全重试的网络错误：
 *  - 连接级错误（ECONNREFUSED / ETIMEDOUT / ERR_NETWORK）：后端重启/reload 时短暂不可用
 *  - Vite proxy 错误（500 + X-Proxy-Error: backend-unavailable）：worker 意外退出导致 socket hang up
 *  - 标准网关错误（502 / 503）：反向代理/负载均衡上游不可用
 */
function isRetryableNetworkError(e: unknown): boolean {
  const err = e as {
    code?: string;
    response?: { status?: number; headers?: Record<string, string> };
  };

  // 连接级错误（无 HTTP 响应）
  if (
    !err.response &&
    (err.code === 'ECONNREFUSED' || err.code === 'ETIMEDOUT' || err.code === 'ERR_NETWORK')
  ) {
    return true;
  }

  const status = err.response?.status;

  // Vite proxy 将 socket hang up / ECONNRESET 转换为 500，并附带自定义 header
  if (status === 500 && err.response?.headers?.['x-proxy-error'] === 'backend-unavailable') {
    return true;
  }

  // 标准网关/服务不可用错误
  return status === 502 || status === 503;
}

/**
 * GET 请求，在后端暂时不可用时自动重试（用于轮询接口）。
 * 重试间隔加入少量随机抖动，避免多个请求同时打到刚恢复的后端。
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
        // 加入 ±20% 抖动，避免多请求同时重试
        const jitter = delayMs * 0.2 * (Math.random() * 2 - 1);
        await new Promise((r) => setTimeout(r, delayMs + jitter));
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
