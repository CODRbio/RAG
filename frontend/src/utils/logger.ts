/**
 * 前端统一日志工具
 *
 * 分作用域（scope）管理日志，生产环境自动静音 debug/info，
 * error 级别可选上报后端（logs/frontend/）。
 *
 * 用法：
 *   import { logger } from '@/utils/logger'
 *   logger.api.info('POST /chat/submit', { url, bodyLen })
 *   logger.ui.debug('user submitted query', { queryLen: query.length })
 *   logger.error('Canvas load failed', err)
 *
 * 作用域：
 *   api    - fetch / API 调用层
 *   ui     - 组件交互事件
 *   error  - 跨域错误（始终输出）
 */

type Level = 'debug' | 'info' | 'warn' | 'error'
type Scope = 'api' | 'ui' | 'error'

const LEVEL_RANK: Record<Level, number> = { debug: 0, info: 1, warn: 2, error: 3 }

// 开发环境打印 debug+，生产环境只打印 warn+
const ACTIVE_LEVEL: Level = import.meta.env.DEV ? 'debug' : 'warn'

// 后端日志上报地址（仅生产环境 error 级别启用）
const REPORT_URL = '/api/logs/frontend'
const REPORT_ENABLED = !import.meta.env.DEV

function shouldLog(level: Level): boolean {
  return LEVEL_RANK[level] >= LEVEL_RANK[ACTIVE_LEVEL]
}

function emit(scope: Scope, level: Level, msg: string, data?: unknown): void {
  if (!shouldLog(level)) return

  const prefix = `[rag:${scope}]`
  const consoleFn = console[level] ?? console.log

  if (data !== undefined) {
    consoleFn(prefix, msg, data)
  } else {
    consoleFn(prefix, msg)
  }

  // error 级别在生产环境上报到后端
  if (level === 'error' && REPORT_ENABLED) {
    reportToBackend(scope, msg, data).catch(() => {/* 上报失败静默处理 */})
  }
}

async function reportToBackend(scope: Scope, msg: string, data?: unknown): Promise<void> {
  const payload = {
    scope,
    level: 'error' as Level,
    msg,
    data: data instanceof Error
      ? { name: data.name, message: data.message, stack: data.stack }
      : data,
    url: window.location.href,
    ts: new Date().toISOString(),
  }
  try {
    await fetch(REPORT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      keepalive: true,
    })
  } catch {
    // 网络错误：静默忽略，不能因日志上报影响主流程
  }
}

// ── 作用域 logger 工厂 ────────────────────────────────────────────────────────

function makeScope(scope: Scope) {
  return {
    debug: (msg: string, data?: unknown) => emit(scope, 'debug', msg, data),
    info:  (msg: string, data?: unknown) => emit(scope, 'info',  msg, data),
    warn:  (msg: string, data?: unknown) => emit(scope, 'warn',  msg, data),
    error: (msg: string, data?: unknown) => emit(scope, 'error', msg, data),
  }
}

// ── 公共导出 ──────────────────────────────────────────────────────────────────

export const logger = {
  /** HTTP fetch / API 调用层 */
  api: makeScope('api'),
  /** 组件交互事件 */
  ui:  makeScope('ui'),
  /** 直接记录跨域 error（始终输出） */
  error: (msg: string, data?: unknown) => emit('error', 'error', msg, data),
}
