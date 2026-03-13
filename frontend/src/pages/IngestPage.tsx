import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Database,
  HardDrive,
  Plus,
  FileText,
  Trash2,
  FolderOpen,
  Loader2,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  RotateCw,
  ChevronLeft,
  ChevronRight,
  List,
  Link2,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useConfigStore, useUIStore, useToastStore } from '../stores';
import { Modal } from '../components/ui/Modal';
import {
  listCollections,
  createCollection,
  deleteCollection,
  getCollectionScope,
  updateCollectionScope,
  refreshCollectionScope,
  ensureCollectionLibraryBinding,
  listPapers,
  deletePaper,
  cancelIngestJob,
  getIngestJob,
  listIngestJobs,
  streamIngestJobEvents,
  listLLMProviders,
  type CollectionInfo,
  type IngestProgressEvent,
  type PaperInfo,
  type LLMProviderInfo,
  type EnrichmentOptions,
} from '../api/ingest';
import {
  listLibraries as listScholarLibraries,
  getLibraryPapers as getScholarLibraryPapers,
  ingestScholarLibrary,
  importLibraryPdfs,
  createLibrary,
  deleteLibrary,
  type LibraryImportPdfSummary,
  type ScholarLibrary,
} from '../api/scholar';
import { login as apiLogin } from '../api/auth';
import { useAuthStore } from '../stores/useAuthStore';
import { logger } from '../utils/logger';

// ---- Types ----

/** One knowledge base: merged by name from Scholar Library + Vector Collection. */
export interface MergedBase {
  name: string;
  library: (ScholarLibrary & { downloaded_count: number }) | null;
  collection: CollectionInfo | null;
}

interface IngestScholarLibraryOption extends ScholarLibrary {
  downloaded_count: number;
}

// ---- Component ----

export function IngestPage() {
  const { t } = useTranslation();
  const { dbAddress, setDbStatus, currentCollection, setCurrentCollection, setCollections, setCollectionInfos } =
    useConfigStore();
  const { showCreateCollectionModal, setShowCreateCollectionModal } = useUIStore();
  const addToast = useToastStore((s) => s.addToast);
  const user = useAuthStore((s) => s.user);

  // Connection
  const [connectError, setConnectError] = useState('');

  // Collections and libraries from backend
  const [backendCollections, setBackendCollections] = useState<CollectionInfo[]>([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);
  const [scholarLibraries, setScholarLibraries] = useState<IngestScholarLibraryOption[]>([]);
  const [scholarLibrariesLoading, setScholarLibrariesLoading] = useState(false);

  /** Merged knowledge bases by name (library + collection). */
  const mergedBases = useMemo((): MergedBase[] => {
    const names = new Set<string>([
      ...scholarLibraries.filter((l) => l.id >= 0).map((l) => l.name),
      ...backendCollections.map((c) => c.name),
    ]);
    return Array.from(names).map((name) => ({
      name,
      library: scholarLibraries.find((l) => l.name === name && l.id >= 0) ?? null,
      collection: backendCollections.find((c) => c.name === name) ?? null,
    }));
  }, [scholarLibraries, backendCollections]);

  /** Selected knowledge base by name (drives sync, upload, papers list). */
  const [selectedBaseName, setSelectedBaseName] = useState<string | null>(null);

  const selectedBase = useMemo(
    () => (selectedBaseName ? mergedBases.find((b) => b.name === selectedBaseName) ?? null : null),
    [mergedBases, selectedBaseName],
  );
  const selectedLibraryId = selectedBase?.library?.id ?? null;
  const selectedLibrary = selectedBase?.library ?? null;

  // Papers in selected collection (for "已入库文件" list)
  const [papers, setPapers] = useState<PaperInfo[]>([]);
  const [papersLoading, setPapersLoading] = useState(false);
  const [papersPageSize, setPapersPageSize] = useState<10 | 20>(10);
  const [papersPage, setPapersPage] = useState(1);

  // Sync / ingest
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [globalProgress, setGlobalProgress] = useState('');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [enrichLogs, setEnrichLogs] = useState<Record<string, string[]>>({});
  const [autoDownloadMissingScholarPdfs, setAutoDownloadMissingScholarPdfs] = useState(false);

  // Upload PDFs to library (import to selected base's library)
  const [boundLibraryImportFiles, setBoundLibraryImportFiles] = useState<File[]>([]);
  const [boundLibraryImporting, setBoundLibraryImporting] = useState(false);
  const [boundLibraryImportSummary, setBoundLibraryImportSummary] = useState<LibraryImportPdfSummary | null>(null);

  // Enrichment options (table parsing defaults to qwen / qwen3.5 without thinking)
  const [enrichment, setEnrichment] = useState<EnrichmentOptions>({
    enrich_tables: false,
    enrich_figures: false,
    llm_text_provider: 'qwen',
    llm_text_model: null,
    llm_text_concurrency: 1,
    llm_vision_provider: null,
    llm_vision_model: null,
    llm_vision_concurrency: 1,
  });
  const [llmProviders, setLlmProviders] = useState<LLMProviderInfo[]>([]);

  const loadLlmProviderOptions = useCallback(async () => {
    try {
      const res = await listLLMProviders();
      setLlmProviders(res.providers || []);
    } catch (err) {
      logger.api.error('Failed to load LLM providers', err);
    }
  }, []);

  useEffect(() => {
    loadLlmProviderOptions();
  }, [loadLlmProviderOptions]);

  const tableProvider = useMemo(
    () => llmProviders.find((p) => p.id === enrichment.llm_text_provider),
    [llmProviders, enrichment.llm_text_provider]
  );
  const figureProvider = useMemo(
    () => llmProviders.find((p) => p.id === enrichment.llm_vision_provider),
    [llmProviders, enrichment.llm_vision_provider]
  );

  const concurrencyOptions = [1, 2, 4, 8, 16];

  // Scope modal
  const [scopeRefreshingCollection, setScopeRefreshingCollection] = useState<string | null>(null);
  const [scopeModalOpen, setScopeModalOpen] = useState(false);
  const [scopeModalCollection, setScopeModalCollection] = useState<string | null>(null);
  const [scopeSummaryText, setScopeSummaryText] = useState('');
  const [scopeUpdatedAt, setScopeUpdatedAt] = useState<string | null>(null);
  const [scopeModalLoading, setScopeModalLoading] = useState(false);
  const [scopeModalSaving, setScopeModalSaving] = useState(false);

  // New knowledge base modal
  const [newCollectionName, setNewCollectionName] = useState('');

  // Delete collection confirmation (password)
  const [deleteConfirmModal, setDeleteConfirmModal] = useState<{ baseName: string; libraryId: number } | null>(null);
  const [deletePassword, setDeletePassword] = useState('');
  const [deleteVerifying, setDeleteVerifying] = useState(false);

  // Delete paper confirmation (password)
  const [deletePaperModal, setDeletePaperModal] = useState<{ paperId: string; filename: string; collection: string } | null>(null);
  const [deletePaperPassword, setDeletePaperPassword] = useState('');
  const [deletePaperVerifying, setDeletePaperVerifying] = useState(false);

  // Sync selectedBaseName to config store for sidebar/chat
  useEffect(() => {
    if (selectedBaseName && selectedBaseName !== currentCollection) {
      setCurrentCollection(selectedBaseName);
    }
  }, [selectedBaseName, currentCollection, setCurrentCollection]);

  const boundLibraryFolderInputRef = useRef<HTMLInputElement>(null);
  const boundLibraryFileInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const loadCollections = useCallback(async () => {
    setCollectionsLoading(true);
    setConnectError('');
    try {
      const cols = await listCollections();
      setBackendCollections(cols);
      setDbStatus('connected');
      setCollections(cols.map((c) => c.name));
      setCollectionInfos(cols);
    } catch (err) {
      logger.api.error('Failed to load collections', err);
      setConnectError(err instanceof Error ? err.message : '无法连接后端服务');
    } finally {
      setCollectionsLoading(false);
    }
  }, [setCollectionInfos, setCollections, setDbStatus]);

  const loadScholarLibrariesForIngest = useCallback(async () => {
    setScholarLibrariesLoading(true);
    try {
      const libs = (await listScholarLibraries()).filter((lib) => lib.id >= 0);
      const enriched = await Promise.all(
        libs.map(async (lib) => {
          try {
            const papers = await getScholarLibraryPapers(lib.id);
            const downloaded_count = papers.filter((p) => Boolean(p.is_downloaded || p.downloaded_at)).length;
            return { ...lib, downloaded_count } as IngestScholarLibraryOption;
          } catch {
            return { ...lib, downloaded_count: 0 } as IngestScholarLibraryOption;
          }
        }),
      );
      setScholarLibraries(enriched);
      setSelectedBaseName((prev) => {
        if (prev && enriched.some((l) => l.name === prev)) return prev;
        return enriched[0]?.name ?? null;
      });
    } catch (err) {
      logger.api.warn('Failed to load scholar libraries for ingest', err);
      setScholarLibraries([]);
      setSelectedBaseName(null);
    } finally {
      setScholarLibrariesLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCollections();
    loadScholarLibrariesForIngest();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let cancelled = false;
    const reconnectCtrl = new AbortController();

    const reconnectRunningJob = async () => {
      try {
        const savedJobId = localStorage.getItem('ingest_active_job_id');
        let targetJobId: string | null = savedJobId;
        if (!targetJobId) {
          const [running, pending] = await Promise.all([
            listIngestJobs(5, 'running'),
            listIngestJobs(5, 'pending'),
          ]);
          targetJobId = running[0]?.job_id ?? pending[0]?.job_id ?? null;
        }
        if (!targetJobId || cancelled) return;

        // Verify the job is still active before subscribing to its event stream.
        // If the backend already finished/errored the job (e.g. worker restarted),
        // we clear stale local state instead of getting stuck in isProcessing=true.
        try {
          const jobInfo = await getIngestJob(targetJobId);
          const terminalStatuses = ['done', 'error', 'cancelled'];
          if (terminalStatuses.includes(jobInfo.status)) {
            localStorage.removeItem('ingest_active_job_id');
            if (jobInfo.status === 'error') {
              addToast(`上一入库任务已终止: ${jobInfo.error_message || jobInfo.status}`, 'info');
            }
            return;
          }
        } catch {
          // If the job no longer exists (404 etc.), clear stale state and bail.
          localStorage.removeItem('ingest_active_job_id');
          return;
        }

        setActiveJobId(targetJobId);
        setIsProcessing(true);
        setGlobalProgress('已恢复后台任务进度...');
        abortRef.current = reconnectCtrl;
        addToast(`已连接后台任务 ${targetJobId.slice(0, 8)}`, 'info');
        for await (const evt of streamIngestJobEvents(targetJobId, reconnectCtrl.signal, 0)) {
          if (cancelled) break;
          applyIngestEvent(evt);
        }
      } catch (e) {
        if ((e as Error).name !== 'AbortError') {
          localStorage.removeItem('ingest_active_job_id');
        }
      } finally {
        if (!cancelled) {
          setIsProcessing(false);
          if (!activeJobId) {
            abortRef.current = null;
          }
        }
      }
    };

    reconnectRunningJob();
    return () => {
      cancelled = true;
      reconnectCtrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadPapers = useCallback(async (): Promise<PaperInfo[]> => {
    if (!selectedBaseName) { setPapers([]); setPapersPage(1); return []; }
    setPapersLoading(true);
    try {
      const list = await listPapers(selectedBaseName);
      setPapers(list);
      setPapersPage(1);
      return list;
    } catch {
      setPapers([]);
      return [];
    } finally {
      setPapersLoading(false);
    }
  }, [selectedBaseName]);

  useEffect(() => {
    loadPapers();
  }, [loadPapers]);

  const handleDeletePaper = (paperId: string, filename: string) => {
    if (!selectedBaseName) return;
    setDeletePaperModal({ paperId, filename, collection: selectedBaseName });
    setDeletePaperPassword('');
  };

  const handleConfirmDeletePaperWithPassword = async () => {
    if (!deletePaperModal || !deletePaperPassword.trim() || !user) return;
    setDeletePaperVerifying(true);
    try {
      await apiLogin({ user_id: user.user_id, password: deletePaperPassword });
      const { paperId, filename, collection } = deletePaperModal;
      addToast(`正在删除 ${filename || paperId}...`, 'info');
      const res = await deletePaper(collection, paperId);
      addToast(`已删除 ${filename || paperId} (${res.deleted_chunks} chunks)`, 'success');
      setDeletePaperModal(null);
      setDeletePaperPassword('');
      loadPapers();
      loadCollections();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`删除失败: ${msg}`, 'error');
    } finally {
      setDeletePaperVerifying(false);
    }
  };

  const handlePickBoundLibraryFolder = () => boundLibraryFolderInputRef.current?.click();
  const handlePickBoundLibraryFiles = () => boundLibraryFileInputRef.current?.click();

  const handleBoundLibraryFolderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(e.target.files ?? []).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
    setBoundLibraryImportFiles(picked);
    setBoundLibraryImportSummary(null);
    if (picked.length === 0) {
      addToast('未找到可导入的 PDF 文件', 'info');
    } else {
      addToast(`已选择 ${picked.length} 个 PDF，点击“导入到文献库”开始`, 'info');
    }
    e.target.value = '';
  };

  const handleBoundLibraryFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(e.target.files ?? []).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
    setBoundLibraryImportFiles(picked);
    setBoundLibraryImportSummary(null);
    if (picked.length === 0) {
      addToast('未找到可导入的 PDF 文件', 'info');
    } else {
      addToast(`已选择 ${picked.length} 个 PDF，点击「导入到文献库」开始`, 'info');
    }
    e.target.value = '';
  };

  const handleImportBoundLibraryPdfs = async () => {
    if (selectedLibraryId == null || selectedLibraryId < 0) {
      addToast('请先选择一个知识库（或新建知识库）', 'info');
      return;
    }
    if (boundLibraryImportFiles.length === 0) {
      addToast('请先选择文件或文件夹', 'info');
      return;
    }
    if (boundLibraryImporting) return;
    setBoundLibraryImporting(true);
    try {
      const summary = await importLibraryPdfs(selectedLibraryId, boundLibraryImportFiles);
      setBoundLibraryImportSummary(summary);
      setBoundLibraryImportFiles([]);
      await loadScholarLibrariesForIngest();
      await loadCollections();
      addToast(
        `导入完成：新增 ${summary.imported}，关联已有 ${summary.linked_existing}，跳过重复 ${summary.skipped_duplicates}`,
        'success',
      );
      if (summary.errors.length > 0) {
        addToast(`部分文件处理失败（${summary.errors.length}）`, 'error');
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`导入失败: ${msg}`, 'error');
    } finally {
      setBoundLibraryImporting(false);
    }
  };

  // ---- Upload & Process ----
  const applyIngestEvent = (evt: IngestProgressEvent) => {
    const d = evt.data as Record<string, unknown>;
    switch (evt.event) {
      case 'job_created': {
        const jobId = d.job_id as string | undefined;
        if (jobId) {
          setActiveJobId(jobId);
          localStorage.setItem('ingest_active_job_id', jobId);
        }
        break;
      }
      case 'start':
        setGlobalProgress(`开始处理 ${d.total} 个文件...`);
        setEnrichLogs({});
        break;
      case 'enrich_progress': {
        const file = d.file as string;
        const kind = (d.kind as string) || 'table';
        const index = d.index as number;
        const total = d.total as number;
        const status = d.status as string;
        const message = d.message as string | undefined;
        const kindLabel = kind === 'figure' ? '图片' : '表格';
        const statusLabel =
          status === 'success'
            ? '完成'
            : status === 'skip'
              ? '跳过'
              : status === 'start'
                ? '进行中'
                : '失败';
        const line = message
          ? `${kindLabel} ${index}/${total} ${statusLabel}: ${message}`
          : `${kindLabel} ${index}/${total} ${statusLabel}`;
        setEnrichLogs((prev) => {
          const list = [...(prev[file] || []), line].slice(-30);
          return { ...prev, [file]: list };
        });
        break;
      }
      case 'progress':
      case 'heartbeat': {
        const message = d.message as string;
        if (message) setGlobalProgress(message);
        break;
      }
      case 'file_done':
      case 'file_error': {
        const message = d.message as string;
        if (message) setGlobalProgress(message);
        if (evt.event === 'file_done') {
          loadPapers();
          loadCollections();
          loadScholarLibrariesForIngest();
        }
        break;
      }
      case 'error':
        addToast(`处理错误: ${d.message}`, 'error');
        break;
      case 'done':
        if (d.cancelled) {
          setGlobalProgress('任务已取消');
          addToast('处理任务已取消', 'info');
        } else {
          setGlobalProgress(`完成: ${d.total_upserted} 条记录入库 (${d.total_chunks} chunks)`);
          addToast(`入库完成: ${d.total_upserted} 条记录`, 'success');
        }
        loadCollections();
        loadPapers();
        setActiveJobId(null);
        localStorage.removeItem('ingest_active_job_id');
        setIsCancelling(false);
        break;
      case 'cancelled':
        setGlobalProgress('任务已取消');
        setActiveJobId(null);
        localStorage.removeItem('ingest_active_job_id');
        setIsCancelling(false);
        break;
    }
  };

  const handleStartScholarLibraryIngest = async () => {
    if (!selectedBaseName) {
      addToast('请先选择一个知识库', 'info');
      return;
    }
    if (selectedLibraryId == null || selectedLibraryId < 0) {
      addToast('该知识库尚无文献库，请先新建文献库或从文献检索导入', 'info');
      return;
    }
    if (isProcessing) {
      // Before blocking the user, verify the job is still genuinely active on the backend.
      // If the worker restarted and marked the job as terminal, clear stale frontend state
      // so the user can proceed instead of being stuck on "任务正在运行".
      if (activeJobId) {
        try {
          const jobInfo = await getIngestJob(activeJobId);
          const terminalStatuses = ['done', 'error', 'cancelled'];
          if (terminalStatuses.includes(jobInfo.status)) {
            setActiveJobId(null);
            setIsProcessing(false);
            localStorage.removeItem('ingest_active_job_id');
            // Fall through: allow the current action to continue
          } else {
            addToast('当前有任务正在运行，请稍后重试', 'info');
            return;
          }
        } catch {
          // Cannot verify — err on the side of allowing the action
          setActiveJobId(null);
          setIsProcessing(false);
          localStorage.removeItem('ingest_active_job_id');
        }
      } else {
        addToast('当前有任务正在运行，请稍后重试', 'info');
        return;
      }
    }
    if (!selectedLibrary) {
      addToast('所选文献库不存在，请刷新后重试', 'error');
      return;
    }
    if (!autoDownloadMissingScholarPdfs && selectedLibrary.downloaded_count <= 0) {
      addToast('该文献库暂无已下载 PDF，请先上传/导入 PDF 或开启“自动下载缺失 PDF”', 'info');
      return;
    }
    const missingCount = Math.max(0, selectedLibrary.paper_count - selectedLibrary.downloaded_count);
    const confirmed = window.confirm(
      `将文献库「${selectedLibrary.name}」同步到向量库「${selectedBaseName}」。\n` +
      `总文献 ${selectedLibrary.paper_count}，已下载 ${selectedLibrary.downloaded_count}，缺失 ${missingCount}。\n` +
      `${autoDownloadMissingScholarPdfs ? '已开启自动下载缺失 PDF，任务耗时可能较长。' : '未开启自动下载缺失 PDF。'}\n继续执行吗？`,
    );
    if (!confirmed) return;

    setIsProcessing(true);
    setGlobalProgress('正在提交文献库→向量库同步任务...');
    const abortCtrl = new AbortController();
    abortRef.current = abortCtrl;
    try {
      if (!selectedBase?.collection) {
        await createCollection(selectedBaseName);
        await loadCollections();
      }
      await ensureCollectionLibraryBinding(selectedBaseName);
      const res = await ingestScholarLibrary(selectedLibraryId, {
        collection: selectedBaseName,
        skip_duplicate_doi: true,
        skip_unchanged: true,
        auto_download_missing: autoDownloadMissingScholarPdfs,
        enrich_tables: enrichment.enrich_tables,
        enrich_figures: enrichment.enrich_figures,
        llm_text_provider: enrichment.llm_text_provider,
        llm_text_model: enrichment.llm_text_model,
        llm_text_concurrency: enrichment.llm_text_concurrency,
        llm_vision_provider: enrichment.llm_vision_provider,
        llm_vision_model: enrichment.llm_vision_model,
        llm_vision_concurrency: enrichment.llm_vision_concurrency,
      });
      if (!res?.job_id) {
        throw new Error('未返回 job_id');
      }
      setActiveJobId(res.job_id);
      localStorage.setItem('ingest_active_job_id', res.job_id);
      setGlobalProgress(
        `任务已提交：准备入库 ${res.pdf_ready_count} 篇（缺失 PDF ${res.missing_pdf_count} 篇）`,
      );
      if ((res.removed_papers ?? 0) > 0) {
        addToast(
          `已从向量库移除 ${res.removed_papers} 篇（文献库中已删除）；可入库 ${res.pdf_ready_count} 篇`,
          'success',
        );
      } else if (res.attempted_downloads > 0) {
        addToast(
          `任务已启动：可入库 ${res.pdf_ready_count} 篇；自动下载 ${res.downloaded_now}/${res.attempted_downloads} 成功`,
          'success',
        );
      } else {
        addToast(
          `文献库→向量库同步任务已启动（可入库 ${res.pdf_ready_count} 篇）`,
          'success',
        );
      }

      for await (const evt of streamIngestJobEvents(res.job_id, abortCtrl.signal, 0)) {
        applyIngestEvent(evt);
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        const msg = err instanceof Error ? err.message : String(err);
        addToast(`文献库→向量库同步失败: ${msg}`, 'error');
        setGlobalProgress(`失败: ${msg}`);
      }
    } finally {
      setIsProcessing(false);
      setIsCancelling(false);
      abortRef.current = null;
    }
  };


  const handleAbort = async () => {
    if (isCancelling) return;
    setIsCancelling(true);
    try {
      if (activeJobId) {
        await cancelIngestJob(activeJobId);
        setGlobalProgress('取消请求已发送，正在停止后台任务...');
        addToast('正在取消后台任务...', 'info');
      } else {
        abortRef.current?.abort();
        setIsProcessing(false);
        setGlobalProgress('已停止处理');
        addToast('已停止处理', 'info');
        setIsCancelling(false);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`取消失败: ${msg}`, 'error');
      setIsCancelling(false);
    }
  };

  // ---- Create knowledge base (library) ----

  const handleCreateKnowledgeBase = async () => {
    if (!newCollectionName.trim()) return;
    setShowCreateCollectionModal(false);
    addToast(`正在创建知识库: ${newCollectionName}...`, 'info');
    try {
      await createLibrary({ name: newCollectionName.trim() });
      addToast(`知识库 ${newCollectionName} 创建成功`, 'success');
      setNewCollectionName('');
      await loadScholarLibrariesForIngest();
      await loadCollections();
      setSelectedBaseName(newCollectionName.trim());
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`创建失败: ${msg}`, 'error');
    }
  };

  // ---- 覆盖范围摘要：打开弹窗并加载 ----
  const handleOpenScopeModal = async (name: string) => {
    setScopeModalCollection(name);
    setScopeModalOpen(true);
    setScopeSummaryText('');
    setScopeUpdatedAt(null);
    setScopeModalLoading(true);
    try {
      const res = await getCollectionScope(name);
      setScopeSummaryText(res.scope_summary ?? '');
      setScopeUpdatedAt(res.updated_at ?? null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`加载摘要失败: ${msg}`, 'error');
    } finally {
      setScopeModalLoading(false);
    }
  };

  const handleSaveScope = async () => {
    if (scopeModalCollection == null) return;
    setScopeModalSaving(true);
    try {
      await updateCollectionScope(scopeModalCollection, scopeSummaryText);
      setScopeUpdatedAt(new Date().toISOString());
      addToast('已保存覆盖范围摘要', 'success');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`保存失败: ${msg}`, 'error');
    } finally {
      setScopeModalSaving(false);
    }
  };

  const handleRefreshScopeInModal = async () => {
    if (scopeModalCollection == null) return;
    setScopeRefreshingCollection(scopeModalCollection);
    try {
      const res = await refreshCollectionScope(scopeModalCollection);
      setScopeSummaryText(res.scope_summary ?? '');
      setScopeUpdatedAt(new Date().toISOString());
      addToast('已用 LLM 重新生成摘要', 'success');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`刷新失败: ${msg}`, 'error');
    } finally {
      setScopeRefreshingCollection(null);
    }
  };

  // ---- Refresh collection scope (列表内按钮) ----
  const handleRefreshScope = async (name: string) => {
    setScopeRefreshingCollection(name);
    try {
      const res = await refreshCollectionScope(name);
      addToast(res.scope_summary ? `已刷新「${name}」覆盖范围摘要` : `已刷新「${name}」（暂无摘要）`, 'success');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`刷新覆盖范围失败: ${msg}`, 'error');
    } finally {
      setScopeRefreshingCollection(null);
    }
  };

  const [forceRebuildingBase, setForceRebuildingBase] = useState<string | null>(null);

  const handleForceRebuild = async (baseName: string) => {
    const base = mergedBases.find((b) => b.name === baseName);
    if (!base?.library || base.library.id < 0) {
      addToast('该知识库尚无文献库，无法重新建库', 'info');
      return;
    }
    if (
      !window.confirm(
        `确定要强制重新建库「${baseName}」吗？\n将删除向量库数据并重新切块、向量化，文献库保留。此操作耗时较长。`,
      )
    ) {
      return;
    }
    setForceRebuildingBase(baseName);
    const abortCtrl = new AbortController();
    abortRef.current = abortCtrl;
    try {
      if (base.collection) {
        await deleteCollection(baseName);
        await loadCollections();
      }
      await createCollection(baseName);
      await loadCollections();
      await ensureCollectionLibraryBinding(baseName);
      const res = await ingestScholarLibrary(base.library.id, {
        collection: baseName,
        skip_duplicate_doi: true,
        skip_unchanged: false,
        auto_download_missing: false,
        enrich_tables: enrichment.enrich_tables,
        enrich_figures: enrichment.enrich_figures,
        llm_text_provider: enrichment.llm_text_provider,
        llm_text_model: enrichment.llm_text_model,
        llm_text_concurrency: enrichment.llm_text_concurrency,
        llm_vision_provider: enrichment.llm_vision_provider,
        llm_vision_model: enrichment.llm_vision_model,
        llm_vision_concurrency: enrichment.llm_vision_concurrency,
      });
      if (!res?.job_id) throw new Error('未返回 job_id');
      setActiveJobId(res.job_id);
      setGlobalProgress(`强制重新建库已启动：${baseName}`);
      for await (const evt of streamIngestJobEvents(res.job_id, abortCtrl.signal, 0)) {
        applyIngestEvent(evt);
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        const msg = err instanceof Error ? err.message : String(err);
        addToast(`强制重新建库失败: ${msg}`, 'error');
      }
    } finally {
      setForceRebuildingBase(null);
      abortRef.current = null;
    }
  };

  const handleConfirmDeleteWithPassword = async () => {
    if (!deleteConfirmModal || !deletePassword.trim() || !user) return;
    setDeleteVerifying(true);
    try {
      await apiLogin({ user_id: user.user_id, password: deletePassword });
      const { baseName, libraryId } = deleteConfirmModal;
      if (libraryId >= 0) {
        await deleteLibrary(libraryId);
      }
      try {
        await deleteCollection(baseName);
      } catch (e) {
        // 可能仅有文献库无集合
        if (String((e as Error).message).toLowerCase().includes('not found') === false) throw e;
      }
      addToast(`知识库 ${baseName} 已删除`, 'success');
      if (selectedBaseName === baseName) {
        setSelectedBaseName(null);
        setCurrentCollection('');
      }
      setDeleteConfirmModal(null);
      setDeletePassword('');
      await loadScholarLibrariesForIngest();
      await loadCollections();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(msg.includes('Invalid') || msg.includes('401') ? '密码错误' : `删除失败: ${msg}`, 'error');
    } finally {
      setDeleteVerifying(false);
    }
  };

  const hasCollections = backendCollections.length > 0;
  const isConnected = !connectError && !collectionsLoading;
  const canUploadToLibrary = isConnected && selectedLibraryId != null && selectedLibraryId >= 0;

  // ---- Connection error: show reconnect panel ----
  if (connectError && !collectionsLoading) {
    return (
      <div className="max-w-2xl mx-auto p-8 animate-in fade-in">
        <div className="bg-white p-12 rounded-3xl shadow-xl text-center border border-gray-100">
          <div className="w-20 h-20 bg-red-50 text-red-500 rounded-full flex items-center justify-center mx-auto mb-6">
            <AlertCircle size={40} />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">无法连接后端服务</h2>
          <p className="text-gray-500 mb-2">请确认后端服务已启动，Milvus 数据库可用</p>
          <p className="text-xs text-red-400 mb-6 font-mono">{connectError}</p>
          <div className="text-sm text-gray-400 mb-4">
            当前地址: <span className="font-mono text-gray-600">{dbAddress}</span>
          </div>
          <button
            onClick={loadCollections}
            className="px-6 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
          >
            重新连接
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto space-y-8 p-8 animate-in fade-in">
      {/* Hidden file inputs */}
      <input
        ref={boundLibraryFileInputRef}
        id="bound-library-file-input"
        type="file"
        accept=".pdf"
        multiple
        className="hidden"
        onChange={handleBoundLibraryFileChange}
      />
      <input
        ref={boundLibraryFolderInputRef}
        id="bound-library-folder-input"
        name="bound-library-folder-input"
        type="file"
        accept=".pdf"
        multiple
        // @ts-expect-error: webkitdirectory is non-standard but widely supported
        webkitdirectory=""
        className="hidden"
        onChange={handleBoundLibraryFolderChange}
      />

      {/* 状态卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="col-span-1 bg-gray-900 text-white rounded-2xl p-6 shadow-lg flex flex-col justify-between relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4 opacity-10">
            <Database size={100} />
          </div>
          <div>
            <div className="flex items-center gap-2 text-gray-400 text-xs font-bold uppercase mb-2">
              <HardDrive size={14} /> Milvus
            </div>
            <div className="text-2xl font-bold font-mono tracking-tight">
              {dbAddress}
            </div>
          </div>
          <div className="mt-6 space-y-2 text-sm text-gray-300">
            <div className="flex justify-between">
              <span>状态:</span>
              <span className="text-green-400 flex items-center gap-1">
                <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                已连接
              </span>
            </div>
            <div className="flex justify-between">
              <span>集合数:</span>
              <span className="text-white">{backendCollections.length}</span>
            </div>
            {selectedBaseName && (
              <>
                <div className="flex justify-between">
                  <span>当前知识库:</span>
                  <span className="text-green-400 truncate">{selectedBaseName}</span>
                </div>
                {selectedBase?.collection && (
                  <div className="flex justify-between">
                    <span>向量记录数:</span>
                    <span className="text-white">
                      {selectedBase.collection.count >= 0 ? selectedBase.collection.count : '?'}
                    </span>
                  </div>
                )}
                {selectedBase?.library && (
                  <div className="flex justify-between gap-2">
                    <span>文献:</span>
                    <span className="text-white truncate text-right">
                      {selectedBase.library.paper_count} 篇 · 已下载 {selectedBase.library.downloaded_count}
                    </span>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        <div className="col-span-2 bg-white border rounded-2xl p-6 shadow-sm flex flex-col justify-center">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h3 className="text-lg font-bold text-gray-900">知识库列表</h3>
              <p className="text-xs text-gray-500">文献库与向量库强绑定，以名称一致管理</p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => { loadCollections(); loadScholarLibrariesForIngest(); }}
                disabled={collectionsLoading || scholarLibrariesLoading}
                className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                title="刷新列表"
              >
                <RefreshCw size={16} className={collectionsLoading || scholarLibrariesLoading ? 'animate-spin' : ''} />
              </button>
              <button
                onClick={() => setShowCreateCollectionModal(true)}
                className="flex items-center gap-2 px-4 py-2 bg-blue-50 text-blue-600 rounded-lg text-sm font-bold hover:bg-blue-100 transition-colors"
              >
                <Plus size={16} /> 新建知识库
              </button>
            </div>
          </div>

          {mergedBases.length > 0 ? (
            <div className="space-y-2 max-h-64 overflow-auto">
              {mergedBases.map((base) => (
                <div
                  key={base.name}
                  onClick={() => setSelectedBaseName(base.name)}
                  className={`flex items-center justify-between p-3 rounded-xl border cursor-pointer transition-colors ${
                    selectedBaseName === base.name
                      ? 'border-blue-400 bg-blue-50'
                      : 'border-gray-200 bg-gray-50 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <Database size={16} className={selectedBaseName === base.name ? 'text-blue-500' : 'text-gray-400'} />
                    <div className="min-w-0">
                      <div className="font-medium text-sm text-gray-800 truncate">{base.name}</div>
                      <div className="text-xs text-gray-400">
                        {base.library
                          ? `文献 ${base.library.paper_count} · 已下载 ${base.library.downloaded_count}`
                          : '仅向量库'}
                        {base.collection && base.collection.count >= 0
                          ? ` · 向量 ${base.collection.count} 条`
                          : ''}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1 flex-shrink-0">
                    {base.collection && (
                      <>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleOpenScopeModal(base.name); }}
                          className="p-1.5 text-gray-300 hover:text-emerald-600 hover:bg-emerald-50 rounded-lg transition-colors"
                          title="查看/编辑覆盖范围摘要"
                        >
                          <List size={14} />
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleRefreshScope(base.name); }}
                          disabled={scopeRefreshingCollection === base.name}
                          className="p-1.5 text-gray-300 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors disabled:opacity-50"
                          title="刷新覆盖范围摘要（LLM 重新生成）"
                        >
                          <RefreshCw size={14} className={scopeRefreshingCollection === base.name ? 'animate-spin' : ''} />
                        </button>
                        {base.library && base.library.id >= 0 && (
                          <button
                            onClick={(e) => { e.stopPropagation(); handleForceRebuild(base.name); }}
                            disabled={forceRebuildingBase === base.name || isProcessing}
                            className="p-1.5 text-gray-300 hover:text-amber-600 hover:bg-amber-50 rounded-lg transition-colors disabled:opacity-50"
                            title="强制重新建库（清空向量后全量重新切块、向量化）"
                          >
                            {forceRebuildingBase === base.name ? <Loader2 size={14} className="animate-spin" /> : <RotateCw size={14} />}
                          </button>
                        )}
                      </>
                    )}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteConfirmModal({ baseName: base.name, libraryId: base.library?.id ?? -1 });
                        setDeletePassword('');
                      }}
                      className="p-1.5 text-gray-300 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                      title={`删除知识库 ${base.name}（需验证密码）`}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-center">
              <p className="text-amber-700 text-sm font-medium mb-2">暂无知识库</p>
              <p className="text-amber-600 text-xs mb-3">
                点击「新建知识库」创建文献库与向量库（名称一致）
              </p>
              <button
                onClick={() => setShowCreateCollectionModal(true)}
                className="px-4 py-2 bg-amber-600 text-white rounded-lg text-sm font-bold hover:bg-amber-700 transition-colors"
              >
                <Plus size={14} className="inline mr-1 -mt-0.5" /> 新建知识库
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 同步文献库到向量库 + 上传本地 PDF 到文献库 */}
      <div className="bg-white border rounded-2xl p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-bold text-gray-900">同步文献库到向量库</h3>
            <p className="text-xs text-gray-500">
              上传本地 PDF 到当前知识库文献库；将文献库增量同步到向量库（有则增删，无则初始化建库）
            </p>
          </div>
          <button
            onClick={() => { loadScholarLibrariesForIngest(); loadCollections(); }}
            disabled={scholarLibrariesLoading}
            className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
            title="刷新"
          >
            <RefreshCw size={16} className={scholarLibrariesLoading ? 'animate-spin' : ''} />
          </button>
        </div>

        {!selectedBaseName ? (
          <div className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
            请先在左侧选择一个知识库。
          </div>
        ) : !selectedLibrary ? (
          <div className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
            当前知识库尚无文献库，请先在「文献检索」页面创建并导入文献，或使用下方按钮上传 PDF 后刷新。
          </div>
        ) : (
          <div className="space-y-4">
            <div className="text-sm text-gray-700 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2">
              当前知识库：<span className="font-semibold text-gray-900">{selectedBaseName}</span>
              {' · '}文献 {selectedLibrary.paper_count} · 已下载 {selectedLibrary.downloaded_count}
              {selectedBase?.collection && (
                <span> · 向量 {selectedBase.collection.count >= 0 ? selectedBase.collection.count : '?'} 条</span>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-gray-500 mr-1">上传本地 PDF 到文献库：</span>
              <button
                type="button"
                onClick={handlePickBoundLibraryFiles}
                disabled={boundLibraryImporting || isProcessing || !canUploadToLibrary}
                className="flex items-center gap-2 px-4 py-2 rounded-lg border border-gray-300 text-gray-700 text-sm font-medium hover:bg-gray-50 disabled:opacity-50"
              >
                <FileText size={14} /> 选择文件
              </button>
              <button
                type="button"
                onClick={handlePickBoundLibraryFolder}
                disabled={boundLibraryImporting || isProcessing || !canUploadToLibrary}
                className="flex items-center gap-2 px-4 py-2 rounded-lg border border-gray-300 text-gray-700 text-sm font-medium hover:bg-gray-50 disabled:opacity-50"
              >
                <FolderOpen size={14} /> 选择文件夹
              </button>
              <button
                type="button"
                onClick={handleImportBoundLibraryPdfs}
                disabled={boundLibraryImportFiles.length === 0 || boundLibraryImporting || isProcessing}
                className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50 inline-flex items-center gap-2"
              >
                {boundLibraryImporting ? <Loader2 size={14} className="animate-spin" /> : null}
                导入到文献库
              </button>
              {boundLibraryImportFiles.length > 0 && (
                <span className="text-xs text-gray-500">已选择 {boundLibraryImportFiles.length} 个 PDF</span>
              )}
            </div>

            {boundLibraryImportFiles.length > 0 && (
              <div className="text-xs text-gray-600 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 max-h-28 overflow-auto space-y-0.5">
                {boundLibraryImportFiles.slice(0, 8).map((f, idx) => {
                  const relPath = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
                  return <div key={`${relPath}-${idx}`}>{relPath}</div>;
                })}
                {boundLibraryImportFiles.length > 8 ? (
                  <div className="text-gray-400">... 还有 {boundLibraryImportFiles.length - 8} 个文件</div>
                ) : null}
              </div>
            )}

            {boundLibraryImportSummary && (
              <div className="text-xs text-gray-700 bg-blue-50 border border-blue-100 rounded-lg px-3 py-2 space-y-1">
                <div>
                  导入统计：总计 {boundLibraryImportSummary.total_files}，新增 {boundLibraryImportSummary.imported}，关联已有{' '}
                  {boundLibraryImportSummary.linked_existing}，跳过重复 {boundLibraryImportSummary.skipped_duplicates}
                </div>
                {boundLibraryImportSummary.errors.length > 0 && (
                  <div className="text-red-600">错误示例：{boundLibraryImportSummary.errors.slice(0, 2).join('；')}</div>
                )}
              </div>
            )}

            {/* LLM 增强选项 */}
            <div className="pt-4 border-t border-gray-100">
              <div className="text-xs font-semibold text-gray-700 mb-3">
                入库解析选项
              </div>
              <div className="flex flex-col md:flex-row gap-3">
                <div
                  className={`flex-1 px-4 py-3 rounded-xl border transition-all ${
                    enrichment.enrich_figures
                      ? 'border-blue-400 bg-blue-50 text-blue-700 shadow-sm'
                      : 'border-gray-200 bg-gray-50 text-gray-500 hover:border-gray-300'
                  }`}
                >
                  <label className="flex items-center gap-2.5 cursor-pointer">
                    <input
                      id="enrich-figures"
                      name="enrich-figures"
                      type="checkbox"
                      checked={enrichment.enrich_figures}
                      onChange={(e) =>
                        setEnrichment((prev) => ({ ...prev, enrich_figures: e.target.checked }))
                      }
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <div className="text-left">
                      <div className="text-sm font-medium">图片描述</div>
                      <div className="text-[10px] text-gray-400">Vision 模型解析图表含义</div>
                    </div>
                  </label>
                  <div className="mt-2 space-y-1">
                    <select
                      value={enrichment.llm_vision_provider || ''}
                      onChange={(e) => {
                        const providerId = e.target.value || null;
                        const provider = llmProviders.find((p) => p.id === providerId) || null;
                        setEnrichment((prev) => ({
                          ...prev,
                          llm_vision_provider: providerId,
                          llm_vision_model: provider?.default_model || null,
                        }));
                      }}
                      disabled={isProcessing || llmProviders.length === 0}
                      className="w-full rounded border border-gray-300 bg-white px-2 py-1 text-xs text-gray-700"
                    >
                      <option value="">选择 Provider</option>
                      {llmProviders.map((p) => (
                        <option key={`vision-provider-${p.id}`} value={p.id}>
                          {p.id}
                        </option>
                      ))}
                    </select>
                    <select
                      value={enrichment.llm_vision_model || '__default__'}
                      onChange={(e) =>
                        setEnrichment((prev) => ({
                          ...prev,
                          llm_vision_model: e.target.value === '__default__' ? null : e.target.value,
                        }))
                      }
                      disabled={isProcessing || !figureProvider}
                      className="w-full rounded border border-gray-300 bg-white px-2 py-1 text-xs text-gray-700"
                    >
                      <option value="__default__">模型默认（Provider default）</option>
                      {(figureProvider?.models || []).map((m) => (
                        <option key={`vision-model-${m}`} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-gray-500 whitespace-nowrap">并发</span>
                      <select
                        value={enrichment.llm_vision_concurrency ?? 1}
                        onChange={(e) =>
                          setEnrichment((prev) => ({
                            ...prev,
                            llm_vision_concurrency: Number(e.target.value) || 1,
                          }))
                        }
                        disabled={isProcessing}
                        className="w-full rounded border border-gray-300 bg-white px-2 py-1 text-xs text-gray-700"
                      >
                        {concurrencyOptions.map((v) => (
                          <option key={`vision-concurrency-${v}`} value={v}>
                            {v}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
                <div
                  className={`flex-1 px-4 py-3 rounded-xl border transition-all ${
                    enrichment.enrich_tables
                      ? 'border-purple-400 bg-purple-50 text-purple-700 shadow-sm'
                      : 'border-gray-200 bg-gray-50 text-gray-500 hover:border-gray-300'
                  }`}
                >
                  <label className="flex items-center gap-2.5 cursor-pointer">
                    <input
                      id="enrich-tables"
                      name="enrich-tables"
                      type="checkbox"
                      checked={enrichment.enrich_tables}
                      onChange={(e) =>
                        setEnrichment((prev) => ({ ...prev, enrich_tables: e.target.checked }))
                      }
                      className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                    />
                    <div className="text-left">
                      <div className="text-sm font-medium">表格解析</div>
                      <div className="text-[10px] text-gray-400">LLM 生成表格语义摘要</div>
                    </div>
                  </label>
                  <div className="mt-2 space-y-1">
                    <select
                      value={enrichment.llm_text_provider || ''}
                      onChange={(e) => {
                        const providerId = e.target.value || null;
                        const provider = llmProviders.find((p) => p.id === providerId) || null;
                        setEnrichment((prev) => ({
                          ...prev,
                          llm_text_provider: providerId,
                          llm_text_model: provider?.default_model || null,
                        }));
                      }}
                      disabled={isProcessing || llmProviders.length === 0}
                      className="w-full rounded border border-gray-300 bg-white px-2 py-1 text-xs text-gray-700"
                    >
                      <option value="">选择 Provider</option>
                      {llmProviders.map((p) => (
                        <option key={`text-provider-${p.id}`} value={p.id}>
                          {p.id}
                        </option>
                      ))}
                    </select>
                    <select
                      value={enrichment.llm_text_model || '__default__'}
                      onChange={(e) =>
                        setEnrichment((prev) => ({
                          ...prev,
                          llm_text_model: e.target.value === '__default__' ? null : e.target.value,
                        }))
                      }
                      disabled={isProcessing || !tableProvider}
                      className="w-full rounded border border-gray-300 bg-white px-2 py-1 text-xs text-gray-700"
                    >
                      <option value="__default__">模型默认（Provider default）</option>
                      {(tableProvider?.models || []).map((m) => (
                        <option key={`text-model-${m}`} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-gray-500 whitespace-nowrap">并发</span>
                      <select
                        value={enrichment.llm_text_concurrency ?? 1}
                        onChange={(e) =>
                          setEnrichment((prev) => ({
                            ...prev,
                            llm_text_concurrency: Number(e.target.value) || 1,
                          }))
                        }
                        disabled={isProcessing}
                        className="w-full rounded border border-gray-300 bg-white px-2 py-1 text-xs text-gray-700"
                      >
                        {concurrencyOptions.map((v) => (
                          <option key={`text-concurrency-${v}`} value={v}>
                            {v}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              </div>
              {!enrichment.enrich_figures && !enrichment.enrich_tables && (
                <p className="text-[10px] text-gray-400 mt-2">
                  未选择增强项，将仅执行基础解析（速度更快）。
                </p>
              )}
            </div>

            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={autoDownloadMissingScholarPdfs}
                onChange={(e) => setAutoDownloadMissingScholarPdfs(e.target.checked)}
                className="accent-blue-600"
              />
              同步前自动尝试下载缺失 PDF（耗时较长）
            </label>

            <div className="text-xs text-gray-500 bg-blue-50 border border-blue-100 rounded-lg px-3 py-2">
              增量策略：默认启用 DOI 去重与未变化文件跳过，重复执行不会重复入库。
            </div>

            <div className="flex justify-end gap-2">
              {isProcessing && (
                <button
                  onClick={handleAbort}
                  disabled={isCancelling}
                  className="px-4 py-2 bg-red-50 text-red-600 rounded-lg text-sm font-medium hover:bg-red-100"
                >
                  {isCancelling ? '取消中...' : '取消任务'}
                </button>
              )}
              <button
                onClick={handleStartScholarLibraryIngest}
                disabled={!selectedLibraryId || selectedLibraryId < 0 || isProcessing}
                className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
              >
                同步文献库到向量库
              </button>
            </div>
          </div>
        )}
      </div>

      {/* 已入库文件列表 */}
      {hasCollections && selectedBaseName && (() => {
        const totalPapers = papers.length;
        const totalPages = Math.max(1, Math.ceil(totalPapers / papersPageSize));
        const effectivePage = Math.min(Math.max(1, papersPage), totalPages);
        const displayedPapers = papers.slice(
          (effectivePage - 1) * papersPageSize,
          effectivePage * papersPageSize,
        );
        return (
          <div className="bg-white border rounded-2xl overflow-hidden shadow-sm">
            <div className="px-6 py-4 border-b bg-gray-50 flex justify-between items-center flex-wrap gap-2">
              <div className="flex items-center gap-3 flex-wrap">
                <span className="font-bold text-sm">已入库文件</span>
                <span className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full">
                  {selectedBaseName}
                </span>
                {totalPapers > 0 && (
                  <span className="text-xs text-gray-400">
                    {totalPapers} 个文件 · {papers.reduce((s, p) => s + p.row_count, 0)} 条记录
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {totalPapers > 0 && (
                  <label className="flex items-center gap-1.5 text-xs text-gray-500">
                    每页
                    <select
                      value={papersPageSize}
                      onChange={(e) => {
                        setPapersPageSize(Number(e.target.value) as 10 | 20);
                        setPapersPage(1);
                      }}
                      className="border rounded px-2 py-1 text-gray-700 bg-white"
                    >
                      <option value={10}>10</option>
                      <option value={20}>20</option>
                    </select>
                    条
                  </label>
                )}
                <button
                  onClick={loadPapers}
                  disabled={papersLoading}
                  className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                  title="刷新文件列表"
                >
                  <RefreshCw size={14} className={papersLoading ? 'animate-spin' : ''} />
                </button>
              </div>
            </div>
            {papersLoading ? (
              <div className="px-6 py-8 text-center text-gray-400 text-sm">
                <Loader2 size={20} className="animate-spin mx-auto mb-2" />
                加载中...
              </div>
            ) : totalPapers === 0 ? (
              <div className="px-6 py-8 text-center text-gray-400 text-sm">
                该集合暂无已入库的文件
              </div>
            ) : (
              <>
                <table className="w-full text-sm table-fixed">
                  <thead>
                    <tr className="text-left text-gray-500 border-b">
                      <th className="px-6 py-2.5 font-medium w-[40%]">文件名</th>
                      <th className="px-6 py-2.5 font-medium w-24">大小</th>
                      <th className="px-6 py-2.5 font-medium w-20">Chunks</th>
                      <th className="px-6 py-2.5 font-medium w-40">图表解析</th>
                      <th className="px-6 py-2.5 font-medium w-24">状态</th>
                      <th className="px-6 py-2.5 font-medium w-36">入库时间</th>
                      <th className="px-6 py-2.5 font-medium text-right w-16">操作</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {displayedPapers.map((p) => (
                      <tr key={p.paper_id} className="hover:bg-gray-50 transition-colors">
                        <td className="px-6 py-2.5 align-middle">
                          <div className="min-w-0">
                            <div className="flex items-center gap-2 min-w-0">
                              <FileText size={16} className="text-gray-400 flex-shrink-0" />
                              <span
                                className="text-base text-gray-800 leading-relaxed truncate block min-w-0"
                                style={{ maxWidth: '100%' }}
                                title={p.filename || p.paper_id}
                              >
                                {p.filename || p.paper_id}
                              </span>
                              {p.library_id != null && (
                                <span className="flex-shrink-0 inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-blue-50 text-blue-600" title={`文献库 ID: ${p.library_id}`}>
                                  <Link2 size={10} className="mr-0.5" />
                                  {t('ingest.fromLibrary')}
                                </span>
                              )}
                            </div>
                            {(p.title || p.doi || p.venue || p.year) && (
                              <div className="mt-1 text-xs text-gray-500 truncate" title={p.title || p.doi || undefined}>
                                {p.title && p.title !== p.filename ? p.title : p.doi || ''}
                                {p.venue ? ` · ${p.venue}` : ''}
                                {p.year != null ? ` · ${p.year}` : ''}
                              </div>
                            )}
                          </div>
                        </td>
                        <td className="px-6 py-2.5 text-gray-500">{formatSize(p.file_size)}</td>
                        <td className="px-6 py-2.5 text-gray-500">{p.chunk_count}</td>
                        <td className="px-6 py-2.5 text-[11px] text-gray-500">
                          <div className="flex flex-col gap-0.5">
                            <span>
                              表格: {p.enrich_tables_enabled ? `已开 (${p.table_success || 0}/${p.table_count || 0})` : '未开'}
                            </span>
                            <span>
                              图片: {p.enrich_figures_enabled ? `已开 (${p.figure_success || 0}/${p.figure_count || 0})` : '未开'}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-2.5">
                          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${
                            p.status === 'done'
                              ? 'bg-green-50 text-green-600'
                              : 'bg-red-50 text-red-600'
                          }`}>
                            {p.status === 'done' ? <CheckCircle2 size={10} /> : <AlertCircle size={10} />}
                            {p.status === 'done' ? '已入库' : '失败'}
                          </span>
                        </td>
                        <td className="px-6 py-2.5 text-gray-400 text-xs">
                          {new Date(p.created_at * 1000).toLocaleString('zh-CN')}
                        </td>
                        <td className="px-6 py-2.5 text-right">
                          <button
                            onClick={() => handleDeletePaper(p.paper_id, p.filename)}
                            className="p-1.5 text-gray-300 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                            title={`删除 ${p.filename || p.paper_id} 及其全部向量数据`}
                          >
                            <Trash2 size={14} />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {totalPages > 1 && (
                  <div className="px-6 py-3 border-t bg-gray-50 flex items-center justify-between text-sm">
                    <span className="text-gray-500">
                      第 {effectivePage} / {totalPages} 页，共 {totalPapers} 条
                    </span>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setPapersPage((p) => Math.max(1, p - 1))}
                        disabled={effectivePage <= 1}
                        className="p-2 rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-100 disabled:opacity-40 disabled:pointer-events-none"
                        title="上一页"
                      >
                        <ChevronLeft size={16} />
                      </button>
                      <button
                        onClick={() => setPapersPage((p) => Math.min(totalPages, p + 1))}
                        disabled={effectivePage >= totalPages}
                        className="p-2 rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-100 disabled:opacity-40 disabled:pointer-events-none"
                        title="下一页"
                      >
                        <ChevronRight size={16} />
                      </button>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        );
      })()}

      {/* 解析状态：同步任务进度 */}
      {isProcessing && globalProgress && (
        <div className="mb-4 bg-amber-50/80 border border-amber-200 rounded-xl px-4 py-3 flex items-center gap-2 text-sm text-amber-800">
          <Loader2 size={18} className="animate-spin flex-shrink-0" />
          {globalProgress}
        </div>
      )}

      {/* 删除知识库：密码确认 Modal */}
      <Modal
        open={deleteConfirmModal !== null}
        onClose={() => { setDeleteConfirmModal(null); setDeletePassword(''); }}
        title="删除知识库（危险操作）"
        maxWidth="max-w-md"
      >
        {deleteConfirmModal && (
          <div className="space-y-4">
            <p className="text-sm text-gray-700">
              将永久删除知识库「<strong>{deleteConfirmModal.baseName}</strong>」及其文献库、向量库与所有相关文件，此操作不可恢复。
            </p>
            <p className="text-sm text-amber-700">
              请输入当前登录账号密码以确认：
            </p>
            <input
              type="password"
              value={deletePassword}
              onChange={(e) => setDeletePassword(e.target.value)}
              placeholder="密码"
              className="w-full border rounded-md p-2 text-sm focus:ring-2 focus:ring-red-500 outline-none"
              onKeyDown={(e) => e.key === 'Enter' && handleConfirmDeleteWithPassword()}
            />
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => { setDeleteConfirmModal(null); setDeletePassword(''); }}
                className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg text-sm"
              >
                取消
              </button>
              <button
                onClick={handleConfirmDeleteWithPassword}
                disabled={!deletePassword.trim() || deleteVerifying}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 text-sm flex items-center gap-2"
              >
                {deleteVerifying ? <Loader2 size={14} className="animate-spin" /> : null}
                确认删除
              </button>
            </div>
          </div>
        )}
      </Modal>

      {/* 删除文献：密码确认 Modal */}
      <Modal
        open={deletePaperModal !== null}
        onClose={() => { setDeletePaperModal(null); setDeletePaperPassword(''); }}
        title="删除文献（危险操作）"
        maxWidth="max-w-md"
      >
        {deletePaperModal && (
          <div className="space-y-4">
            <p className="text-sm text-gray-700">
              将永久删除文献「<strong>{deletePaperModal.filename || deletePaperModal.paperId}</strong>」及其向量数据与磁盘文件，此操作不可恢复。
            </p>
            <p className="text-sm text-amber-700">
              请输入当前登录账号密码以确认：
            </p>
            <input
              type="password"
              value={deletePaperPassword}
              onChange={(e) => setDeletePaperPassword(e.target.value)}
              placeholder="密码"
              className="w-full border rounded-md p-2 text-sm focus:ring-2 focus:ring-red-500 outline-none"
              onKeyDown={(e) => e.key === 'Enter' && handleConfirmDeletePaperWithPassword()}
            />
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => { setDeletePaperModal(null); setDeletePaperPassword(''); }}
                className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg text-sm"
              >
                取消
              </button>
              <button
                onClick={handleConfirmDeletePaperWithPassword}
                disabled={!deletePaperPassword.trim() || deletePaperVerifying}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 text-sm flex items-center gap-2"
              >
                {deletePaperVerifying ? <Loader2 size={14} className="animate-spin" /> : null}
                确认删除
              </button>
            </div>
          </div>
        )}
      </Modal>

      {/* 新建知识库 Modal */}
      <Modal
        open={showCreateCollectionModal}
        onClose={() => setShowCreateCollectionModal(false)}
        title="新建知识库"
        maxWidth="max-w-md"
      >
        <div className="space-y-4">
          <div>
            <label className="text-xs font-medium text-gray-500 uppercase">
              知识库名称（即文献库名称）
            </label>
            <input
              id="new-collection-name"
              name="new-collection-name"
              type="text"
              value={newCollectionName}
              onChange={(e) => setNewCollectionName(e.target.value)}
              placeholder="例如 my_research_2026"
              className="w-full mt-1 border rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
              onKeyDown={(e) => e.key === 'Enter' && handleCreateKnowledgeBase()}
            />
          </div>
          <p className="text-xs text-gray-400">
            将创建同名文献库，同步到向量库时再创建向量集合。
          </p>
        </div>
        <div className="mt-6 flex justify-end gap-2">
          <button
            onClick={() => setShowCreateCollectionModal(false)}
            className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg text-sm"
          >
            取消
          </button>
          <button
            onClick={handleCreateKnowledgeBase}
            disabled={!newCollectionName.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm"
          >
            创建
          </button>
        </div>
      </Modal>

      {/* 覆盖范围摘要 - 查看/编辑 */}
      <Modal
        open={scopeModalOpen}
        onClose={() => setScopeModalOpen(false)}
        title={scopeModalCollection ? `覆盖范围摘要 · ${scopeModalCollection}` : '覆盖范围摘要'}
        maxWidth="max-w-xl"
      >
        {scopeModalCollection && (
          <div className="space-y-4">
            {scopeModalLoading ? (
              <div className="py-8 flex items-center justify-center text-gray-500">
                <Loader2 size={28} className="animate-spin" />
              </div>
            ) : (
              <>
                <p className="text-sm text-gray-600 leading-relaxed">
                  用于判断用户查询是否与本地库匹配；可手动编辑或使用下方「LLM 重新生成」根据集合内文档题目自动生成更详细的摘要。
                </p>
                <textarea
                  value={scopeSummaryText}
                  onChange={(e) => setScopeSummaryText(e.target.value)}
                  placeholder="例如：深海共生、海洋生物、共生关系、深海生态与相关研究；或更详细的 2～4 句描述"
                  rows={6}
                  className="w-full min-h-[8rem] border border-gray-200 rounded-lg p-4 text-base leading-relaxed text-gray-800 placeholder:text-gray-400 focus:ring-2 focus:ring-blue-500 outline-none resize-y"
                />
                {scopeUpdatedAt && (
                  <p className="text-xs text-gray-500">
                    更新时间：{new Date(scopeUpdatedAt).toLocaleString('zh-CN')}
                  </p>
                )}
                <div className="flex justify-end gap-3 pt-2">
                  <button
                    onClick={() => setScopeModalOpen(false)}
                    className="px-4 py-2.5 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg text-sm font-medium"
                  >
                    关闭
                  </button>
                  <button
                    onClick={handleRefreshScopeInModal}
                    disabled={scopeRefreshingCollection === scopeModalCollection}
                    className="px-4 py-2.5 bg-gray-100 text-gray-700 hover:bg-gray-200 rounded-lg text-sm font-medium flex items-center gap-1.5 disabled:opacity-50"
                  >
                    {scopeRefreshingCollection === scopeModalCollection ? (
                      <Loader2 size={16} className="animate-spin" />
                    ) : (
                      <RefreshCw size={16} />
                    )}
                    LLM 重新生成
                  </button>
                  <button
                    onClick={handleSaveScope}
                    disabled={scopeModalSaving}
                    className="px-4 py-2.5 bg-blue-600 text-white hover:bg-blue-700 rounded-lg text-sm font-medium disabled:opacity-50 flex items-center gap-1.5"
                  >
                    {scopeModalSaving ? <Loader2 size={16} className="animate-spin" /> : null}
                    保存
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
