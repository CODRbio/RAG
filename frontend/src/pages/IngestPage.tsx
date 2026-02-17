import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Database,
  HardDrive,
  UploadCloud,
  Plus,
  FileText,
  Trash2,
  FolderOpen,
  Loader2,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { useConfigStore, useUIStore, useToastStore } from '../stores';
import { Modal } from '../components/ui/Modal';
import {
  listCollections,
  createCollection,
  deleteCollection,
  listPapers,
  deletePaper,
  uploadFiles,
  processFiles,
  cancelIngestJob,
  listIngestJobs,
  streamIngestJobEvents,
  listLLMProviders,
  type CollectionInfo,
  type IngestProgressEvent,
  type LLMProviderInfo,
  type PaperInfo,
  type UploadedFile,
  type EnrichmentOptions,
} from '../api/ingest';

// ---- Types ----

type FileStatus = 'pending' | 'uploading' | 'parsing' | 'chunking' | 'embedding' | 'indexing' | 'done' | 'error' | 'skipped';

interface FileItem {
  id: string;
  name: string;
  size: number;
  status: FileStatus;
  message?: string;
  chunks?: number;
  /** 上传后后端返回的路径 */
  serverPath?: string;
}

const STATUS_LABELS: Record<FileStatus, string> = {
  pending: '待处理',
  uploading: '上传中',
  parsing: '解析中',
  chunking: '切块中',
  embedding: '向量化中',
  indexing: '入库中',
  done: '完成',
  error: '失败',
  skipped: '已跳过',
};

const STATUS_COLORS: Record<FileStatus, string> = {
  pending: 'bg-gray-100 text-gray-500',
  uploading: 'bg-blue-50 text-blue-600',
  parsing: 'bg-blue-50 text-blue-600',
  chunking: 'bg-blue-50 text-blue-600',
  embedding: 'bg-purple-50 text-purple-600',
  indexing: 'bg-orange-50 text-orange-600',
  done: 'bg-green-50 text-green-600',
  error: 'bg-red-50 text-red-600',
  skipped: 'bg-gray-100 text-gray-500',
};

// ---- Collection Templates ----

const COLLECTION_TEMPLATES = [
  { name: 'deepsea_global', desc: '通用深海文献库' },
  { name: 'deepsea_life', desc: '深海生物与生态' },
  { name: 'deepsea_ocean', desc: '海洋学与地质' },
  { name: 'deepsea_env', desc: '深海环境与保护' },
];

// ---- Component ----

export function IngestPage() {
  const { dbAddress, setDbStatus, currentCollection, setCurrentCollection, setCollections, addCollection } =
    useConfigStore();
  const { showCreateCollectionModal, setShowCreateCollectionModal } = useUIStore();
  const addToast = useToastStore((s) => s.addToast);

  // Connection
  const [connectError, setConnectError] = useState('');

  // Collections from backend
  const [backendCollections, setBackendCollections] = useState<CollectionInfo[]>([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);

  // Papers in collection (持久化文件列表)
  const [papers, setPapers] = useState<PaperInfo[]>([]);
  const [papersLoading, setPapersLoading] = useState(false);
  const [papersPageSize, setPapersPageSize] = useState<10 | 20>(10);
  const [papersPage, setPapersPage] = useState(1);

  // Files
  const [files, setFiles] = useState<FileItem[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [globalProgress, setGlobalProgress] = useState('');
  const [enrichment, setEnrichment] = useState<EnrichmentOptions>({
    enrich_tables: false,
    enrich_figures: false,
    llm_text_provider: null,
    llm_text_model: null,
    llm_text_concurrency: 1,
    llm_vision_provider: null,
    llm_vision_model: null,
    llm_vision_concurrency: 1,
  });
  const [llmProviders, setLlmProviders] = useState<LLMProviderInfo[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  /** 解析状态：表格/图片 enrich 进度，按文件名聚合，用于解析阶段展示 */
  const [enrichLogs, setEnrichLogs] = useState<Record<string, string[]>>({});
  const [newCollectionName, setNewCollectionName] = useState('');
  /** 上传后若发现已入库重复（按 content_hash），弹窗让用户选 跳过 / 覆盖 */
  const [duplicateModal, setDuplicateModal] = useState<{
    uploaded: UploadedFile[];
    duplicatePairs: { uploadedFile: UploadedFile; existingPaper: PaperInfo }[];
  } | null>(null);
  /** 对本次及之后所有重复项执行相同操作，不弹窗 */
  const [duplicateActionPreference, setDuplicateActionPreference] = useState<'skip' | 'overwrite' | null>(null);
  /** 弹窗内勾选「应用到所有类似」 */
  const [applyToAllSimilar, setApplyToAllSimilar] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // ---- Auto-connect & load collections on mount ----
  const loadCollections = useCallback(async () => {
    setCollectionsLoading(true);
    setConnectError('');
    try {
      const cols = await listCollections();
      setBackendCollections(cols);
      setDbStatus('connected');
      // Sync to config store
      const names = cols.map((c) => c.name);
      setCollections(names);
      // If current collection not in list, select first
      if (names.length > 0 && !names.includes(currentCollection)) {
        setCurrentCollection(names[0]);
      }
    } catch (err) {
      console.error('Failed to load collections:', err);
      setConnectError(err instanceof Error ? err.message : '无法连接后端服务');
    } finally {
      setCollectionsLoading(false);
    }
  }, [currentCollection, setCollections, setCurrentCollection, setDbStatus, addToast]);

  const loadLlmProviderOptions = useCallback(async () => {
    try {
      const data = await listLLMProviders();
      setLlmProviders(data.providers || []);
      const defaults = data.parser_defaults || {};
      setEnrichment((prev) => ({
        ...prev,
        llm_text_provider: defaults.llm_text_provider ?? prev.llm_text_provider,
        llm_text_model: defaults.llm_text_model ?? prev.llm_text_model,
        llm_text_concurrency:
          typeof defaults.llm_text_concurrency === 'number'
            ? defaults.llm_text_concurrency
            : (prev.llm_text_concurrency ?? 1),
        llm_vision_provider: defaults.llm_vision_provider ?? prev.llm_vision_provider,
        llm_vision_model: defaults.llm_vision_model ?? prev.llm_vision_model,
        llm_vision_concurrency:
          typeof defaults.llm_vision_concurrency === 'number'
            ? defaults.llm_vision_concurrency
            : (prev.llm_vision_concurrency ?? 1),
      }));
    } catch (err) {
      console.warn('Failed to load llm providers:', err);
    }
  }, []);

  useEffect(() => {
    loadCollections();
    loadLlmProviderOptions();
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
          const running = await listIngestJobs(1, 'running');
          targetJobId = running[0]?.job_id || null;
        }
        if (!targetJobId || cancelled) return;
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

  // ---- Load papers when collection changes ----
  const loadPapers = useCallback(async (): Promise<PaperInfo[]> => {
    if (!currentCollection) { setPapers([]); setPapersPage(1); return []; }
    setPapersLoading(true);
    try {
      const list = await listPapers(currentCollection);
      setPapers(list);
      setPapersPage(1);
      return list;
    } catch {
      setPapers([]);
      return [];
    } finally {
      setPapersLoading(false);
    }
  }, [currentCollection]);

  useEffect(() => {
    loadPapers();
  }, [loadPapers]);

  // ---- Delete paper from collection ----
  const handleDeletePaper = async (paperId: string, filename: string) => {
    if (!window.confirm(`确定删除「${filename || paperId}」？\n将同时删除该文件在集合中的所有向量数据，不可恢复。`)) {
      return;
    }
    addToast(`正在删除 ${filename || paperId}...`, 'info');
    try {
      const res = await deletePaper(currentCollection, paperId);
      addToast(`已删除 ${filename || paperId} (${res.deleted_chunks} chunks)`, 'success');
      loadPapers();
      loadCollections(); // 刷新集合记录数
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`删除失败: ${msg}`, 'error');
    }
  };

  // ---- File selection helpers ----

  const addFilesToList = (fileList: FileList | File[]) => {
    const arr = Array.from(fileList);
    const newItems: FileItem[] = arr
      .filter((f) => f.name.toLowerCase().endsWith('.pdf'))
      .map((f) => ({
        id: `${f.name}-${f.size}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        name: f.name,
        size: f.size,
        status: 'pending' as FileStatus,
      }));
    if (newItems.length === 0) {
      addToast('未找到 PDF 文件', 'info');
      return;
    }
    setFiles((prev) => [...prev, ...newItems]);
    // Store raw File objects in a map for upload
    for (const item of newItems) {
      rawFileMap.current.set(item.id, arr.find((f) => f.name === item.name)!);
    }
  };

  const rawFileMap = useRef<Map<string, File>>(new Map());

  const handleSelectFiles = () => fileInputRef.current?.click();
  const handleSelectFolder = () => folderInputRef.current?.click();

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFilesToList(e.target.files);
    e.target.value = ''; // reset so same files can be re-selected
  };

  const handleRemoveFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
    rawFileMap.current.delete(id);
  };

  const removeFilesByStatuses = (statuses: FileStatus[]) => {
    const target = new Set(statuses);
    setFiles((prev) => {
      const removeIds = prev.filter((f) => target.has(f.status)).map((f) => f.id);
      for (const id of removeIds) {
        rawFileMap.current.delete(id);
      }
      return prev.filter((f) => !target.has(f.status));
    });
  };

  const handleClearDone = () => {
    removeFilesByStatuses(['done', 'skipped']);
  };

  const handleRetryFailed = () => {
    setFiles((prev) =>
      prev.map((f) => (f.status === 'error' ? { ...f, status: 'pending' as FileStatus, message: undefined } : f)),
    );
  };

  const handleClearFailed = () => {
    removeFilesByStatuses(['error']);
  };

  const handleClearPending = () => {
    removeFilesByStatuses(['pending']);
  };

  // ---- Drag & Drop ----

  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files) addFilesToList(e.dataTransfer.files);
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
        const fname = d.file as string;
        const stage = d.stage as FileStatus;
        const message = d.message as string;
        // 后端 events 流的连接保活 heartbeat 仅包含 {job_id, message}
        // 不应当落入文件状态更新，否则会生成空文件行。
        if (fname && stage) {
          updateFileStatusByName(fname, stage, message);
        }
        if (message) setGlobalProgress(message);
        break;
      }
      case 'file_done': {
        const fname = d.file as string;
        const chunks = (d.chunks as number) || 0;
        updateFileStatusByName(fname, 'done', `完成 (${chunks} chunks)`);
        break;
      }
      case 'file_error': {
        const fname = d.file as string;
        const errMsg = d.error as string;
        updateFileStatusByName(fname, 'error', errMsg);
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
          addToast(`入库完成: ${d.total_upserted} 条记录写入 ${currentCollection}`, 'success');
        }
        loadCollections();
        loadPapers();
        setActiveJobId(null);
        localStorage.removeItem('ingest_active_job_id');
        setIsCancelling(false);
        break;
      case 'cancelled':
        setGlobalProgress('任务已取消');
        setFiles((prev) =>
          prev.map((f) =>
            ['uploading', 'parsing', 'chunking', 'embedding', 'indexing'].includes(f.status)
              ? { ...f, status: 'pending' as FileStatus, message: '已取消，可重新开始' }
              : f,
          ),
        );
        setActiveJobId(null);
        localStorage.removeItem('ingest_active_job_id');
        setIsCancelling(false);
        break;
    }
  };

  const handleStartProcess = async () => {
    console.log('[IngestPage] handleStartProcess triggered');
    const pendingFiles = files.filter((f) => f.status === 'pending');
    console.log('[IngestPage] pendingFiles:', pendingFiles.length, 'rawFileMap keys:', Array.from(rawFileMap.current.keys()));
    if (pendingFiles.length === 0) {
      addToast('没有待处理的文件', 'info');
      return;
    }

    setIsProcessing(true);
    const abortCtrl = new AbortController();
    abortRef.current = abortCtrl;

    try {
      // Step 1: Upload files to backend
      setGlobalProgress('上传文件...');
      for (const pf of pendingFiles) {
        updateFileStatus(pf.id, 'uploading', '上传中...');
      }

      const rawFiles = pendingFiles
        .map((pf) => {
          const f = rawFileMap.current.get(pf.id);
          console.log('[IngestPage] rawFileMap lookup:', pf.id, '->', f ? f.name : 'NOT FOUND');
          return f;
        })
        .filter(Boolean) as File[];

      console.log('[IngestPage] rawFiles count:', rawFiles.length, 'collection:', currentCollection);

      if (rawFiles.length === 0) {
        addToast('无法读取文件，请重新选择', 'error');
        setIsProcessing(false);
        return;
      }

      let uploaded: UploadedFile[];
      try {
        console.log('[IngestPage] calling uploadFiles...');
        uploaded = await uploadFiles(rawFiles, currentCollection);
        console.log('[IngestPage] upload success:', uploaded);
      } catch (err: unknown) {
        console.error('[IngestPage] upload failed:', err);
        const msg = err instanceof Error ? err.message : String(err);
        addToast(`上传失败: ${msg}`, 'error');
        for (const pf of pendingFiles) {
          updateFileStatus(pf.id, 'error', `上传失败: ${msg}`);
        }
        setIsProcessing(false);
        return;
      }

      const pathMap: Record<string, string> = {};
      for (const u of uploaded) {
        pathMap[u.filename] = u.path;
      }
      setFiles((prev) =>
        prev.map((f) => {
          if (pathMap[f.name]) {
            return { ...f, serverPath: pathMap[f.name], status: 'parsing' as FileStatus, message: '等待处理...' };
          }
          return f;
        }),
      );

      // 检测已入库重复（按 content_hash），先刷新列表
      const existingPapers = await loadPapers();
      const existingHashes = new Set(
        existingPapers.map((p) => p.content_hash).filter((h): h is string => Boolean(h)),
      );
      const duplicatePairs = uploaded
        .filter((u) => u.content_hash && existingHashes.has(u.content_hash))
        .map((u) => ({
          uploadedFile: u,
          existingPaper: existingPapers.find((p) => p.content_hash === u.content_hash)!,
        }))
        .filter((d) => d.existingPaper);
      if (duplicatePairs.length > 0) {
        const modalData = { uploaded, duplicatePairs };
        if (duplicateActionPreference) {
          setDuplicateModal(null);
          await runProcessAfterDuplicateChoice(duplicateActionPreference, modalData);
          return;
        }
        setIsProcessing(false);
        setDuplicateModal(modalData);
        return;
      }

      // Step 2: Process (SSE)
      const filePaths = uploaded.map((u) => u.path);
      const contentHashes = uploaded.reduce<Record<string, string>>((acc, u) => {
        if (u.content_hash) acc[u.path] = u.content_hash;
        return acc;
      }, {});
      setGlobalProgress('处理中...');
      for await (const evt of processFiles(filePaths, currentCollection, enrichment, abortCtrl.signal, contentHashes)) {
        console.log('[IngestPage] SSE event:', evt.event, evt.data);
        applyIngestEvent(evt);
      }
    } catch (err: unknown) {
      console.error('[IngestPage] process error:', err);
      if ((err as Error).name !== 'AbortError') {
        const msg = err instanceof Error ? err.message : String(err);
        addToast(`处理失败: ${msg}`, 'error');
        setGlobalProgress(`失败: ${msg}`);
      }
    } finally {
      setIsProcessing(false);
      setIsCancelling(false);
      abortRef.current = null;
    }
  };

  /** 用户选择 跳过/覆盖 后执行；modalData 可由调用方传入（自动应用偏好时） */
  const runProcessAfterDuplicateChoice = async (
    choice: 'skip' | 'overwrite',
    modalData?: { uploaded: UploadedFile[]; duplicatePairs: { uploadedFile: UploadedFile; existingPaper: PaperInfo }[] },
  ) => {
    const data = modalData ?? duplicateModal;
    if (!data) return;
    const { uploaded, duplicatePairs } = data;
    if (!modalData) setDuplicateModal(null);
    if (applyToAllSimilar && !modalData) setDuplicateActionPreference(choice);
    const dupPaths = new Set(duplicatePairs.map((d) => d.uploadedFile.path));
    const contentHashesAll = uploaded.reduce<Record<string, string>>((acc, u) => {
      if (u.content_hash) acc[u.path] = u.content_hash;
      return acc;
    }, {});

    if (choice === 'skip') {
      const toProcess = uploaded.filter((u) => !dupPaths.has(u.path));
      const toProcessPaths = toProcess.map((u) => u.path);
      setFiles((prev) =>
        prev.map((f) => {
          if (f.serverPath && dupPaths.has(f.serverPath)) return { ...f, status: 'skipped' as FileStatus, message: '已存在，已跳过' };
          return f;
        }),
      );
      if (toProcessPaths.length === 0) {
        addToast('全部为重复文件，未入库', 'info');
        return;
      }
      setGlobalProgress('处理中...');
      setIsProcessing(true);
      const abortCtrl = new AbortController();
      abortRef.current = abortCtrl;
      try {
        const contentHashes = toProcess.reduce<Record<string, string>>((acc, u) => {
          if (u.content_hash) acc[u.path] = u.content_hash;
          return acc;
        }, {});
        for await (const evt of processFiles(toProcessPaths, currentCollection, enrichment, abortCtrl.signal, contentHashes)) {
          applyIngestEvent(evt);
        }
      } catch (e) {
        if ((e as Error).name !== 'AbortError') addToast(`处理失败: ${(e as Error).message}`, 'error');
      } finally {
        setIsProcessing(false);
        setIsCancelling(false);
        abortRef.current = null;
      }
      return;
    }

    // choice === 'overwrite': 按已存在 paper_id 删除再全量 process
    setGlobalProgress('正在删除已存在记录...');
    setIsProcessing(true);
    const abortCtrl = new AbortController();
    abortRef.current = abortCtrl;
    try {
      for (const d of duplicatePairs) {
        await deletePaper(currentCollection, d.existingPaper.paper_id);
      }
      loadPapers();
      setGlobalProgress('处理中...');
      const filePaths = uploaded.map((u) => u.path);
      for await (const evt of processFiles(filePaths, currentCollection, enrichment, abortCtrl.signal, contentHashesAll)) {
        applyIngestEvent(evt);
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') addToast(`处理失败: ${(e as Error).message}`, 'error');
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

  // ---- Helpers to update file status ----

  const updateFileStatus = (id: string, status: FileStatus, message?: string) => {
    setFiles((prev) =>
      prev.map((f) => (f.id === id ? { ...f, status, message } : f)),
    );
  };

  const updateFileStatusByName = (name: string, status: FileStatus, message?: string) => {
    if (!name || !status) return;
    setFiles((prev) => {
      let found = false;
      const mapped = prev.map((f) => {
        const serverName = f.serverPath ? f.serverPath.split(/[/\\]/).pop() : '';
        // 优先匹配展示名；如果后端为重名文件加了后缀，则匹配 serverPath basename
        if (f.name === name || serverName === name) {
          found = true;
          if (f.status !== 'done' && f.status !== 'error' && f.status !== 'skipped') {
            return { ...f, status, message };
          }
        }
        return f;
      });
      if (!found) {
        mapped.push({
          id: `resume-${name}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          name,
          size: 0,
          status,
          message,
        });
      }
      return mapped;
    });
  };

  // ---- Create collection ----

  const handleCreateCollection = async () => {
    if (!newCollectionName.trim()) return;
    setShowCreateCollectionModal(false);
    addToast(`正在创建集合: ${newCollectionName}...`, 'info');
    try {
      await createCollection(newCollectionName.trim());
      addCollection(newCollectionName.trim());
      addToast(`集合 ${newCollectionName} 创建成功`, 'success');
      setNewCollectionName('');
      loadCollections();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`创建失败: ${msg}`, 'error');
    }
  };

  // ---- Delete collection ----
  const handleDeleteCollection = async (name: string) => {
    if (!window.confirm(`确定要删除集合「${name}」？此操作不可恢复，集合内所有数据将被永久删除。`)) {
      return;
    }
    addToast(`正在删除集合: ${name}...`, 'info');
    try {
      await deleteCollection(name);
      addToast(`集合 ${name} 已删除`, 'success');
      // If deleted the current selection, clear it
      if (currentCollection === name) {
        setCurrentCollection('');
      }
      loadCollections();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`删除失败: ${msg}`, 'error');
    }
  };

  // ---- Stats ----
  const pendingCount = files.filter((f) => f.status === 'pending').length;
  const doneCount = files.filter((f) => f.status === 'done').length;
  const skippedCount = files.filter((f) => f.status === 'skipped').length;
  const errorCount = files.filter((f) => f.status === 'error').length;
  const currentCollectionInfo = backendCollections.find((c) => c.name === currentCollection);
  const hasCollections = backendCollections.length > 0;
  const isConnected = !connectError && !collectionsLoading;
  const canUpload = hasCollections && isConnected;
  const tableProvider =
    llmProviders.find((p) => p.id === enrichment.llm_text_provider) || null;
  const figureProvider =
    llmProviders.find((p) => p.id === enrichment.llm_vision_provider) || null;
  const concurrencyOptions = [1, 2, 3, 4, 6, 8];

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
        ref={fileInputRef}
        id="ingest-file-input"
        name="ingest-file-input"
        type="file"
        accept=".pdf"
        multiple
        className="hidden"
        onChange={handleFileInputChange}
      />
      <input
        ref={folderInputRef}
        id="ingest-folder-input"
        name="ingest-folder-input"
        type="file"
        accept=".pdf"
        multiple
        // @ts-expect-error: webkitdirectory is non-standard but widely supported
        webkitdirectory=""
        className="hidden"
        onChange={handleFileInputChange}
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
            {hasCollections && (
              <>
                <div className="flex justify-between">
                  <span>当前集合:</span>
                  <span className="text-green-400">{currentCollection}</span>
                </div>
                {currentCollectionInfo && (
                  <div className="flex justify-between">
                    <span>文档数:</span>
                    <span className="text-white">
                      {currentCollectionInfo.count >= 0 ? currentCollectionInfo.count : '?'}
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
              <h3 className="text-lg font-bold text-gray-900">选择集合 (Collection)</h3>
              <p className="text-xs text-gray-500">将文档上传至指定的知识库分区</p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={loadCollections}
                disabled={collectionsLoading}
                className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                title="刷新集合列表"
              >
                <RefreshCw size={16} className={collectionsLoading ? 'animate-spin' : ''} />
              </button>
              <button
                onClick={() => setShowCreateCollectionModal(true)}
                className="flex items-center gap-2 px-4 py-2 bg-blue-50 text-blue-600 rounded-lg text-sm font-bold hover:bg-blue-100 transition-colors"
              >
                <Plus size={16} /> 新建集合
              </button>
            </div>
          </div>

          {hasCollections ? (
            <div className="space-y-2">
              {backendCollections.map((c) => (
                <div
                  key={c.name}
                  onClick={() => setCurrentCollection(c.name)}
                  className={`flex items-center justify-between p-3 rounded-xl border cursor-pointer transition-colors ${
                    currentCollection === c.name
                      ? 'border-blue-400 bg-blue-50'
                      : 'border-gray-200 bg-gray-50 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <Database size={16} className={currentCollection === c.name ? 'text-blue-500' : 'text-gray-400'} />
                    <div className="min-w-0">
                      <div className="font-medium text-sm text-gray-800 truncate">{c.name}</div>
                      <div className="text-xs text-gray-400">
                        {c.count >= 0 ? `${c.count} 条记录` : '加载中...'}
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteCollection(c.name);
                    }}
                    className="p-1.5 text-gray-300 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                    title={`删除集合 ${c.name}`}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-center">
              <p className="text-amber-700 text-sm font-medium mb-2">暂无集合</p>
              <p className="text-amber-600 text-xs mb-3">
                请先创建一个向量集合，然后再上传文档进行入库
              </p>
              <button
                onClick={() => setShowCreateCollectionModal(true)}
                className="px-4 py-2 bg-amber-600 text-white rounded-lg text-sm font-bold hover:bg-amber-700 transition-colors"
              >
                <Plus size={14} className="inline mr-1 -mt-0.5" /> 创建第一个集合
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 已入库文件列表 */}
      {hasCollections && currentCollection && (() => {
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
                  {currentCollection}
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
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-gray-400 border-b">
                      <th className="px-6 py-2.5 font-medium">文件名</th>
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
                        <td className="px-6 py-2.5 font-medium flex items-center gap-2">
                          <FileText size={14} className="text-gray-400 flex-shrink-0" />
                          <span className="truncate max-w-[300px]" title={p.filename || p.paper_id}>
                            {p.filename || p.paper_id}
                          </span>
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

      {/* 上传区域 */}
      <div
        className={`bg-white p-8 rounded-2xl border-2 border-dashed shadow-sm text-center transition-colors ${
          !canUpload
            ? 'border-gray-100 opacity-60 pointer-events-none'
            : isDragOver
              ? 'border-blue-400 bg-blue-50'
              : 'border-gray-200'
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          if (canUpload) setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={(e) => canUpload && handleDrop(e)}
      >
        <UploadCloud size={48} className="mx-auto text-blue-500 mb-4" />
        <h2 className="text-xl font-bold">上传文档</h2>
        <p className="text-gray-500 mt-2 mb-6">
          {canUpload
            ? '拖放 PDF 文件到此处，或点击下方按钮选择文件/文件夹'
            : '请先选择或创建一个集合'}
        </p>
        <div className="flex justify-center gap-3">
          <button
            onClick={handleSelectFiles}
            disabled={isProcessing || !canUpload}
            className="flex items-center gap-2 bg-gray-900 text-white px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-gray-800 disabled:opacity-50 transition-colors"
          >
            <FileText size={16} /> 选择文件
          </button>
          <button
            onClick={handleSelectFolder}
            disabled={isProcessing || !canUpload}
            className="flex items-center gap-2 bg-white text-gray-700 border border-gray-300 px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-gray-50 disabled:opacity-50 transition-colors"
          >
            <FolderOpen size={16} /> 选择文件夹
          </button>
        </div>

        {/* LLM 增强选项 */}
        <div className="mt-6 max-w-md mx-auto">
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            LLM 增强选项
          </div>
          <div className="flex justify-center items-stretch gap-3">
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
            <p className="text-[10px] text-gray-400 mt-2 text-center">
              未选择增强项，将仅执行基础解析（更快）
            </p>
          )}
        </div>
      </div>

      {/* 解析状态：表格/图片 enrich 进度（仅在有解析中文件且有待展示日志时显示） */}
      {isProcessing && Object.keys(enrichLogs).length > 0 && (
        <div className="mb-4 bg-amber-50/80 border border-amber-200 rounded-xl overflow-hidden">
          <div className="px-4 py-2 border-b border-amber-200 bg-amber-100/80 text-amber-800 text-sm font-medium">
            解析状态（表格/图片）
          </div>
          <div className="px-4 py-3 max-h-40 overflow-y-auto space-y-2">
            {Object.entries(enrichLogs).map(([file, lines]) => (
              <div key={file} className="text-xs">
                <div className="font-medium text-gray-700 mb-1 truncate" title={file}>
                  {file}
                </div>
                <ul className="text-gray-600 space-y-0.5 font-mono">
                  {lines.slice(-10).map((line, i) => (
                    <li key={i}>{line}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 文件列表 + 处理 */}
      {files.length > 0 && (
        <div className="bg-white border rounded-2xl overflow-hidden shadow-sm">
          <div className="px-6 py-4 border-b bg-gray-50 flex justify-between items-center">
            <div className="flex items-center gap-3">
              <span className="font-bold text-sm">
                文件列表 ({files.length} 个)
              </span>
              {pendingCount > 0 && (
                <span className="text-xs bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full">
                  {pendingCount} 待处理
                </span>
              )}
              {doneCount > 0 && (
                <span className="text-xs bg-green-50 text-green-600 px-2 py-0.5 rounded-full">
                  {doneCount} 完成
                </span>
              )}
              {errorCount > 0 && (
                <span className="text-xs bg-red-50 text-red-600 px-2 py-0.5 rounded-full">
                  {errorCount} 失败
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {errorCount > 0 && !isProcessing && (
                <button
                  onClick={handleRetryFailed}
                  className="text-xs text-orange-500 hover:text-orange-700 px-3 py-1 rounded-lg hover:bg-orange-50 font-medium"
                >
                  重试失败 ({errorCount})
                </button>
              )}
              {pendingCount > 0 && !isProcessing && (
                <button
                  onClick={handleClearPending}
                  className="text-xs text-gray-500 hover:text-gray-700 px-3 py-1 rounded-lg hover:bg-gray-100"
                >
                  删除待处理 ({pendingCount})
                </button>
              )}
              {errorCount > 0 && !isProcessing && (
                <button
                  onClick={handleClearFailed}
                  className="text-xs text-red-500 hover:text-red-700 px-3 py-1 rounded-lg hover:bg-red-50"
                >
                  删除失败 ({errorCount})
                </button>
              )}
              {(doneCount > 0 || skippedCount > 0) && !isProcessing && (
                <button
                  onClick={handleClearDone}
                  className="text-xs text-gray-400 hover:text-gray-600 px-3 py-1 rounded-lg hover:bg-gray-100"
                >
                  清除已完成 ({doneCount + skippedCount})
                </button>
              )}
              {isProcessing ? (
                <button
                  onClick={handleAbort}
                  disabled={isCancelling}
                  className="px-4 py-2 bg-red-50 text-red-600 rounded-lg text-sm font-bold hover:bg-red-100 transition-colors"
                >
                  {isCancelling ? '取消中...' : '取消处理'}
                </button>
              ) : (
                <button
                  onClick={handleStartProcess}
                  disabled={pendingCount === 0}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-bold hover:bg-blue-700 disabled:opacity-50 transition-colors"
                >
                  开始入库 ({pendingCount})
                </button>
              )}
            </div>
          </div>

          {/* Progress bar */}
          {globalProgress && (
            <div className="px-6 py-3 bg-blue-50 border-b text-sm text-blue-700 flex items-center gap-2">
              {isProcessing && <Loader2 size={14} className="animate-spin" />}
              {globalProgress}
            </div>
          )}

          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-400 border-b">
                <th className="px-6 py-3 font-medium">文件名</th>
                <th className="px-6 py-3 font-medium w-24">大小</th>
                <th className="px-6 py-3 font-medium w-32">状态</th>
                <th className="px-6 py-3 font-medium">信息</th>
                <th className="px-6 py-3 font-medium text-right w-16">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {files.map((item) => (
                <tr key={item.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-6 py-3 font-medium flex items-center gap-2">
                    <FileText size={16} className="text-gray-400 flex-shrink-0" />
                    <span className="truncate max-w-[300px]">{item.name}</span>
                  </td>
                  <td className="px-6 py-3 text-gray-500">
                    {formatSize(item.size)}
                  </td>
                  <td className="px-6 py-3">
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-[10px] font-bold uppercase ${STATUS_COLORS[item.status]}`}
                    >
                      {item.status === 'done' && <CheckCircle2 size={10} />}
                      {item.status === 'error' && <AlertCircle size={10} />}
                      {['uploading', 'parsing', 'chunking', 'embedding', 'indexing'].includes(item.status) && (
                        <Loader2 size={10} className="animate-spin" />
                      )}
                      {STATUS_LABELS[item.status]}
                    </span>
                  </td>
                  <td className="px-6 py-3 text-gray-500 text-xs truncate max-w-[200px]">
                    {item.message || '-'}
                  </td>
                  <td className="px-6 py-3 text-right">
                    {item.status === 'pending' && !isProcessing && (
                      <button
                        onClick={() => handleRemoveFile(item.id)}
                        className="text-gray-300 hover:text-red-500"
                      >
                        <Trash2 size={16} />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* 重复文件确认 Modal（按 content_hash 判定） */}
      <Modal
        open={duplicateModal !== null}
        onClose={() => setDuplicateModal(null)}
        title="以下文件已在本集合中（内容相同）"
        maxWidth="max-w-md"
      >
        {duplicateModal && (
          <>
            <p className="text-sm text-gray-600 mb-3">
              选择「跳过」不重复入库，选择「覆盖」将先删除该文件在本集合中的向量数据再重新入库。
            </p>
            <ul className="mb-4 max-h-48 overflow-y-auto rounded-lg border border-gray-200 bg-gray-50 p-2 text-sm">
              {duplicateModal.duplicatePairs.map((d) => (
                <li key={d.uploadedFile.path} className="py-1 truncate" title={d.uploadedFile.filename}>
                  {d.uploadedFile.filename}
                </li>
              ))}
            </ul>
            <label className="flex items-center gap-2 mb-4 text-sm text-gray-600">
              <input
                type="checkbox"
                checked={applyToAllSimilar}
                onChange={(e) => setApplyToAllSimilar(e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              对本次及之后所有重复项执行相同操作
            </label>
            {duplicateActionPreference && (
              <p className="text-xs text-gray-400 mb-2">
                当前偏好：{duplicateActionPreference === 'skip' ? '跳过已存在' : '覆盖并重新入库'}
                <button
                  type="button"
                  onClick={() => setDuplicateActionPreference(null)}
                  className="ml-2 text-blue-600 hover:underline"
                >
                  清除偏好
                </button>
              </p>
            )}
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setDuplicateModal(null)}
                className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg text-sm"
              >
                取消
              </button>
              <button
                onClick={() => runProcessAfterDuplicateChoice('skip')}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 text-sm"
              >
                跳过已存在
              </button>
              <button
                onClick={() => runProcessAfterDuplicateChoice('overwrite')}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
              >
                覆盖并重新入库
              </button>
            </div>
          </>
        )}
      </Modal>

      {/* 新建集合 Modal */}
      <Modal
        open={showCreateCollectionModal}
        onClose={() => setShowCreateCollectionModal(false)}
        title="新建向量集合"
        maxWidth="max-w-md"
      >
        <div className="space-y-4">
          <div>
            <label className="text-xs font-medium text-gray-500 uppercase">
              集合名称
            </label>
            <input
              id="new-collection-name"
              name="new-collection-name"
              type="text"
              value={newCollectionName}
              onChange={(e) => setNewCollectionName(e.target.value)}
              placeholder="自定义集合名称，例如 my_research_2026"
              className="w-full mt-1 border rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
              onKeyDown={(e) => e.key === 'Enter' && handleCreateCollection()}
            />
          </div>

          {/* 推荐模板 */}
          <div>
            <label className="text-xs font-medium text-gray-500 uppercase mb-2 block">
              快速选择模板
            </label>
            <div className="grid grid-cols-2 gap-2">
              {COLLECTION_TEMPLATES.map((t) => (
                <button
                  key={t.name}
                  onClick={() => setNewCollectionName(t.name)}
                  className={`text-left p-2.5 rounded-lg border text-xs transition-colors ${
                    newCollectionName === t.name
                      ? 'border-blue-400 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300 text-gray-600'
                  }`}
                >
                  <div className="font-medium">{t.name}</div>
                  <div className="text-gray-400 mt-0.5">{t.desc}</div>
                </button>
              ))}
            </div>
          </div>

          <p className="text-xs text-gray-400">
            使用 v2 schema (chunk_id 主键)，支持 upsert 去重。名称创建后不可修改。
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
            onClick={handleCreateCollection}
            disabled={!newCollectionName.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm"
          >
            创建
          </button>
        </div>
      </Modal>
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
