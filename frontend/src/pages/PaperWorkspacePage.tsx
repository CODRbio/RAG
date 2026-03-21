import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Archive,
  BookOpen,
  CheckCircle2,
  FileImage,
  FileWarning,
  FolderSymlink,
  Image,
  Loader2,
  MessageSquare,
  Search,
  Sparkles,
  Star,
  Tag,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { PdfDocumentView } from '../components/ui/PdfDocumentView';
import { useAcademicAssistantStore, useConfigStore, useResourceStore, useScholarStore, useToastStore } from '../stores';
import { buildPaperWorkspacePath } from '../utils/appRoutes';
import { fetchPdfAsBlobUrl, getLibraryPdfViewUrl, getPdfViewUrl, ingestScholarLibrary } from '../api/scholar';
import { transformMarkdownMediaUrl } from '../utils/mediaUrl';
import type {
  AcademicAssistantTaskState,
  AssistantScope,
  PaperLocator,
  ResourceAnnotation,
  ResourceNote,
  ResourceReadStatus,
  ResourceRef,
} from '../types';

type PaperWorkspaceTab = 'summary' | 'ask' | 'notes' | 'annotations' | 'media';

const WORKSPACE_TABS: PaperWorkspaceTab[] = ['summary', 'ask', 'notes', 'annotations', 'media'];

function proseClassName() {
  return 'prose prose-invert prose-sm max-w-none prose-headings:text-sky-300 prose-p:text-slate-300 prose-li:text-slate-300 prose-strong:text-sky-200 prose-a:text-sky-400 prose-a:no-underline hover:prose-a:underline prose-code:bg-slate-900/60 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sky-300 prose-code:border prose-code:border-slate-700/50 prose-pre:bg-slate-950/80 prose-pre:border prose-pre:border-slate-800';
}

function taskSummary(task: AcademicAssistantTaskState | undefined): string {
  if (!task) return '';
  if (task.error_message) return task.error_message;
  if (task.message) return task.message;
  const result = task.result as { summary_md?: string } | null;
  return result?.summary_md || '';
}

function resourceKey(ref: ResourceRef | null): string | null {
  return ref ? `${ref.resource_type}:${ref.resource_id}` : null;
}

function extractAnnotationFocus(annotation: ResourceAnnotation | null): { page: number; bbox?: number[] } | null {
  if (!annotation) return null;
  const locator = annotation.target_locator || {};
  const rawPage = locator.page ?? locator.page_num;
  const page = Number(rawPage);
  const bbox = Array.isArray(locator.bbox) ? (locator.bbox as number[]) : undefined;
  if (!Number.isFinite(page) && !bbox) return null;
  return {
    page: Number.isFinite(page) ? Math.max(1, page) : 1,
    bbox,
  };
}

export function PaperWorkspacePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { paperUid = '' } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const initialTab = (searchParams.get('tab') as PaperWorkspaceTab | null) || 'summary';
  const initialLibraryId = searchParams.get('libraryId')
    ? Number(searchParams.get('libraryId'))
    : Number.NaN;

  const currentCollection = useConfigStore((s) => s.currentCollection);
  const collectionInfos = useConfigStore((s) => s.collectionInfos);
  const addToast = useToastStore((s) => s.addToast);

  const libraries = useScholarStore((s) => s.libraries);
  const activeLibraryId = useScholarStore((s) => s.activeLibraryId);
  const libraryPapers = useScholarStore((s) => s.libraryPapers);
  const libraryLoading = useScholarStore((s) => s.libraryLoading);
  const loadLibraries = useScholarStore((s) => s.loadLibraries);
  const loadLibraryPapers = useScholarStore((s) => s.loadLibraryPapers);
  const setActiveLibrary = useScholarStore((s) => s.setActiveLibrary);
  const downloadLibraryPaperAndOpen = useScholarStore((s) => s.downloadLibraryPaperAndOpen);

  const summaries = useAcademicAssistantStore((s) => s.summaries);
  const answers = useAcademicAssistantStore((s) => s.answers);
  const assistantTasks = useAcademicAssistantStore((s) => s.tasks);
  const annotationCache = useAcademicAssistantStore((s) => s.annotations);
  const loadingKeys = useAcademicAssistantStore((s) => s.loadingKeys);
  const summarizePaperAction = useAcademicAssistantStore((s) => s.summarizePaper);
  const askPaperAction = useAcademicAssistantStore((s) => s.askPaper);
  const startMediaAnalysisTask = useAcademicAssistantStore((s) => s.startMediaAnalysisTask);
  const streamTask = useAcademicAssistantStore((s) => s.streamTask);
  const listAnnotationsAction = useAcademicAssistantStore((s) => s.listAnnotations);
  const saveAnnotationAction = useAcademicAssistantStore((s) => s.saveAnnotation);

  const resourceStates = useResourceStore((s) => s.states);
  const resourceTags = useResourceStore((s) => s.tags);
  const resourceNotes = useResourceStore((s) => s.notes);
  const loadState = useResourceStore((s) => s.loadState);
  const upsertState = useResourceStore((s) => s.upsertState);
  const loadTags = useResourceStore((s) => s.loadTags);
  const addTag = useResourceStore((s) => s.addTag);
  const removeTag = useResourceStore((s) => s.removeTag);
  const loadNotes = useResourceStore((s) => s.loadNotes);
  const createNote = useResourceStore((s) => s.createNote);
  const updateNote = useResourceStore((s) => s.updateNote);
  const removeNote = useResourceStore((s) => s.removeNote);

  const [activeTab, setActiveTab] = useState<PaperWorkspaceTab>(
    WORKSPACE_TABS.includes(initialTab) ? initialTab : 'summary',
  );
  const [listQuery, setListQuery] = useState('');
  const [question, setQuestion] = useState('');
  const [noteDraft, setNoteDraft] = useState('');
  const [tagDraft, setTagDraft] = useState('');
  const [editingNote, setEditingNote] = useState<ResourceNote | null>(null);
  const [annotationKind, setAnnotationKind] = useState('chunk');
  const [annotationTargetText, setAnnotationTargetText] = useState('');
  const [annotationDirective, setAnnotationDirective] = useState('');
  const [editingAnnotation, setEditingAnnotation] = useState<ResourceAnnotation | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState('');
  const [focusPage, setFocusPage] = useState(1);
  const [focusBbox, setFocusBbox] = useState<number[] | undefined>(undefined);
  const [ingestingLibrary, setIngestingLibrary] = useState(false);
  const [preparingPdf, setPreparingPdf] = useState(false);

  const currentCollectionInfo = useMemo(
    () => collectionInfos.find((item) => item.name === currentCollection),
    [collectionInfos, currentCollection],
  );

  const desiredLibraryId = useMemo(() => {
    if (Number.isFinite(initialLibraryId)) return initialLibraryId;
    return currentCollectionInfo?.associated_library_id ?? activeLibraryId ?? null;
  }, [activeLibraryId, currentCollectionInfo?.associated_library_id, initialLibraryId]);

  useEffect(() => {
    void loadLibraries().catch(() => {});
  }, [loadLibraries]);

  useEffect(() => {
    if (desiredLibraryId == null) return;
    if (activeLibraryId !== desiredLibraryId) {
      setActiveLibrary(desiredLibraryId);
      return;
    }
    void loadLibraryPapers(desiredLibraryId).catch(() => {});
  }, [activeLibraryId, desiredLibraryId, loadLibraryPapers, setActiveLibrary]);

  useEffect(() => {
    const next = new URLSearchParams(searchParams);
    if (next.get('tab') !== activeTab) {
      next.set('tab', activeTab);
      setSearchParams(next, { replace: true });
    }
  }, [activeTab, searchParams, setSearchParams]);

  useEffect(() => {
    const nextTab = (searchParams.get('tab') as PaperWorkspaceTab | null) || 'summary';
    if (WORKSPACE_TABS.includes(nextTab) && nextTab !== activeTab) {
      setActiveTab(nextTab);
    }
  }, [activeTab, searchParams]);

  useEffect(() => {
    setQuestion('');
    setEditingNote(null);
    setNoteDraft('');
    setEditingAnnotation(null);
    setAnnotationDirective('');
    setAnnotationTargetText('');
    setAnnotationKind('chunk');
    setFocusPage(1);
    setFocusBbox(undefined);
  }, [paperUid]);

  useEffect(() => {
    return () => {
      if (pdfUrl?.startsWith('blob:')) {
        URL.revokeObjectURL(pdfUrl);
      }
    };
  }, [pdfUrl]);

  const currentLibrary = libraries.find((item) => item.id === desiredLibraryId) || null;
  const filteredLibraryPapers = useMemo(() => {
    const q = listQuery.trim().toLowerCase();
    if (!q) return libraryPapers;
    return libraryPapers.filter((paper) => {
      const haystack = [paper.title, paper.paper_uid || '', paper.authors?.join(', ') || '']
        .join(' ')
        .toLowerCase();
      return haystack.includes(q);
    });
  }, [libraryPapers, listQuery]);

  const currentPaper = useMemo(
    () => libraryPapers.find((paper) => (paper.paper_uid || '').trim() === paperUid.trim()) || null,
    [libraryPapers, paperUid],
  );

  const locator: PaperLocator | null = currentPaper?.paper_uid ? { paper_uid: currentPaper.paper_uid } : null;
  const defaultScope: AssistantScope = useMemo(() => {
    if (currentCollection) {
      return { scope_type: 'collection', scope_key: currentCollection };
    }
    if (desiredLibraryId != null) {
      return { scope_type: 'library', scope_key: String(desiredLibraryId) };
    }
    return { scope_type: 'global', scope_key: 'global' };
  }, [currentCollection, desiredLibraryId]);

  const resourceRef: ResourceRef | null =
    currentPaper && desiredLibraryId != null
      ? { resource_type: 'scholar_library_paper', resource_id: String(currentPaper.id) }
      : null;
  const resourceStoreKey = resourceKey(resourceRef);
  const resourceState = resourceStoreKey ? resourceStates[resourceStoreKey] : undefined;
  const tags = resourceStoreKey ? resourceTags[resourceStoreKey] || [] : [];
  const notes = resourceStoreKey ? resourceNotes[resourceStoreKey] || [] : [];
  const annotations = currentPaper?.paper_uid ? annotationCache[currentPaper.paper_uid] || [] : [];
  const summaryKey = locator ? locator.paper_uid || `${locator.paper_id || ''}:${locator.collection || ''}` : '';
  const summary = summaryKey ? summaries[summaryKey] : undefined;
  const qaKey = locator ? `qa:${locator.paper_uid || `${locator.paper_id || ''}:${locator.collection || ''}`}:${question}` : '';
  const answer = qaKey ? answers[qaKey] : undefined;
  const mediaTasks = useMemo(
    () =>
      Object.values(assistantTasks)
        .filter((task) => task.task_type === 'media-analysis')
        .sort((left, right) => right.updated_at - left.updated_at),
    [assistantTasks],
  );
  const readiness = {
    hasPdf: Boolean(currentPaper?.downloaded_at && currentPaper.paper_id),
    canRunAssistant: Boolean(currentPaper?.collection_paper_id || currentPaper?.in_collection),
    canRunMedia: Boolean(currentPaper?.paper_uid),
  };

  useEffect(() => {
    if (!resourceRef) return;
    void loadState(resourceRef).catch(() => {});
    void loadTags(resourceRef).catch(() => {});
    void loadNotes(resourceRef).catch(() => {});
  }, [loadNotes, loadState, loadTags, resourceRef]);

  useEffect(() => {
    if (!currentPaper?.paper_uid) return;
    void listAnnotationsAction({
      paper_uid: currentPaper.paper_uid,
      resource_type: resourceRef?.resource_type,
      resource_id: resourceRef?.resource_id,
      status: 'active',
    }).catch(() => {});
  }, [currentPaper?.paper_uid, listAnnotationsAction, resourceRef?.resource_id, resourceRef?.resource_type]);

  useEffect(() => {
    if (!currentPaper || !readiness.hasPdf || !currentPaper.paper_id) {
      setPdfError('');
      setPdfUrl((prev) => {
        if (prev?.startsWith('blob:')) URL.revokeObjectURL(prev);
        return null;
      });
      return;
    }
    let cancelled = false;
    setPdfLoading(true);
    setPdfError('');
    const url =
      desiredLibraryId != null && desiredLibraryId >= 0
        ? getLibraryPdfViewUrl(desiredLibraryId, currentPaper.paper_id)
        : getPdfViewUrl(currentPaper.paper_id);
    const load = async () => {
      try {
        const nextUrl =
          desiredLibraryId != null && desiredLibraryId >= 0
            ? await fetchPdfAsBlobUrl(url).catch(() => url)
            : url;
        if (cancelled) {
          if (nextUrl.startsWith('blob:')) URL.revokeObjectURL(nextUrl);
          return;
        }
        setPdfUrl((prev) => {
          if (prev?.startsWith('blob:')) URL.revokeObjectURL(prev);
          return nextUrl;
        });
      } catch (error) {
        if (!cancelled) {
          setPdfError(error instanceof Error ? error.message : 'PDF 加载失败');
        }
      } finally {
        if (!cancelled) setPdfLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [currentPaper, desiredLibraryId, readiness.hasPdf]);

  const handleSelectPaper = (nextPaperUid: string) => {
    navigate(
      buildPaperWorkspacePath(nextPaperUid, {
        libraryId: desiredLibraryId,
        tab: activeTab,
      }),
    );
  };

  const handlePreparePdf = async () => {
    if (!currentPaper) return;
    setPreparingPdf(true);
    try {
      const taskId = await downloadLibraryPaperAndOpen(currentPaper);
      if (!taskId) {
        addToast(t('scholar.downloadFailed'), 'error');
      } else {
        addToast(t('scholar.downloadStarted'), 'success');
      }
    } finally {
      setPreparingPdf(false);
    }
  };

  const handleIngestLibrary = async () => {
    if (desiredLibraryId == null || desiredLibraryId < 0 || !currentCollection) return;
    setIngestingLibrary(true);
    try {
      const result = await ingestScholarLibrary(desiredLibraryId, {
        collection: currentCollection,
        skip_duplicate_doi: true,
        skip_unchanged: true,
        auto_download_missing: false,
        enrich_tables: false,
        enrich_figures: false,
      });
      await loadLibraryPapers(desiredLibraryId);
      addToast(
        t('paperWorkspace.ingestComplete', {
          defaultValue: 'Library ingest finished: {{ready}} PDF-ready, {{downloaded}} downloaded now.',
          ready: result.pdf_ready_count ?? 0,
          downloaded: result.downloaded_now ?? 0,
        }),
        'success',
      );
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('scholar.downloadFailed'), 'error');
    } finally {
      setIngestingLibrary(false);
    }
  };

  const handleRunSummary = async () => {
    if (!locator || !readiness.canRunAssistant) return;
    try {
      await summarizePaperAction({ locator, scope: defaultScope });
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.summaryFailed', 'Summary failed'), 'error');
    }
  };

  const handleAsk = async () => {
    if (!locator || !question.trim() || !readiness.canRunAssistant) return;
    try {
      await askPaperAction({
        locator,
        question: question.trim(),
        scope: defaultScope,
      });
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.qaFailed', 'Question answering failed'), 'error');
    }
  };

  const handleToggleState = async (field: 'favorite' | 'archived', next: boolean) => {
    if (!resourceRef) return;
    try {
      await upsertState({
        resource_type: resourceRef.resource_type,
        resource_id: resourceRef.resource_id,
        [field]: next,
      });
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.actionFailed', 'Operation failed'), 'error');
    }
  };

  const handleReadStatusChange = async (nextStatus: ResourceReadStatus) => {
    if (!resourceRef) return;
    try {
      await upsertState({
        resource_type: resourceRef.resource_type,
        resource_id: resourceRef.resource_id,
        read_status: nextStatus,
      });
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.actionFailed', 'Operation failed'), 'error');
    }
  };

  const handleAddTag = async () => {
    if (!resourceRef || !tagDraft.trim()) return;
    try {
      await addTag({
        resource_type: resourceRef.resource_type,
        resource_id: resourceRef.resource_id,
        tag: tagDraft.trim(),
      });
      setTagDraft('');
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.actionFailed', 'Operation failed'), 'error');
    }
  };

  const handleSaveNote = async () => {
    if (!resourceRef || !noteDraft.trim()) return;
    try {
      if (editingNote) {
        await updateNote(resourceRef, editingNote.id, { note_md: noteDraft.trim() });
      } else {
        await createNote({
          resource_type: resourceRef.resource_type,
          resource_id: resourceRef.resource_id,
          note_md: noteDraft.trim(),
        });
      }
      setEditingNote(null);
      setNoteDraft('');
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.actionFailed', 'Operation failed'), 'error');
    }
  };

  const handleSaveAnnotation = async () => {
    if (!currentPaper?.paper_uid || !annotationDirective.trim()) return;
    try {
      await saveAnnotationAction({
        annotation_id: editingAnnotation?.id,
        resource_type: resourceRef?.resource_type || 'paper',
        resource_id: resourceRef?.resource_id || currentPaper.paper_uid,
        paper_uid: currentPaper.paper_uid,
        target_kind: annotationKind,
        target_locator:
          editingAnnotation?.target_locator ||
          {
            page: focusPage,
            bbox: focusBbox || undefined,
          },
        target_text: annotationTargetText.trim(),
        directive: annotationDirective.trim(),
        status: 'active',
        collection: currentCollection || undefined,
      });
      setEditingAnnotation(null);
      setAnnotationTargetText('');
      setAnnotationDirective('');
      setAnnotationKind('chunk');
      await listAnnotationsAction({
        paper_uid: currentPaper.paper_uid,
        resource_type: resourceRef?.resource_type,
        resource_id: resourceRef?.resource_id,
        status: 'active',
      });
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.actionFailed', 'Operation failed'), 'error');
    }
  };

  const handleRunMedia = async () => {
    if (!currentPaper?.paper_uid) return;
    try {
      const task = await startMediaAnalysisTask({
        paper_uids: [currentPaper.paper_uid],
        scope: defaultScope,
        upsert_vectors: true,
      });
      await streamTask(task.task_id);
      addToast(t('paperWorkspace.mediaStarted', 'Media analysis started'), 'success');
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.mediaFailed', 'Media analysis failed'), 'error');
    }
  };

  const handleFocusAnnotation = (annotation: ResourceAnnotation) => {
    const focus = extractAnnotationFocus(annotation);
    if (focus) {
      setFocusPage(focus.page);
      setFocusBbox(focus.bbox);
    }
    setEditingAnnotation(annotation);
    setAnnotationKind(annotation.target_kind || 'chunk');
    setAnnotationTargetText(annotation.target_text || '');
    setAnnotationDirective(annotation.directive || '');
    setActiveTab('annotations');
  };

  const renderMarkdown = (content?: string) => {
    if (!content) return null;
    return (
      <div className={proseClassName()}>
        <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={transformMarkdownMediaUrl}>
          {content}
        </ReactMarkdown>
      </div>
    );
  };

  const renderTaskCard = (task: AcademicAssistantTaskState) => (
    <div key={task.task_id} className="rounded-xl border border-slate-700/60 bg-slate-950/50 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-sm font-medium text-slate-100">{task.task_type}</div>
        <span className="text-xs text-slate-400">{task.status}</span>
      </div>
      {taskSummary(task) && (
        <div className="mt-2 text-xs text-slate-300 whitespace-pre-wrap">{taskSummary(task)}</div>
      )}
    </div>
  );

  return (
    <div className="flex h-full min-h-0 bg-[radial-gradient(circle_at_top_left,rgba(14,165,233,0.12),transparent_34%),linear-gradient(180deg,rgba(15,23,42,0.94),rgba(2,6,23,0.98))]">
      <aside className="w-[320px] shrink-0 border-r border-slate-800/80 bg-slate-950/55 backdrop-blur-md">
        <div className="border-b border-slate-800/80 px-4 py-4">
          <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.2em] text-sky-300/80">
            <BookOpen size={14} />
            Paper Workspace
          </div>
          <h2 className="mt-2 text-lg font-semibold text-slate-100">
            {currentLibrary?.name || t('paperWorkspace.libraryFirst', 'Library First')}
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            {currentCollection
              ? t('paperWorkspace.collectionContext', {
                  defaultValue: 'Collection: {{collection}}',
                  collection: currentCollection,
                })
              : t('paperWorkspace.noCollection', 'No active collection')}
          </p>
          <div className="mt-3 rounded-xl border border-slate-800 bg-slate-900/70 px-3 py-2">
            <div className="text-[11px] text-slate-500">{t('paperWorkspace.currentTarget', 'Current target')}</div>
            <div className="mt-1 line-clamp-2 text-sm font-medium text-slate-100">
              {currentPaper?.title || paperUid}
            </div>
            <div className="mt-1 text-xs text-slate-500">{paperUid}</div>
          </div>
          <div className="mt-3 flex items-center gap-2">
            <button
              type="button"
              onClick={() => navigate('/scholar')}
              className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800"
            >
              {t('paperWorkspace.backToScholar', 'Back to Scholar')}
            </button>
          </div>
        </div>
        <div className="border-b border-slate-800/80 px-4 py-3">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              value={listQuery}
              onChange={(event) => setListQuery(event.target.value)}
              placeholder={t('paperWorkspace.filterPapers', 'Filter current library papers')}
              className="w-full rounded-xl border border-slate-700 bg-slate-900/80 py-2 pl-9 pr-3 text-sm text-slate-200 placeholder:text-slate-500"
            />
          </div>
        </div>
        <div className="h-[calc(100%-13rem)] overflow-y-auto px-3 py-3">
          {libraryLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 size={20} className="animate-spin text-sky-400" />
            </div>
          ) : filteredLibraryPapers.length === 0 ? (
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm text-slate-500">
              {t('paperWorkspace.noPapers', 'No papers found in the current workspace library.')}
            </div>
          ) : (
            <div className="space-y-2">
              {filteredLibraryPapers.map((paper) => {
                const isActive = (paper.paper_uid || '').trim() === paperUid.trim();
                return (
                  <button
                    key={`${paper.id}-${paper.paper_uid || paper.title}`}
                    type="button"
                    onClick={() => paper.paper_uid && handleSelectPaper(paper.paper_uid)}
                    disabled={!paper.paper_uid}
                    className={`w-full rounded-2xl border px-3 py-3 text-left transition ${
                      isActive
                        ? 'border-sky-500/50 bg-sky-500/12'
                        : 'border-slate-800 bg-slate-900/55 hover:border-slate-700 hover:bg-slate-900/80'
                    } disabled:cursor-not-allowed disabled:opacity-60`}
                  >
                    <div className="line-clamp-2 text-sm font-medium text-slate-100">{paper.title}</div>
                    <div className="mt-1 text-xs text-slate-500">
                      {paper.authors?.join(', ')} {paper.year != null ? `· ${paper.year}` : ''}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {paper.in_collection && (
                        <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] text-emerald-300">
                          ready
                        </span>
                      )}
                      {paper.downloaded_at && (
                        <span className="rounded-full bg-slate-700/70 px-2 py-0.5 text-[10px] text-slate-300">
                          pdf
                        </span>
                      )}
                      {!paper.paper_uid && (
                        <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] text-amber-300">
                          no uid
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </aside>

      <main className="flex min-w-0 flex-1">
        <section className="flex min-w-0 flex-1 flex-col border-r border-slate-800/70">
          <div className="border-b border-slate-800/70 px-6 py-4">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <FolderSymlink size={14} />
                  {currentLibrary?.name || t('paperWorkspace.workspace', 'Workspace')}
                </div>
                <h1 className="mt-2 line-clamp-2 text-2xl font-semibold text-slate-100">
                  {currentPaper?.title || t('paperWorkspace.paperNotFound', 'Paper not found')}
                </h1>
                <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-400">
                  <span>{paperUid}</span>
                  {currentPaper?.year != null && <span>{currentPaper.year}</span>}
                  {currentPaper?.venue && <span>{currentPaper.venue}</span>}
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span className={`rounded-full px-3 py-1 text-xs ${readiness.hasPdf ? 'bg-emerald-500/15 text-emerald-300' : 'bg-amber-500/15 text-amber-300'}`}>
                  {readiness.hasPdf ? t('paperWorkspace.pdfReady', 'PDF ready') : t('paperWorkspace.pdfMissing', 'PDF missing')}
                </span>
                <span className={`rounded-full px-3 py-1 text-xs ${readiness.canRunAssistant ? 'bg-sky-500/15 text-sky-200' : 'bg-slate-700/70 text-slate-300'}`}>
                  {readiness.canRunAssistant ? t('paperWorkspace.aiReady', 'AI ready') : t('paperWorkspace.needsIngest', 'Needs ingest')}
                </span>
              </div>
            </div>
          </div>

          {!currentPaper ? (
            <div className="flex flex-1 items-center justify-center p-8">
              <div className="max-w-md rounded-3xl border border-slate-800 bg-slate-950/70 p-6 text-center">
                <FileWarning className="mx-auto text-amber-300" size={28} />
                <h2 className="mt-4 text-lg font-semibold text-slate-100">{t('paperWorkspace.paperNotFound', 'Paper not found')}</h2>
                <p className="mt-2 text-sm text-slate-400">
                  {t('paperWorkspace.paperMissingHint', 'Open this workspace from Scholar after the paper is saved into a library with a stable paper_uid.')}
                </p>
              </div>
            </div>
          ) : !readiness.hasPdf ? (
            <div className="flex flex-1 items-center justify-center p-8">
              <div className="max-w-lg rounded-3xl border border-slate-800 bg-slate-950/70 p-6">
                <div className="flex items-center gap-3">
                  <FileWarning className="text-amber-300" size={24} />
                  <div>
                    <h2 className="text-lg font-semibold text-slate-100">{t('paperWorkspace.preparePaper', 'Prepare this paper')}</h2>
                    <p className="text-sm text-slate-400">
                      {t('paperWorkspace.preparePaperHint', 'The paper exists in your library, but the local PDF is not ready for reading yet.')}
                    </p>
                  </div>
                </div>
                <div className="mt-5 flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => void handlePreparePdf()}
                    disabled={preparingPdf}
                    className="inline-flex items-center gap-2 rounded-xl bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
                  >
                    {preparingPdf ? <Loader2 size={15} className="animate-spin" /> : <BookOpen size={15} />}
                    {t('paperWorkspace.downloadPdf', 'Download PDF')}
                  </button>
                  <button
                    type="button"
                    onClick={() => navigate('/scholar')}
                    className="rounded-xl border border-slate-700 px-4 py-2 text-sm text-slate-300 hover:bg-slate-800"
                  >
                    {t('paperWorkspace.openInScholar', 'Open in Scholar')}
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 min-h-0 p-5">
              {pdfLoading ? (
                <div className="flex h-full items-center justify-center">
                  <Loader2 size={22} className="animate-spin text-sky-400" />
                </div>
              ) : pdfError ? (
                <div className="rounded-3xl border border-red-500/20 bg-red-500/10 p-6 text-sm text-red-200">{pdfError}</div>
              ) : pdfUrl ? (
                <PdfDocumentView
                  key={`${pdfUrl}-${focusPage}-${focusBbox?.join(',') || ''}`}
                  pdfUrl={pdfUrl}
                  pageNumber={focusPage}
                  bbox={focusBbox}
                  title={currentPaper.title}
                  className="h-full"
                />
              ) : null}
            </div>
          )}
        </section>

        <aside className="w-[420px] shrink-0 bg-slate-950/55 backdrop-blur-md">
          <div className="border-b border-slate-800/80 px-4 pt-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <div className="text-[11px] uppercase tracking-[0.2em] text-sky-300/80">
                  {t('paperWorkspace.currentTarget', 'Current target')}
                </div>
                <div className="mt-1 text-sm font-medium text-slate-100">
                  {currentPaper?.paper_uid || paperUid}
                </div>
              </div>
              {resourceRef && (
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => void handleToggleState('favorite', !(resourceState?.favorite ?? false))}
                    className={`rounded-lg p-2 ${resourceState?.favorite ? 'text-amber-300' : 'text-slate-400 hover:text-amber-300'}`}
                    title={t('academicAssistant.favorite', 'Favorite')}
                  >
                    <Star size={15} />
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleToggleState('archived', !(resourceState?.archived ?? false))}
                    className={`rounded-lg p-2 ${resourceState?.archived ? 'text-slate-100' : 'text-slate-400 hover:text-slate-200'}`}
                    title={t('academicAssistant.archive', 'Archive')}
                  >
                    <Archive size={15} />
                  </button>
                </div>
              )}
            </div>
            <div className="flex flex-wrap gap-2 pb-4">
              {WORKSPACE_TABS.map((tab) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setActiveTab(tab)}
                  className={`rounded-full px-3 py-1.5 text-xs ${
                    activeTab === tab
                      ? 'bg-sky-500/20 text-sky-200'
                      : 'bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>

          <div className="h-[calc(100%-8.5rem)] overflow-y-auto px-4 py-4">
            {!readiness.canRunAssistant && activeTab !== 'notes' && activeTab !== 'annotations' && (
              <div className="mb-4 rounded-2xl border border-amber-500/25 bg-amber-500/10 p-4 text-sm text-amber-100">
                <div className="flex items-start gap-3">
                  <FileWarning size={18} className="mt-0.5 shrink-0" />
                  <div>
                    <div className="font-medium">{t('paperWorkspace.needsIngest', 'Needs ingest')}</div>
                    <div className="mt-1 text-xs text-amber-200/80">
                      {t('paperWorkspace.needsIngestHint', 'Summary and QA require this paper to be ingested into the active collection.')}
                    </div>
                    <div className="mt-3 flex gap-2">
                      <button
                        type="button"
                        onClick={() => void handleIngestLibrary()}
                        disabled={ingestingLibrary || desiredLibraryId == null || desiredLibraryId < 0 || !currentCollection}
                        className="inline-flex items-center gap-2 rounded-lg bg-amber-500/20 px-3 py-1.5 text-xs text-amber-100 disabled:opacity-50"
                      >
                        {ingestingLibrary ? <Loader2 size={12} className="animate-spin" /> : <FolderSymlink size={12} />}
                        {t('paperWorkspace.ingestBoundLibrary', 'Ingest bound library')}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'summary' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium text-slate-100">{t('academicAssistant.generateSummary', 'Summary')}</div>
                  <button
                    type="button"
                    onClick={() => void handleRunSummary()}
                    disabled={!readiness.canRunAssistant || loadingKeys[`summary:${summaryKey}`]}
                    className="inline-flex items-center gap-2 rounded-xl bg-sky-600 px-3 py-1.5 text-xs text-white hover:bg-sky-500 disabled:opacity-50"
                  >
                    {loadingKeys[`summary:${summaryKey}`] ? <Loader2 size={12} className="animate-spin" /> : <Sparkles size={12} />}
                    {t('paperWorkspace.runSummary', 'Run summary')}
                  </button>
                </div>
                <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                  {summary?.summary_md ? renderMarkdown(summary.summary_md) : (
                    <p className="text-sm text-slate-500">{t('academicAssistant.noSummaryYet', 'Run summary to generate a paper briefing with citations.')}</p>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'ask' && (
              <div className="space-y-4">
                <div className="text-sm font-medium text-slate-100">{t('academicAssistant.askQuestion', 'Targeted QA')}</div>
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder={t('academicAssistant.questionPlaceholder', 'Ask about method, result, limitation, or figures...')}
                  className="h-28 w-full rounded-2xl border border-slate-700 bg-slate-900/80 px-3 py-3 text-sm text-slate-200"
                />
                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={() => void handleAsk()}
                    disabled={!readiness.canRunAssistant || !question.trim() || loadingKeys[qaKey]}
                    className="inline-flex items-center gap-2 rounded-xl bg-sky-600 px-4 py-2 text-sm text-white hover:bg-sky-500 disabled:opacity-50"
                  >
                    {loadingKeys[qaKey] ? <Loader2 size={14} className="animate-spin" /> : <MessageSquare size={14} />}
                    {t('academicAssistant.runQa', 'Ask')}
                  </button>
                </div>
                <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                  {answer?.answer_md ? renderMarkdown(answer.answer_md) : (
                    <p className="text-sm text-slate-500">{t('academicAssistant.noAnswerYet', 'Ask a question to retrieve paper-grounded evidence.')}</p>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'notes' && (
              <div className="space-y-4">
                <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div className="text-sm font-medium text-slate-100">{t('academicAssistant.resourceState', 'Resource State')}</div>
                    <select
                      value={resourceState?.read_status || 'unread'}
                      onChange={(event) => void handleReadStatusChange(event.target.value as ResourceReadStatus)}
                      className="rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200"
                    >
                      <option value="unread">{t('academicAssistant.readStatusUnread', 'Unread')}</option>
                      <option value="reading">{t('academicAssistant.readStatusReading', 'Reading')}</option>
                      <option value="read">{t('academicAssistant.readStatusRead', 'Read')}</option>
                    </select>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => void handleToggleState('favorite', !(resourceState?.favorite ?? false))}
                      className={`inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs ${
                        resourceState?.favorite ? 'bg-amber-500/20 text-amber-200' : 'bg-slate-800 text-slate-300'
                      }`}
                    >
                      <Star size={13} />
                      {t('academicAssistant.favorite', 'Favorite')}
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleToggleState('archived', !(resourceState?.archived ?? false))}
                      className={`inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs ${
                        resourceState?.archived ? 'bg-slate-700 text-slate-100' : 'bg-slate-800 text-slate-300'
                      }`}
                    >
                      <Archive size={13} />
                      {t('academicAssistant.archive', 'Archive')}
                    </button>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                  <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
                    <Tag size={14} />
                    {t('academicAssistant.tags', 'Tags')}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {tags.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => resourceRef && void removeTag({ resource_type: resourceRef.resource_type, resource_id: resourceRef.resource_id, tag: item.tag })}
                        className="rounded-full border border-slate-700 bg-slate-900 px-2 py-0.5 text-xs text-slate-300 hover:border-red-400 hover:text-red-300"
                      >
                        #{item.tag}
                      </button>
                    ))}
                  </div>
                  <div className="mt-3 flex gap-2">
                    <input
                      value={tagDraft}
                      onChange={(event) => setTagDraft(event.target.value)}
                      placeholder={t('academicAssistant.addTagPlaceholder', 'Add a tag')}
                      className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200"
                    />
                    <button
                      type="button"
                      onClick={() => void handleAddTag()}
                      className="rounded-lg bg-sky-600 px-3 py-2 text-xs text-white hover:bg-sky-500"
                    >
                      {t('common.save')}
                    </button>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                  <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
                    <MessageSquare size={14} />
                    {t('academicAssistant.notes', 'Notes')}
                  </div>
                  <div className="space-y-2">
                    {notes.map((item) => (
                      <div key={item.id} className="rounded-xl border border-slate-800 bg-slate-950/70 p-3 text-sm text-slate-300">
                        <div className="whitespace-pre-wrap">{item.note_md}</div>
                        <div className="mt-3 flex items-center gap-3 text-xs">
                          <button
                            type="button"
                            onClick={() => {
                              setEditingNote(item);
                              setNoteDraft(item.note_md);
                            }}
                            className="text-sky-300"
                          >
                            {t('academicAssistant.edit', 'Edit')}
                          </button>
                          <button
                            type="button"
                            onClick={() => resourceRef && void removeNote(resourceRef, item.id)}
                            className="text-red-300"
                          >
                            {t('common.delete')}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                  <textarea
                    value={noteDraft}
                    onChange={(event) => setNoteDraft(event.target.value)}
                    placeholder={t('academicAssistant.notePlaceholder', 'Write a resource note')}
                    className="mt-3 h-28 w-full rounded-2xl border border-slate-700 bg-slate-900 px-3 py-3 text-sm text-slate-200"
                  />
                  <div className="mt-3 flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => void handleSaveNote()}
                      className="rounded-xl bg-sky-600 px-3 py-2 text-xs text-white hover:bg-sky-500"
                    >
                      {editingNote ? t('academicAssistant.updateNote', 'Update note') : t('academicAssistant.addNote', 'Add note')}
                    </button>
                    {editingNote && (
                      <button
                        type="button"
                        onClick={() => {
                          setEditingNote(null);
                          setNoteDraft('');
                        }}
                        className="rounded-xl border border-slate-700 px-3 py-2 text-xs text-slate-300"
                      >
                        {t('common.cancel')}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'annotations' && (
              <div className="space-y-4">
                <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                  <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
                    <CheckCircle2 size={14} />
                    {t('academicAssistant.annotations', 'Annotations')}
                  </div>
                  <div className="space-y-2">
                    {annotations.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => handleFocusAnnotation(item)}
                        className="block w-full rounded-xl border border-slate-800 bg-slate-950/70 p-3 text-left text-sm text-slate-300 hover:border-slate-700"
                      >
                        <div className="font-medium text-slate-100">{item.target_kind}</div>
                        {item.target_text && <div className="mt-1 text-xs text-slate-500">{item.target_text}</div>}
                        <div className="mt-2 line-clamp-2">{item.directive}</div>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                  <div className="mb-3 text-sm font-medium text-slate-100">{t('paperWorkspace.addAnchoredNote', 'Anchored note')}</div>
                  <div className="grid grid-cols-2 gap-2">
                    <select
                      value={annotationKind}
                      onChange={(event) => setAnnotationKind(event.target.value)}
                      className="rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200"
                    >
                      <option value="chunk">chunk</option>
                      <option value="figure">figure</option>
                      <option value="page_region">page_region</option>
                      <option value="canvas_section">canvas_section</option>
                    </select>
                    <input
                      value={annotationTargetText}
                      onChange={(event) => setAnnotationTargetText(event.target.value)}
                      placeholder={t('academicAssistant.annotationTarget', 'Target text')}
                      className="rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200"
                    />
                  </div>
                  <textarea
                    value={annotationDirective}
                    onChange={(event) => setAnnotationDirective(event.target.value)}
                    placeholder={t('academicAssistant.annotationDirective', 'Add your anchored note or directive')}
                    className="mt-3 h-24 w-full rounded-2xl border border-slate-700 bg-slate-900 px-3 py-3 text-sm text-slate-200"
                  />
                  <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
                    <FileImage size={12} />
                    {t('paperWorkspace.annotationFocus', {
                      defaultValue: 'Current focus: page {{page}}',
                      page: focusPage,
                    })}
                  </div>
                  <div className="mt-3 flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => void handleSaveAnnotation()}
                      className="rounded-xl bg-sky-600 px-3 py-2 text-xs text-white hover:bg-sky-500"
                    >
                      {editingAnnotation ? t('academicAssistant.updateAnnotation', 'Update annotation') : t('academicAssistant.addAnnotation', 'Add annotation')}
                    </button>
                    {editingAnnotation && (
                      <button
                        type="button"
                        onClick={() => {
                          setEditingAnnotation(null);
                          setAnnotationKind('chunk');
                          setAnnotationTargetText('');
                          setAnnotationDirective('');
                        }}
                        className="rounded-xl border border-slate-700 px-3 py-2 text-xs text-slate-300"
                      >
                        {t('common.cancel')}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'media' && (
              <div className="space-y-4">
                <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium text-slate-100">{t('academicAssistant.mediaAnalysis', 'Media analysis')}</div>
                      <div className="mt-1 text-xs text-slate-500">
                        {t('paperWorkspace.mediaHint', 'Backfill image interpretation and vectorize figure evidence on demand.')}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => void handleRunMedia()}
                      disabled={!readiness.canRunMedia}
                      className="inline-flex items-center gap-2 rounded-xl bg-sky-600 px-3 py-1.5 text-xs text-white hover:bg-sky-500 disabled:opacity-50"
                    >
                      <Image size={12} />
                      {t('paperWorkspace.runMedia', 'Run media')}
                    </button>
                  </div>
                </div>
                <div className="space-y-3">
                  {mediaTasks.length > 0 ? mediaTasks.map(renderTaskCard) : (
                    <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4 text-sm text-slate-500">
                      {t('paperWorkspace.noMediaTasks', 'No media-analysis task has been run for this workspace yet.')}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}
