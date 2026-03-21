import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Archive,
  Image,
  Loader2,
  MessageSquare,
  Search,
  Sparkles,
  Star,
  Tag,
  X,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAcademicAssistantStore, useResourceStore, useToastStore } from '../../stores';
import { transformMarkdownMediaUrl } from '../../utils/mediaUrl';
import type {
  AcademicAssistantDiscoveryResult,
  AcademicAssistantTaskState,
  AssistantScope,
  DiscoveryMode,
  PaperLocator,
  ResourceAnnotation,
  ResourceNote,
  ResourceReadStatus,
  ResourceRef,
} from '../../types';

export interface ScholarAssistantTarget {
  title: string;
  subtitle?: string | null;
  paperUid?: string | null;
  paperId?: string | null;
  collection?: string | null;
  resourceRef?: ResourceRef | null;
  canRunAssistant: boolean;
}

interface ScholarAssistantPanelProps {
  open: boolean;
  onClose: () => void;
  target: ScholarAssistantTarget | null;
  multiTargets: ScholarAssistantTarget[];
  defaultScope: AssistantScope;
}

function buildLocator(target: ScholarAssistantTarget | null): PaperLocator | null {
  if (!target) return null;
  if (target.paperUid) return { paper_uid: target.paperUid };
  if (target.paperId) {
    return {
      paper_id: target.paperId,
      collection: target.collection || undefined,
    };
  }
  return null;
}

function qaCacheKey(locator: PaperLocator | null, question: string): string | null {
  if (!locator) return null;
  const base = locator.paper_uid || `${locator.paper_id || ''}:${locator.collection || ''}`;
  return `qa:${base}:${question}`;
}

function compareCacheKey(targets: ScholarAssistantTarget[]): string {
  return targets
    .map((item) => item.paperUid || '')
    .filter(Boolean)
    .sort()
    .join('|');
}

function proseClassName() {
  return 'prose prose-invert prose-sm max-w-none prose-headings:text-sky-300 prose-p:text-slate-300 prose-li:text-slate-300 prose-strong:text-sky-200 prose-a:text-sky-400 prose-a:no-underline hover:prose-a:underline prose-code:bg-slate-900/60 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sky-300 prose-code:border prose-code:border-slate-700/50 prose-pre:bg-slate-950/80 prose-pre:border prose-pre:border-slate-800';
}

function TaskResultCard({ task }: { task: AcademicAssistantTaskState | undefined }) {
  if (!task) return null;
  const result = task.result as AcademicAssistantDiscoveryResult | undefined;
  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-900/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-sm text-slate-200">
          {task.task_type}
          {task.mode ? ` · ${task.mode}` : ''}
        </div>
        <span
          className={`text-xs ${
            task.status === 'completed'
              ? 'text-emerald-300'
              : task.status === 'error'
                ? 'text-red-300'
                : task.status === 'cancelled'
                  ? 'text-amber-300'
                  : 'text-sky-300'
          }`}
        >
          {task.status}
        </span>
      </div>
      {task.message && <p className="mt-1 text-xs text-slate-400">{task.message}</p>}
      {task.error_message && <p className="mt-2 text-xs text-red-300">{task.error_message}</p>}
      {result?.summary_md && (
        <div className={`mt-3 ${proseClassName()}`}>
          <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={transformMarkdownMediaUrl}>
            {result.summary_md}
          </ReactMarkdown>
        </div>
      )}
      {Array.isArray(result?.items) && result.items.length > 0 && (
        <div className="mt-3 space-y-2">
          {result.items.slice(0, 5).map((item, idx) => (
            <div key={`${task.task_id}-${idx}`} className="rounded-lg border border-slate-800 bg-slate-950/50 p-2 text-xs text-slate-300">
              <div className="font-medium text-slate-100">{String(item.title || item.paper_uid || item.author || item.institution || `Item ${idx + 1}`)}</div>
              {'reason' in item && Boolean(item.reason) && <div className="mt-1 text-slate-400">{String(item.reason)}</div>}
              {'year' in item && Boolean(item.year) && <div className="mt-1 text-slate-500">{String(item.year)}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ScholarAssistantPanel({
  open,
  onClose,
  target,
  multiTargets,
  defaultScope,
}: ScholarAssistantPanelProps) {
  const { t } = useTranslation();
  const addToast = useToastStore((s) => s.addToast);
  const summaries = useAcademicAssistantStore((s) => s.summaries);
  const answers = useAcademicAssistantStore((s) => s.answers);
  const comparisons = useAcademicAssistantStore((s) => s.comparisons);
  const assistantTasks = useAcademicAssistantStore((s) => s.tasks);
  const annotationCache = useAcademicAssistantStore((s) => s.annotations);
  const assistantLoading = useAcademicAssistantStore((s) => s.loadingKeys);
  const summarizePaperAction = useAcademicAssistantStore((s) => s.summarizePaper);
  const askPaperAction = useAcademicAssistantStore((s) => s.askPaper);
  const comparePapersAction = useAcademicAssistantStore((s) => s.comparePapers);
  const startDiscoveryTask = useAcademicAssistantStore((s) => s.startDiscoveryTask);
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

  const [question, setQuestion] = useState('');
  const [tagDraft, setTagDraft] = useState('');
  const [noteDraft, setNoteDraft] = useState('');
  const [editingNote, setEditingNote] = useState<ResourceNote | null>(null);
  const [annotationTargetText, setAnnotationTargetText] = useState('');
  const [annotationDirective, setAnnotationDirective] = useState('');
  const [annotationKind, setAnnotationKind] = useState('chunk');
  const [editingAnnotation, setEditingAnnotation] = useState<ResourceAnnotation | null>(null);
  const [panelMode, setPanelMode] = useState<'single' | 'multi'>('single');
  const [trackedTaskIds, setTrackedTaskIds] = useState<string[]>([]);
  const taskControllersRef = useRef<Map<string, AbortController>>(new Map());

  const locator = useMemo(() => buildLocator(target), [target]);
  const qaKey = useMemo(() => qaCacheKey(locator, question), [locator, question]);
  const summary = locator ? summaries[locator.paper_uid || `${locator.paper_id || ''}:${locator.collection || ''}`] : undefined;
  const answer = qaKey ? answers[qaKey] : undefined;
  const compareKeyValue = useMemo(() => compareCacheKey(multiTargets), [multiTargets]);
  const compareResult = compareKeyValue ? comparisons[compareKeyValue] : undefined;
  const selectedPaperUids = useMemo(
    () => multiTargets.map((item) => item.paperUid || '').filter(Boolean),
    [multiTargets],
  );
  const resourceRef = target?.resourceRef || null;
  const resourceState = resourceRef ? resourceStates[`${resourceRef.resource_type}:${resourceRef.resource_id}`] : undefined;
  const tags = resourceRef ? resourceTags[`${resourceRef.resource_type}:${resourceRef.resource_id}`] || [] : [];
  const notes = resourceRef ? resourceNotes[`${resourceRef.resource_type}:${resourceRef.resource_id}`] || [] : [];
  const annotations = useMemo(() => {
    if (!target?.paperUid) return [];
    return annotationCache[target.paperUid] || [];
  }, [annotationCache, target?.paperUid]);

  useEffect(() => {
    if (!open) return;
    if (target) setPanelMode('single');
    else if (multiTargets.length > 0) setPanelMode('multi');
  }, [open, target, multiTargets.length]);

  useEffect(() => {
    if (!open || !resourceRef) return;
    void loadState(resourceRef).catch(() => {});
    void loadTags(resourceRef).catch(() => {});
    void loadNotes(resourceRef).catch(() => {});
  }, [open, resourceRef, loadState, loadTags, loadNotes]);

  useEffect(() => {
    if (!open || !target?.paperUid) return;
    void listAnnotationsAction({
      paper_uid: target.paperUid,
      resource_type: target.resourceRef?.resource_type,
      resource_id: target.resourceRef?.resource_id,
      status: 'active',
    }).catch(() => {});
  }, [open, target?.paperUid, target?.resourceRef?.resource_id, target?.resourceRef?.resource_type, listAnnotationsAction]);

  useEffect(() => () => {
    taskControllersRef.current.forEach((controller) => controller.abort());
    taskControllersRef.current.clear();
  }, []);

  const runTaskStream = async (taskId: string) => {
    const controller = new AbortController();
    taskControllersRef.current.set(taskId, controller);
    try {
      await streamTask(taskId, controller.signal);
    } finally {
      taskControllersRef.current.delete(taskId);
    }
  };

  const handleToggleState = async (field: 'favorite' | 'archived', value: boolean) => {
    if (!resourceRef) return;
    try {
      await upsertState({
        resource_type: resourceRef.resource_type,
        resource_id: resourceRef.resource_id,
        [field]: value,
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
    if (!target?.paperUid || !annotationDirective.trim()) return;
    try {
      await saveAnnotationAction({
        annotation_id: editingAnnotation?.id,
        resource_type: target.resourceRef?.resource_type || 'paper',
        resource_id: target.resourceRef?.resource_id || target.paperUid,
        paper_uid: target.paperUid,
        target_kind: annotationKind,
        target_locator: editingAnnotation?.target_locator || {},
        target_text: annotationTargetText.trim(),
        directive: annotationDirective.trim(),
        status: 'active',
        collection: target.collection || undefined,
      });
      setEditingAnnotation(null);
      setAnnotationTargetText('');
      setAnnotationDirective('');
      setAnnotationKind('chunk');
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.actionFailed', 'Operation failed'), 'error');
    }
  };

  const handleRunSummary = async () => {
    if (!locator || !target?.canRunAssistant) return;
    try {
      await summarizePaperAction({
        locator,
        scope: defaultScope,
      });
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.summaryFailed', 'Summary failed'), 'error');
    }
  };

  const handleAskQuestion = async () => {
    if (!locator || !target?.canRunAssistant || !question.trim()) return;
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

  const handleRunCompare = async () => {
    if (selectedPaperUids.length < 2) return;
    try {
      await comparePapersAction({
        paper_uids: selectedPaperUids,
        scope: defaultScope,
      });
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.compareFailed', 'Compare failed'), 'error');
    }
  };

  const handleStartDiscovery = async (mode: DiscoveryMode) => {
    try {
      const task = await startDiscoveryTask({
        mode,
        paper_uids: selectedPaperUids,
        scope: defaultScope,
      });
      setTrackedTaskIds((prev) => [...new Set([...prev, task.task_id])]);
      void runTaskStream(task.task_id).catch((error) => {
        addToast(error instanceof Error ? error.message : t('academicAssistant.discoveryFailed', 'Discovery failed'), 'error');
      });
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.discoveryFailed', 'Discovery failed'), 'error');
    }
  };

  const handleStartMediaAnalysis = async () => {
    if (!target?.paperUid) return;
    try {
      const task = await startMediaAnalysisTask({
        paper_uids: [target.paperUid],
        scope: defaultScope,
        upsert_vectors: true,
      });
      setTrackedTaskIds((prev) => [...new Set([...prev, task.task_id])]);
      void runTaskStream(task.task_id).catch((error) => {
        addToast(error instanceof Error ? error.message : t('academicAssistant.mediaFailed', 'Media analysis failed'), 'error');
      });
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.mediaFailed', 'Media analysis failed'), 'error');
    }
  };

  const renderMarkdown = (content: string | undefined) => {
    if (!content) return null;
    return (
      <div className={proseClassName()}>
        <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={transformMarkdownMediaUrl}>
          {content}
        </ReactMarkdown>
      </div>
    );
  };

  if (!open) return null;

  return (
    <aside className="w-[380px] max-w-[40vw] min-w-[320px] flex-shrink-0 rounded-xl border border-slate-700/60 bg-slate-900/60 backdrop-blur-sm overflow-hidden flex flex-col">
      <div className="flex items-start justify-between gap-3 border-b border-slate-700/60 px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Sparkles size={16} className="text-sky-400" />
            <h3 className="text-sm font-semibold text-slate-100">
              {panelMode === 'single'
                ? t('academicAssistant.panelTitle', 'Academic Assistant')
                : t('academicAssistant.multiPanelTitle', 'Compare & Discover')}
            </h3>
          </div>
          <p className="mt-1 line-clamp-2 text-xs text-slate-400">
            {panelMode === 'single'
              ? target?.title || t('academicAssistant.noTarget', 'No paper selected')
              : multiTargets.map((item) => item.title).join(' · ')}
          </p>
          {panelMode === 'single' && target?.subtitle && (
            <p className="mt-1 text-[11px] text-slate-500">{target.subtitle}</p>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
        >
          <X size={16} />
        </button>
      </div>

      <div className="border-b border-slate-800 px-4 py-2">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setPanelMode('single')}
            disabled={!target}
            className={`rounded-full px-3 py-1 text-xs ${
              panelMode === 'single'
                ? 'bg-sky-500/20 text-sky-200'
                : 'bg-slate-800 text-slate-400 disabled:opacity-50'
            }`}
          >
            {t('academicAssistant.singleTab', 'Single')}
          </button>
          <button
            type="button"
            onClick={() => setPanelMode('multi')}
            disabled={multiTargets.length === 0}
            className={`rounded-full px-3 py-1 text-xs ${
              panelMode === 'multi'
                ? 'bg-sky-500/20 text-sky-200'
                : 'bg-slate-800 text-slate-400 disabled:opacity-50'
            }`}
          >
            {t('academicAssistant.multiTab', 'Multi')}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {panelMode === 'single' ? (
          target ? (
            <>
              {!target.canRunAssistant && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200">
                  {t('academicAssistant.requiresIngestedPaper', 'This paper must be ingested into the current collection before summary, QA, or media analysis can run.')}
                </div>
              )}

              {resourceRef && (
                <section className="rounded-xl border border-slate-700/60 bg-slate-950/40 p-3">
                  <div className="mb-3 flex items-center justify-between">
                    <h4 className="text-sm font-medium text-slate-100">{t('academicAssistant.resourceState', 'Resource State')}</h4>
                    <select
                      value={resourceState?.read_status || 'unread'}
                      onChange={(e) => handleReadStatusChange(e.target.value as ResourceReadStatus)}
                      className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200"
                    >
                      <option value="unread">{t('academicAssistant.readStatusUnread', 'Unread')}</option>
                      <option value="reading">{t('academicAssistant.readStatusReading', 'Reading')}</option>
                      <option value="read">{t('academicAssistant.readStatusRead', 'Read')}</option>
                    </select>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => handleToggleState('favorite', !(resourceState?.favorite ?? false))}
                      className={`inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs ${
                        resourceState?.favorite
                          ? 'bg-amber-500/20 text-amber-200'
                          : 'bg-slate-800 text-slate-300 hover:text-slate-100'
                      }`}
                    >
                      <Star size={13} />
                      {t('academicAssistant.favorite', 'Favorite')}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleToggleState('archived', !(resourceState?.archived ?? false))}
                      className={`inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs ${
                        resourceState?.archived
                          ? 'bg-slate-600/40 text-slate-100'
                          : 'bg-slate-800 text-slate-300 hover:text-slate-100'
                      }`}
                    >
                      <Archive size={13} />
                      {t('academicAssistant.archive', 'Archive')}
                    </button>
                  </div>

                  <div className="mt-4">
                    <div className="mb-2 flex items-center gap-2 text-xs font-medium text-slate-300">
                      <Tag size={13} />
                      {t('academicAssistant.tags', 'Tags')}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {tags.map((item) => (
                        <button
                          key={item.id}
                          type="button"
                          onClick={() => void removeTag({ resource_type: resourceRef.resource_type, resource_id: resourceRef.resource_id, tag: item.tag })}
                          className="rounded-full border border-slate-700 bg-slate-900 px-2 py-0.5 text-xs text-slate-300 hover:border-red-400 hover:text-red-300"
                        >
                          #{item.tag}
                        </button>
                      ))}
                    </div>
                    <div className="mt-2 flex gap-2">
                      <input
                        value={tagDraft}
                        onChange={(e) => setTagDraft(e.target.value)}
                        placeholder={t('academicAssistant.addTagPlaceholder', 'Add a tag')}
                        className="flex-1 rounded border border-slate-700 bg-slate-900 px-2 py-1.5 text-xs text-slate-200"
                      />
                      <button
                        type="button"
                        onClick={() => void handleAddTag()}
                        className="rounded bg-sky-600 px-3 py-1.5 text-xs text-white hover:bg-sky-500"
                      >
                        {t('common.save')}
                      </button>
                    </div>
                  </div>

                  <div className="mt-4">
                    <div className="mb-2 flex items-center gap-2 text-xs font-medium text-slate-300">
                      <MessageSquare size={13} />
                      {t('academicAssistant.notes', 'Notes')}
                    </div>
                    <div className="space-y-2">
                      {notes.map((item) => (
                        <div key={item.id} className="rounded-lg border border-slate-800 bg-slate-950/70 p-2 text-xs text-slate-300">
                          <div className="whitespace-pre-wrap">{item.note_md}</div>
                          <div className="mt-2 flex items-center gap-2">
                            <button
                              type="button"
                              onClick={() => {
                                setEditingNote(item);
                                setNoteDraft(item.note_md);
                              }}
                              className="text-sky-300 hover:text-sky-200"
                            >
                              {t('academicAssistant.edit', 'Edit')}
                            </button>
                            <button
                              type="button"
                              onClick={() => void removeNote(resourceRef, item.id)}
                              className="text-red-300 hover:text-red-200"
                            >
                              {t('common.delete')}
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                    <textarea
                      value={noteDraft}
                      onChange={(e) => setNoteDraft(e.target.value)}
                      placeholder={t('academicAssistant.notePlaceholder', 'Write a resource note')}
                      className="mt-2 h-24 w-full rounded border border-slate-700 bg-slate-900 px-2 py-2 text-xs text-slate-200"
                    />
                    <div className="mt-2 flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => void handleSaveNote()}
                        className="rounded bg-sky-600 px-3 py-1.5 text-xs text-white hover:bg-sky-500"
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
                          className="rounded border border-slate-700 px-3 py-1.5 text-xs text-slate-300"
                        >
                          {t('common.cancel')}
                        </button>
                      )}
                    </div>
                  </div>
                </section>
              )}

              <section className="rounded-xl border border-slate-700/60 bg-slate-950/40 p-3">
                <div className="mb-3 flex items-center justify-between">
                  <h4 className="text-sm font-medium text-slate-100">{t('academicAssistant.paperAnalysis', 'Paper Analysis')}</h4>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      disabled={!target.canRunAssistant || assistantLoading[`summary:${locator ? locator.paper_uid || `${locator.paper_id || ''}:${locator.collection || ''}` : ''}`]}
                      onClick={() => void handleRunSummary()}
                      className="inline-flex items-center gap-1 rounded-lg bg-sky-600 px-3 py-1.5 text-xs text-white hover:bg-sky-500 disabled:opacity-50"
                    >
                      {assistantLoading[`summary:${locator ? locator.paper_uid || `${locator.paper_id || ''}:${locator.collection || ''}` : ''}`] ? (
                        <Loader2 size={12} className="animate-spin" />
                      ) : (
                        <Sparkles size={12} />
                      )}
                      {t('academicAssistant.generateSummary', 'Summary')}
                    </button>
                    <button
                      type="button"
                      disabled={!target.canRunAssistant || !target.paperUid}
                      onClick={() => void handleStartMediaAnalysis()}
                      className="inline-flex items-center gap-1 rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-200 hover:bg-slate-800 disabled:opacity-50"
                    >
                      <Image size={12} />
                      {t('academicAssistant.mediaAnalysis', 'Media')}
                    </button>
                  </div>
                </div>
                {summary?.summary_md ? renderMarkdown(summary.summary_md) : (
                  <p className="text-xs text-slate-500">{t('academicAssistant.noSummaryYet', 'Run summary to generate a paper briefing with citations.')}</p>
                )}
              </section>

              <section className="rounded-xl border border-slate-700/60 bg-slate-950/40 p-3">
                <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-100">
                  <MessageSquare size={14} />
                  {t('academicAssistant.askQuestion', 'Targeted QA')}
                </div>
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder={t('academicAssistant.questionPlaceholder', 'Ask about method, result, limitation, or figures...')}
                  className="h-24 w-full rounded border border-slate-700 bg-slate-900 px-2 py-2 text-sm text-slate-200"
                />
                <div className="mt-2 flex justify-end">
                  <button
                    type="button"
                    disabled={!target.canRunAssistant || !question.trim() || assistantLoading[qaKey || '']}
                    onClick={() => void handleAskQuestion()}
                    className="inline-flex items-center gap-1 rounded-lg bg-sky-600 px-3 py-1.5 text-xs text-white hover:bg-sky-500 disabled:opacity-50"
                  >
                    {assistantLoading[qaKey || ''] ? <Loader2 size={12} className="animate-spin" /> : <Search size={12} />}
                    {t('academicAssistant.runQa', 'Ask')}
                  </button>
                </div>
                {answer?.answer_md ? (
                  <div className="mt-3">{renderMarkdown(answer.answer_md)}</div>
                ) : (
                  <p className="mt-3 text-xs text-slate-500">{t('academicAssistant.noAnswerYet', 'Ask a question to retrieve paper-grounded evidence.')}</p>
                )}
              </section>

              {target.paperUid && (
                <section className="rounded-xl border border-slate-700/60 bg-slate-950/40 p-3">
                  <h4 className="mb-3 text-sm font-medium text-slate-100">{t('academicAssistant.annotations', 'Annotations')}</h4>
                  <div className="space-y-2">
                    {annotations.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => {
                          setEditingAnnotation(item);
                          setAnnotationKind(item.target_kind);
                          setAnnotationTargetText(item.target_text || '');
                          setAnnotationDirective(item.directive || '');
                        }}
                        className="block w-full rounded-lg border border-slate-800 bg-slate-950/70 p-2 text-left text-xs text-slate-300 hover:border-slate-700"
                      >
                        <div className="font-medium text-slate-100">{item.target_kind}</div>
                        {item.target_text && <div className="mt-1 text-slate-400">{item.target_text}</div>}
                        <div className="mt-1 line-clamp-2 text-slate-300">{item.directive}</div>
                      </button>
                    ))}
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <select
                      value={annotationKind}
                      onChange={(e) => setAnnotationKind(e.target.value)}
                      className="rounded border border-slate-700 bg-slate-900 px-2 py-1.5 text-xs text-slate-200"
                    >
                      <option value="chunk">chunk</option>
                      <option value="figure">figure</option>
                      <option value="page_region">page_region</option>
                      <option value="canvas_section">canvas_section</option>
                    </select>
                    <input
                      value={annotationTargetText}
                      onChange={(e) => setAnnotationTargetText(e.target.value)}
                      placeholder={t('academicAssistant.annotationTarget', 'Target text')}
                      className="rounded border border-slate-700 bg-slate-900 px-2 py-1.5 text-xs text-slate-200"
                    />
                  </div>
                  <textarea
                    value={annotationDirective}
                    onChange={(e) => setAnnotationDirective(e.target.value)}
                    placeholder={t('academicAssistant.annotationDirective', 'Add your anchored note or directive')}
                    className="mt-2 h-20 w-full rounded border border-slate-700 bg-slate-900 px-2 py-2 text-xs text-slate-200"
                  />
                  <div className="mt-2 flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => void handleSaveAnnotation()}
                      className="rounded bg-sky-600 px-3 py-1.5 text-xs text-white hover:bg-sky-500"
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
                        className="rounded border border-slate-700 px-3 py-1.5 text-xs text-slate-300"
                      >
                        {t('common.cancel')}
                      </button>
                    )}
                  </div>
                </section>
              )}
            </>
          ) : (
            <div className="rounded-xl border border-slate-700/60 bg-slate-950/40 p-4 text-sm text-slate-400">
              {t('academicAssistant.noTarget', 'Choose a paper from Scholar to inspect it here.')}
            </div>
          )
        ) : (
          <>
            <section className="rounded-xl border border-slate-700/60 bg-slate-950/40 p-3">
              <div className="mb-3 text-sm font-medium text-slate-100">{t('academicAssistant.multiSelection', 'Selected Papers')}</div>
              {multiTargets.length === 0 ? (
                <p className="text-xs text-slate-500">{t('academicAssistant.noMultiTarget', 'Select library papers to compare or run discovery.')}</p>
              ) : (
                <div className="space-y-2">
                  {multiTargets.map((item, idx) => (
                    <div key={`${item.paperUid || item.title}-${idx}`} className="rounded-lg border border-slate-800 bg-slate-950/70 p-2 text-xs text-slate-300">
                      <div className="font-medium text-slate-100">{item.title}</div>
                      {item.subtitle && <div className="mt-1 text-slate-500">{item.subtitle}</div>}
                    </div>
                  ))}
                </div>
              )}
            </section>

            <section className="rounded-xl border border-slate-700/60 bg-slate-950/40 p-3">
              <div className="mb-3 flex items-center justify-between">
                <h4 className="text-sm font-medium text-slate-100">{t('academicAssistant.compareAndDiscover', 'Compare & Discover')}</h4>
                <button
                  type="button"
                  disabled={selectedPaperUids.length < 2 || assistantLoading[`compare:${compareKeyValue}`]}
                  onClick={() => void handleRunCompare()}
                  className="inline-flex items-center gap-1 rounded-lg bg-sky-600 px-3 py-1.5 text-xs text-white hover:bg-sky-500 disabled:opacity-50"
                >
                  {assistantLoading[`compare:${compareKeyValue}`] ? <Loader2 size={12} className="animate-spin" /> : <Sparkles size={12} />}
                  {t('academicAssistant.compare', 'Compare')}
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  disabled={selectedPaperUids.length === 0}
                  onClick={() => void handleStartDiscovery('missing-core')}
                  className="rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-200 hover:bg-slate-800 disabled:opacity-50"
                >
                  {t('academicAssistant.discoveryMissingCore', 'Missing Core')}
                </button>
                <button
                  type="button"
                  disabled={selectedPaperUids.length === 0}
                  onClick={() => void handleStartDiscovery('forward-tracking')}
                  className="rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-200 hover:bg-slate-800 disabled:opacity-50"
                >
                  {t('academicAssistant.discoveryForwardTracking', 'Forward Tracking')}
                </button>
                <button
                  type="button"
                  disabled={selectedPaperUids.length === 0}
                  onClick={() => void handleStartDiscovery('experts')}
                  className="rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-200 hover:bg-slate-800 disabled:opacity-50"
                >
                  {t('academicAssistant.discoveryExperts', 'Experts')}
                </button>
                <button
                  type="button"
                  disabled={selectedPaperUids.length === 0}
                  onClick={() => void handleStartDiscovery('institutions')}
                  className="rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-200 hover:bg-slate-800 disabled:opacity-50"
                >
                  {t('academicAssistant.discoveryInstitutions', 'Institutions')}
                </button>
              </div>
            </section>

            <section className="rounded-xl border border-slate-700/60 bg-slate-950/40 p-3">
              <h4 className="mb-3 text-sm font-medium text-slate-100">{t('academicAssistant.compareResult', 'Compare Result')}</h4>
              {compareResult ? (
                <>
                  {renderMarkdown(compareResult.narrative)}
                  {Object.keys(compareResult.comparison_matrix || {}).length > 0 && (
                    <div className="mt-4 overflow-x-auto">
                      <table className="w-full text-left text-xs text-slate-300">
                        <thead>
                          <tr className="border-b border-slate-800 text-slate-500">
                            <th className="px-2 py-1">{t('academicAssistant.aspect', 'Aspect')}</th>
                            {compareResult.papers.map((paper) => (
                              <th key={paper.paper_uid || paper.paper_id} className="px-2 py-1">
                                {paper.title}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(compareResult.comparison_matrix).map(([aspect, values]) => (
                            <tr key={aspect} className="border-b border-slate-900/80 align-top">
                              <td className="px-2 py-2 font-medium text-slate-100">{aspect}</td>
                              {compareResult.papers.map((paper) => (
                                <td key={`${aspect}-${paper.paper_uid || paper.paper_id}`} className="px-2 py-2 text-slate-300">
                                  {values[paper.paper_uid || ''] || '—'}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </>
              ) : (
                <p className="text-xs text-slate-500">{t('academicAssistant.noCompareYet', 'Compare selected papers to generate a narrative and aspect matrix.')}</p>
              )}
            </section>
          </>
        )}

        {trackedTaskIds.map((taskId) => (
          <TaskResultCard key={taskId} task={assistantTasks[taskId]} />
        ))}
      </div>
    </aside>
  );
}
