import { Loader2, ChevronRight } from 'lucide-react';
import type { ClarifyQuestion } from '../../../types';

interface ClarifyPhaseProps {
  questions: ClarifyQuestion[];
  answers: Record<string, string>;
  onAnswerChange: (id: string, value: string) => void;
  isClarifying: boolean;
  onRegenerate: () => void;
  depth: 'lite' | 'comprehensive';
  scopeModel: string;
  outputLanguage: string;
  topic: string;
}

const getQuestionRationale = (questionText: string): string => {
  const text = questionText.toLowerCase();
  if (text.includes('范围') || text.includes('scope')) {
    return '用于锁定研究边界，避免大纲发散。';
  }
  if (text.includes('受众') || text.includes('风格') || text.includes('audience') || text.includes('style')) {
    return '用于匹配表达方式和写作深度。';
  }
  if (text.includes('篇幅') || text.includes('深度') || text.includes('字数') || text.includes('length')) {
    return '用于控制章节粒度和信息密度。';
  }
  if (text.includes('排除') || text.includes('exclude')) {
    return '用于减少无关检索和错误扩展。';
  }
  if (text.includes('语言') || text.includes('language')) {
    return '用于提前确定文献与输出语言策略。';
  }
  return '用于减少歧义，让后续大纲更贴合目标。';
};

export function ClarifyPhase({
  questions,
  answers,
  onAnswerChange,
  isClarifying,
  onRegenerate,
  depth,
  scopeModel,
  outputLanguage,
  topic,
}: ClarifyPhaseProps) {
  return (
    <>
      {questions.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-gray-600">
            请补充以下信息（可选，共 {questions.length} 题）
          </h3>
          {questions.map((q) => (
            <div key={q.id} className="bg-gray-50 rounded-lg p-3">
              <label className="block text-sm text-gray-700 mb-1.5">{q.text}</label>
              {q.question_type === 'choice' && q.options.length > 0 ? (
                <select
                  value={answers[q.id] || ''}
                  onChange={(e) => onAnswerChange(q.id, e.target.value)}
                  className="w-full border border-gray-200 rounded-md px-2.5 py-1.5 text-sm bg-white text-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none"
                >
                  {q.options.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={answers[q.id] || ''}
                  onChange={(e) => onAnswerChange(q.id, e.target.value)}
                  placeholder={q.default || '输入回答...'}
                  className="w-full border border-gray-200 bg-white text-gray-900 rounded-md px-2.5 py-1.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                />
              )}
              <div className="mt-1.5 text-xs text-gray-500">
                为什么问：{getQuestionRationale(q.text)}
              </div>
            </div>
          ))}
        </div>
      )}

      {questions.length === 0 && (
        <div className="text-center py-4 text-gray-500 text-sm bg-gray-50 rounded-lg border border-gray-200">
          {isClarifying ? (
            <div className="flex items-center justify-center gap-2">
              <Loader2 size={14} className="animate-spin text-indigo-500" />
              <span>正在生成澄清问题...</span>
            </div>
          ) : (
            'Click "Regenerate" to generate clarification questions, or proceed directly to outline.'
          )}
        </div>
      )}

      <div className="flex items-center justify-between bg-gray-50 border border-gray-200 rounded-lg px-3 py-2.5">
        <div className="text-[11px] text-gray-500 flex items-center gap-1.5 flex-wrap">
          <span className="font-medium text-gray-700">{depth === 'lite' ? 'Lite' : 'Comprehensive'}</span>
          <span className="text-gray-300">|</span>
          <span>Scope: <span className="font-medium">{scopeModel || 'default'}</span></span>
          <span className="text-gray-300">|</span>
          <span>Lang: {outputLanguage === 'auto' ? 'Auto' : outputLanguage}</span>
          <span className="text-gray-300">|</span>
          <span className="text-[10px] text-gray-400">via input &#9881;</span>
        </div>
        <button
          onClick={onRegenerate}
          disabled={isClarifying || !topic.trim()}
          className="inline-flex items-center gap-1 px-2 py-1 border rounded-md text-[11px] text-indigo-600 hover:bg-indigo-50 disabled:opacity-50 shrink-0 ml-2"
        >
          {isClarifying ? <Loader2 size={10} className="animate-spin" /> : <ChevronRight size={10} />}
          Regenerate
        </button>
      </div>
    </>
  );
}
