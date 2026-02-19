import { useMemo } from 'react';
import { Copy } from 'lucide-react';
import type { ResearchMonitorState, EfficiencyRow } from './types';

interface ProgressMonitorProps {
  researchMonitor: ResearchMonitorState;
  progressLogs: string[];
  depth: 'lite' | 'comprehensive';
  optimizationPromptDraft: string;
  onGenerateOptimizationPrompt: (
    lowEfficiencyRows: EfficiencyRow[],
    allRows: EfficiencyRow[],
    targetCoverage: number,
  ) => void;
  onInsertOptimizationPrompt: () => void;
  onCopyOptimizationPrompt: () => void;
}

export function ProgressMonitor({
  researchMonitor,
  progressLogs,
  depth,
  optimizationPromptDraft,
  onGenerateOptimizationPrompt,
  onInsertOptimizationPrompt,
  onCopyOptimizationPrompt,
}: ProgressMonitorProps) {
  const targetCoverage = depth === 'lite' ? 0.6 : 0.8;

  const monitorSectionEntries = useMemo(
    () => Object.entries(researchMonitor.sectionCoverage),
    [researchMonitor.sectionCoverage],
  );

  const sectionEfficiencyRows = useMemo<EfficiencyRow[]>(() => {
    const rows = monitorSectionEntries
      .map(([section, coverageValues]) => {
        if (!coverageValues || coverageValues.length < 2) return null;
        const stepValues = researchMonitor.sectionSteps[section] || [];
        const firstCoverage = coverageValues[0];
        const lastCoverage = coverageValues[coverageValues.length - 1];
        const rounds = coverageValues.length - 1;
        const deltaCoverage = lastCoverage - firstCoverage;
        const avgDelta = deltaCoverage / Math.max(1, rounds);
        const lastDelta = coverageValues[coverageValues.length - 1] - coverageValues[coverageValues.length - 2];
        let per10Steps: number | null = null;
        if (stepValues.length >= 2) {
          const stepSpan = stepValues[stepValues.length - 1] - stepValues[0];
          if (stepSpan > 0) {
            per10Steps = deltaCoverage / (stepSpan / 10);
          }
        }
        const score = avgDelta * 100;
        let level: 'high' | 'medium' | 'low' = 'low';
        if (avgDelta >= 0.08) level = 'high';
        else if (avgDelta >= 0.03) level = 'medium';
        return { section, firstCoverage, lastCoverage, rounds, avgDelta, lastDelta, per10Steps, score, level };
      })
      .filter((row): row is NonNullable<typeof row> => Boolean(row));
    return rows.sort((a, b) => b.score - a.score);
  }, [monitorSectionEntries, researchMonitor.sectionSteps]);

  const highEfficiencyRows = useMemo(
    () => sectionEfficiencyRows.filter((row) => row.level === 'high').slice(0, 3),
    [sectionEfficiencyRows],
  );
  const lowEfficiencyRows = useMemo(
    () => sectionEfficiencyRows.filter((row) => row.level === 'low').slice(0, 3),
    [sectionEfficiencyRows],
  );

  return (
    <div className="space-y-2">
      <div className="text-sm font-medium text-gray-700">Research Progress</div>
      <div className="text-xs text-gray-500">Executing confirmed plan...</div>

      <div className="border border-gray-200 rounded-lg p-2.5 bg-white space-y-2">
        <div className="text-xs font-medium text-gray-700">Research Monitor</div>
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div className="rounded border border-gray-200 bg-gray-50 px-2 py-1.5">
            <div className="text-gray-500">Graph steps</div>
            <div className={`font-medium ${
              researchMonitor.costState === 'force'
                ? 'text-red-600'
                : researchMonitor.costState === 'warn'
                  ? 'text-amber-600'
                  : 'text-gray-800'
            }`}>
              {researchMonitor.graphSteps > 0 ? researchMonitor.graphSteps : '--'}
            </div>
            <div className="text-[10px] text-gray-400">
              warn {researchMonitor.warnSteps ?? '--'} / force {researchMonitor.forceSteps ?? '--'}
            </div>
          </div>
          <div className="rounded border border-gray-200 bg-gray-50 px-2 py-1.5">
            <div className="text-gray-500">Cost status</div>
            <div className={`font-medium ${
              researchMonitor.costState === 'force'
                ? 'text-red-600'
                : researchMonitor.costState === 'warn'
                  ? 'text-amber-600'
                  : 'text-emerald-600'
            }`}>
              {researchMonitor.costState.toUpperCase()}
            </div>
            <div className="text-[10px] text-gray-400">node: {researchMonitor.lastNode || '--'}</div>
          </div>
          <div className="rounded border border-gray-200 bg-gray-50 px-2 py-1.5">
            <div className="text-gray-500">Self-correction</div>
            <div className="font-medium text-gray-800">{researchMonitor.selfCorrectionCount}</div>
          </div>
          <div className="rounded border border-gray-200 bg-gray-50 px-2 py-1.5">
            <div className="text-gray-500">Plateau early-stop</div>
            <div className="font-medium text-gray-800">{researchMonitor.plateauEarlyStopCount}</div>
          </div>
        </div>

        <div className="rounded border border-gray-200 bg-gray-50 px-2 py-1.5 text-[11px]">
          <span className="text-gray-500">Write verification passes: </span>
          <span className="font-medium text-gray-800">{researchMonitor.verificationContextCount}</span>
        </div>

        {monitorSectionEntries.length > 0 && (
          <div className="space-y-1">
            <div className="text-[11px] text-gray-600">Section coverage curve</div>
            {monitorSectionEntries.map(([section, values]) => (
              <div key={`cov-${section}`} className="text-[11px] text-gray-700 border border-gray-200 rounded px-2 py-1 bg-gray-50">
                <span className="font-medium">{section}</span>
                <span className="text-gray-400"> : </span>
                {values.map((v, idx) => (
                  <span key={`${section}-${idx}`} className="font-mono text-[10px]">
                    {idx > 0 ? ' -> ' : ''}
                    {v.toFixed(2)}
                  </span>
                ))}
              </div>
            ))}
          </div>
        )}

        {sectionEfficiencyRows.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-[11px] text-gray-600">Efficiency insights</div>
            {sectionEfficiencyRows.slice(0, 5).map((row) => (
              <div key={`eff-${row.section}`} className="text-[11px] border border-gray-200 rounded px-2 py-1.5 bg-gray-50">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-gray-700">{row.section}</span>
                  <span className={`font-medium ${
                    row.level === 'high'
                      ? 'text-emerald-600'
                      : row.level === 'medium'
                        ? 'text-amber-600'
                        : 'text-red-600'
                  }`}>
                    {row.level.toUpperCase()} ({row.score.toFixed(1)})
                  </span>
                </div>
                <div className="text-[10px] text-gray-500 mt-0.5">
                  cov {row.firstCoverage.toFixed(2)} {'->'} {row.lastCoverage.toFixed(2)}
                  {' | '}avg gain {row.avgDelta.toFixed(3)}/round
                  {row.per10Steps !== null ? ` | gain/10 steps ${row.per10Steps.toFixed(3)}` : ''}
                </div>
              </div>
            ))}

            <div className="rounded border border-gray-200 bg-white px-2 py-1.5 text-[11px] text-gray-600">
              {highEfficiencyRows.length > 0 && (
                <div>
                  Continue deepening: {highEfficiencyRows.map((r) => r.section).join(' / ')}.
                </div>
              )}
              {lowEfficiencyRows.length > 0 && (
                <div>
                  Optimize first (prompt/evidence): {lowEfficiencyRows.map((r) => r.section).join(' / ')}.
                </div>
              )}
              <div>
                Action hint: if low-efficiency section is below target coverage ({targetCoverage.toFixed(2)}), add section-specific constraints,
                terminology variants, or temporary materials in Intervention before next run.
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => onGenerateOptimizationPrompt(lowEfficiencyRows, sectionEfficiencyRows, targetCoverage)}
                className="px-2 py-1 border rounded-md text-[11px] text-indigo-600 hover:bg-indigo-50"
              >
                Generate optimization prompt template
              </button>
              {optimizationPromptDraft.trim() && (
                <>
                  <button
                    onClick={onInsertOptimizationPrompt}
                    className="px-2 py-1 border rounded-md text-[11px] text-emerald-700 hover:bg-emerald-50"
                  >
                    Insert to Intervention
                  </button>
                  <button
                    onClick={onCopyOptimizationPrompt}
                    className="inline-flex items-center gap-1 px-2 py-1 border rounded-md text-[11px] text-gray-600 hover:bg-gray-50"
                  >
                    <Copy size={11} />
                    Copy
                  </button>
                </>
              )}
            </div>

            {optimizationPromptDraft.trim() && (
              <textarea
                readOnly
                value={optimizationPromptDraft}
                className="w-full min-h-32 border border-gray-200 rounded-md px-2 py-1.5 text-[11px] font-mono bg-gray-50"
              />
            )}
          </div>
        )}
      </div>

      <div className="max-h-56 overflow-auto border border-gray-200 rounded-lg p-2 bg-gray-50 text-xs space-y-1">
        {progressLogs.length === 0 ? (
          <div className="text-gray-400">Waiting for progress events...</div>
        ) : (
          progressLogs.map((line, idx) => <div key={`log-${idx}`}>{line}</div>)
        )}
      </div>
    </div>
  );
}
