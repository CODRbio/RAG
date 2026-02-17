import { useEffect, useRef } from 'react';
import { Command } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useChatStore } from '../../stores';
import { COMMAND_LIST, type CommandDefinition } from '../../types';

interface CommandPaletteProps {
  inputValue: string;
  onSelectCommand: (command: CommandDefinition) => void;
}

export function CommandPalette({ inputValue, onSelectCommand }: CommandPaletteProps) {
  const { t } = useTranslation();
  const { showCommandPalette, setShowCommandPalette } = useChatStore();
  const paletteRef = useRef<HTMLDivElement>(null);

  // 检测是否应该显示命令面板
  const shouldShow = inputValue.startsWith('/');
  const filterText = inputValue.slice(1).toLowerCase();
  
  // 过滤匹配的命令
  const filteredCommands = shouldShow
    ? COMMAND_LIST.filter(cmd => 
        cmd.command.toLowerCase().includes(filterText) ||
        t(cmd.label).toLowerCase().includes(filterText)
      )
    : [];

  // 同步显示状态
  useEffect(() => {
    if (shouldShow && filteredCommands.length > 0) {
      setShowCommandPalette(true);
    } else {
      setShowCommandPalette(false);
    }
  }, [shouldShow, filteredCommands.length, setShowCommandPalette]);

  // 点击外部关闭
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (paletteRef.current && !paletteRef.current.contains(event.target as Node)) {
        setShowCommandPalette(false);
      }
    }
    if (showCommandPalette) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showCommandPalette, setShowCommandPalette]);

  if (!showCommandPalette || filteredCommands.length === 0) {
    return null;
  }

  return (
    <div
      ref={paletteRef}
      className="absolute bottom-full left-0 right-0 mb-2 bg-slate-900/95 backdrop-blur-md rounded-xl shadow-2xl border border-slate-700/50 overflow-hidden z-50 animate-in fade-in slide-in-from-bottom-2 duration-150"
    >
      <div className="px-3 py-2 bg-slate-800/50 border-b border-slate-700/50 flex items-center gap-2">
        <Command size={14} className="text-sky-500" />
        <span className="text-xs font-medium text-slate-400">{t('commands.availableCommands')}</span>
      </div>
      
      <div className="max-h-64 overflow-y-auto scrollbar-thin">
        {filteredCommands.map((cmd) => (
          <button
            key={cmd.command}
            onClick={() => onSelectCommand(cmd)}
            className="w-full flex items-start gap-3 px-4 py-3 text-left hover:bg-sky-900/20 transition-colors cursor-pointer border-b border-slate-800/50 last:border-b-0"
          >
            <code className="px-2 py-0.5 bg-slate-800 text-sky-400 rounded text-sm font-mono border border-slate-700/50">
              {cmd.command}
            </code>
            <div className="flex-1 min-w-0">
              <div className="font-medium text-slate-200 text-sm">{t(cmd.label)}</div>
              <div className="text-xs text-slate-400">{t(cmd.description)}</div>
              {cmd.example && (
                <div className="mt-1 text-xs text-slate-500 font-mono truncate">
                  {t('commands.example')} {cmd.example}
                </div>
              )}
            </div>
          </button>
        ))}
      </div>
      
      <div className="px-3 py-2 bg-slate-800/50 border-t border-slate-700/50">
        <div className="text-xs text-slate-500">
          {t('commands.press')} <kbd className="px-1 py-0.5 bg-slate-700 rounded text-slate-300 border border-slate-600">Tab</kbd> {t('commands.orClickToSelect')}
        </div>
      </div>
    </div>
  );
}
