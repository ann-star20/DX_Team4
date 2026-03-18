import { useEffect, useCallback } from 'react';

/**
 * 셋톱박스/리모컨 시뮬레이션: 방향키, Enter, Back
 */
export function useKeyboardNav(options: {
  onUp?: () => void;
  onDown?: () => void;
  onLeft?: () => void;
  onRight?: () => void;
  onEnter?: () => void;
  onBack?: () => void;
}) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      const isTextInput = tag === 'INPUT' || tag === 'TEXTAREA';

      switch (e.key) {
        case 'ArrowUp':
          if (!isTextInput) { e.preventDefault(); options.onUp?.(); }
          break;
        case 'ArrowDown':
          if (!isTextInput) { e.preventDefault(); options.onDown?.(); }
          break;
        case 'ArrowLeft':
          if (!isTextInput) { e.preventDefault(); options.onLeft?.(); }
          break;
        case 'ArrowRight':
          if (!isTextInput) { e.preventDefault(); options.onRight?.(); }
          break;
        case 'Enter':
          if (!isTextInput) { e.preventDefault(); options.onEnter?.(); }
          break;
        case 'Backspace':
          if (!isTextInput) { e.preventDefault(); options.onBack?.(); }
          break;
        case 'Escape':
          e.preventDefault();
          options.onBack?.();
          break;
      }
    },
    [options]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
}
