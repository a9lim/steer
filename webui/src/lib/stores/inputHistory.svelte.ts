/** In-memory command/input recall ring. */

export const INPUT_HISTORY_MAX = 200;

export interface InputHistoryState {
  /** Submitted lines, oldest first. */
  entries: string[];
  /** Cursor into ``entries`` while recall is in flight. */
  index: number | null;
  /** Draft text captured when recall begins. */
  stash: string;
}

export const inputHistory: InputHistoryState = $state({
  entries: [],
  index: null,
  stash: "",
});

export function pushInputHistory(text: string): void {
  const trimmed = text.trim();
  if (!trimmed) return;
  const entries = inputHistory.entries;
  const last = entries.length > 0 ? entries[entries.length - 1] : null;
  if (last !== trimmed) {
    const next = [...entries, trimmed];
    inputHistory.entries = next.length > INPUT_HISTORY_MAX
      ? next.slice(next.length - INPUT_HISTORY_MAX)
      : next;
  }
  inputHistory.index = null;
  inputHistory.stash = "";
}

export function navigateInputHistory(
  delta: -1 | 1,
  currentInput: string,
): string | null {
  const entries = inputHistory.entries;
  if (entries.length === 0) return null;

  if (inputHistory.index === null) {
    if (delta > 0) return null;
    inputHistory.stash = currentInput;
    inputHistory.index = entries.length - 1;
    return entries[inputHistory.index];
  }

  const newIdx = inputHistory.index + delta;
  if (newIdx < 0) {
    inputHistory.index = 0;
    return entries[0];
  }
  if (newIdx >= entries.length) {
    inputHistory.index = null;
    const stash = inputHistory.stash;
    inputHistory.stash = "";
    return stash;
  }
  inputHistory.index = newIdx;
  return entries[newIdx];
}
