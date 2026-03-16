declare function ym(
  counterId: number,
  action: string,
  ...args: unknown[]
): void;

const METRICA_COUNTER_ID = 107726491;

function track(goal: string, params?: Record<string, unknown>): void {
  if (typeof ym === "undefined") return;
  ym(METRICA_COUNTER_ID, "reachGoal", goal, params ?? {});
}

const _sessionStart = Date.now();

function trackSessionStart(abGroup: string): void {
  const isReturning = Boolean(localStorage.getItem("tutorialSeen"));
  track("session_start", { is_returning: isReturning, ab_group: abGroup });

  window.addEventListener("beforeunload", () => {
    const duration = Math.round((Date.now() - _sessionStart) / 1000);
    track("session_end", { duration_sec: duration });
  });
}

function trackLevelStart(level: number, themeLetter: string): void {
  track("level_started", { level, theme_letter: themeLetter });
}

function trackLevelComplete(
  level: number,
  hintsUsed: number,
  wordsTotal: number,
): void {
  track("level_completed", {
    level,
    hints_used: hintsUsed,
    words_total: wordsTotal,
  });
}

function trackHintReceived(
  level: number,
  wordsFound: number,
  wordsTotal: number,
): void {
  track("hint_received", {
    level,
    words_found: wordsFound,
    words_total: wordsTotal,
    completion_pct:
      wordsTotal > 0 ? Math.round((wordsFound / wordsTotal) * 100) : 0,
  });
}

function trackDropOff(
  level: number,
  wordsFound: number,
  wordsTotal: number,
): void {
  track("drop_off", {
    level,
    words_found: wordsFound,
    words_total: wordsTotal,
    completion_pct:
      wordsTotal > 0 ? Math.round((wordsFound / wordsTotal) * 100) : 0,
  });
}
