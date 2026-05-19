<script lang="ts">
  // Single-line generation footer.  Mirrors the TUI's status footer:
  //   ● gen 47/512 [████░░░░░░] · 23 t/s · 2.1s · ppl 8.3
  //
  // Idle state collapses to "○ idle" before the first generation lands —
  // matches the TUI behavior of hiding stats until they have something
  // real to report.

  import Bar from "../lib/charts/Bar.svelte";
  import {
    genStatus,
    geometricMeanPpl,
    pendingActions,
    applyPendingActions,
    sendStop,
    onWsMessage,
  } from "../lib/stores.svelte";

  // Pending-actions badge — relocated here from the retired Topbar.  It
  // interrupts an in-flight gen to flush queued rack/sampling changes.
  const pendingCount = $derived(pendingActions.queue.length);

  // Stop-then-flush.  Subscribes once to the WS done event, sends stop,
  // then drains the queue when done lands.  If no gen is active, flush
  // immediately.
  function applyNow(): void {
    if (!genStatus.active) {
      applyPendingActions();
      return;
    }
    const off = onWsMessage((msg) => {
      if (msg.type === "done" || msg.type === "error") {
        // Store's own handler already drains the queue on done/error.
        off();
      }
    });
    sendStop();
  }

  // Live elapsed counter — ticks while gen is active, freezes on done so
  // the user can still read the final timing after the generation lands.
  let nowMs = $state(performance.now());
  $effect(() => {
    if (!genStatus.active) return;
    const id = setInterval(() => {
      nowMs = performance.now();
    }, 100);
    return () => clearInterval(id);
  });

  const elapsedSec = $derived.by(() => {
    if (!genStatus.startedAt) return 0;
    const end = genStatus.active ? nowMs : nowMs;
    return Math.max(0, (end - genStatus.startedAt) / 1000);
  });

  const tokPerSec = $derived.by(() => {
    if (genStatus.active && elapsedSec > 0) {
      return genStatus.tokensSoFar / elapsedSec;
    }
    return genStatus.tokPerSec;
  });

  const ppl = $derived(geometricMeanPpl(genStatus));

  // Have-anything: only render the full strip once a generation has at
  // least started.  "Active" is the obvious signal, but a finished gen
  // with startedAt set should also keep its trailing stats visible.
  const hasRun = $derived(genStatus.startedAt !== null);
</script>

<footer class="status-footer" aria-label="Generation status">
  {#if !hasRun && !genStatus.active}
    <span class="dot idle" aria-hidden="true">○</span>
    <span class="text">idle</span>
  {:else}
    <span class="dot {genStatus.active ? 'live' : 'done'}" aria-hidden="true">●</span>
    <span class="text">gen {genStatus.tokensSoFar}/{genStatus.maxTokens || "?"}</span>
    <span class="sep" aria-hidden="true">·</span>
    <span class="bar-wrap" aria-label="progress">
      <Bar
        value={genStatus.tokensSoFar}
        max={genStatus.maxTokens || Math.max(genStatus.tokensSoFar, 1)}
        width={120}
        height={6}
        color={genStatus.active ? "var(--accent-green)" : "var(--fg-muted)"}
      />
    </span>
    <span class="sep" aria-hidden="true">·</span>
    <span class="text">{tokPerSec.toFixed(1)} t/s</span>
    <span class="sep" aria-hidden="true">·</span>
    <span class="text">{elapsedSec.toFixed(1)}s</span>
    {#if ppl !== null && Number.isFinite(ppl)}
      <span class="sep" aria-hidden="true">·</span>
      <span class="text">ppl {ppl.toFixed(2)}</span>
    {/if}
    {#if !genStatus.active && genStatus.finishReason}
      <span class="sep" aria-hidden="true">·</span>
      <span class="text muted">{genStatus.finishReason}</span>
    {/if}
  {/if}

  {#if pendingCount > 0}
    <button
      class="pending-badge"
      type="button"
      onclick={applyNow}
      title="Apply queued rack and sampling changes; interrupts in-flight gen if needed"
    >
      {pendingCount} change{pendingCount === 1 ? "" : "s"} pending · apply now
    </button>
  {/if}
</footer>

<style>
  /* Embedded in the chat column, directly above the input row — a thin
   * status line.  Horizontal padding is zero so it aligns with the log
   * and input box; the hairline above separates it from the log. */
  .status-footer {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-2) 0;
    border-top: 1px solid var(--border);
    color: var(--fg-dim);
    font-size: var(--text-sm);
    font-family: var(--font-mono);
    min-height: 22px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .dot.live {
    color: var(--accent-green);
  }
  .dot.done {
    color: var(--fg-muted);
  }
  .dot.idle {
    color: var(--fg-muted);
  }
  .sep {
    color: var(--fg-muted);
  }
  .text.muted {
    color: var(--fg-muted);
  }
  .bar-wrap {
    display: inline-flex;
    align-items: center;
  }

  /* Pending-actions badge — pushed to the right edge of the footer. */
  .pending-badge {
    margin-left: auto;
    background: rgba(242, 184, 75, 0.12);
    color: var(--accent-amber);
    border: 1px solid var(--border);
    padding: var(--space-1) var(--space-4);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    font-family: var(--font-ui);
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  .pending-badge:hover {
    background: rgba(242, 184, 75, 0.2);
  }
</style>
