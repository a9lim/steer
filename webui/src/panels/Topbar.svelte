<script lang="ts">
  // Topbar — a thin identity + status strip.  Navigation (the tool
  // launchers) lives on the WorkspaceRail; the live conversation actions
  // (clear / rewind / regen / stop / transcript / auto-regen) live in the
  // Chat panel.  All this bar carries is the brand, the session status,
  // and the pending-actions badge — a transient alert that interrupts an
  // in-flight gen to flush queued rack/sampling changes.

  import {
    sessionState,
    genStatus,
    pendingActions,
    applyPendingActions,
    sendStop,
    onWsMessage,
  } from "../lib/stores.svelte";

  const status = $derived.by(() => {
    const info = sessionState.info;
    if (!info) return "connecting…";
    const thinking = info.supports_thinking ? "thinking" : "no-thinking";
    return `${info.model_id} · ${info.device}/${info.dtype} · ${thinking}`;
  });

  const pendingCount = $derived(pendingActions.queue.length);

  // Stop-then-flush.  Subscribes once to the WS done event, sends stop,
  // then drains the queue when done lands.  Off the happy path: if no gen
  // is active, just flush immediately.
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
</script>

<header class="topbar" aria-label="saklas top bar">
  <div class="left">
    <span class="brand">saklas</span>
  </div>

  <div class="center">
    <span class="status" title={sessionState.error ?? undefined}>
      {status}
    </span>
  </div>

  <div class="right">
    {#if pendingCount > 0}
      <button
        class="pending-badge"
        type="button"
        onclick={applyNow}
        title="Apply queued rack/sampling changes; interrupts in-flight gen if needed"
      >
        {pendingCount} change{pendingCount === 1 ? "" : "s"} pending — apply now
      </button>
    {/if}
  </div>
</header>

<style>
  .topbar {
    display: grid;
    grid-template-columns: auto minmax(12rem, 1fr) minmax(0, auto);
    align-items: center;
    gap: 1.5em;
    padding: 0.55rem 0.85rem;
    /* Token-driven glass — matches the site's .glass nav. */
    background: var(--bg-panel);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--border);
    min-height: 44px;
    font-family: var(--font-ui);
  }
  .left {
    display: flex;
    align-items: center;
  }
  .brand {
    color: var(--accent);
    font-weight: 700;
    letter-spacing: 0;
    font-size: 0.95rem;
  }
  .center {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .status {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    font-family: var(--font-mono);
  }
  .right {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 0.5em;
  }

  .pending-badge {
    background: rgba(242, 184, 75, 0.12);
    color: var(--accent-amber);
    border: 1px solid rgba(242, 184, 75, 0.5);
    padding: 0.35rem 0.62rem;
    border-radius: var(--radius);
    font-size: 0.78rem;
    font-family: var(--font-ui);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .pending-badge:hover {
    background: rgba(242, 184, 75, 0.2);
    transform: translateY(-1px);
  }
  .pending-badge:active {
    transform: translateY(0);
  }
</style>
