<script lang="ts">
  // Compare drawer — pairwise / ranked cosine similarity.  Mirrors the
  // TUI's ``/compare <a> [b]``.  Two modes:
  //
  // * ranked: one concept input.  Server returns the full N×N matrix;
  //   we slice the row for the chosen concept and render a sorted list
  //   of (other, cosine, layers_shared) rows.
  // * pairwise: two concept inputs.  Render the single number + the
  //   layers_shared count.
  //
  // Concept inputs autocomplete from the live ``vectorRack`` keys via a
  // <datalist> — refreshes after extraction in other drawers because
  // the rack store is reactive.

  import { apiVectors, ApiError } from "../lib/api";
  import {
    closeDrawer,
    vectorRack,
    refreshVectorList,
  } from "../lib/stores.svelte";
  import type { CorrelationData } from "../lib/api";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  type Mode = "ranked" | "pairwise";
  let mode: Mode = $state("ranked");
  let conceptA = $state("");
  let conceptB = $state("");
  let busy = $state(false);
  let errorMsg: string | null = $state(null);
  let result: CorrelationData | null = $state(null);

  const rackNames = $derived([...vectorRack.entries.keys()].sort());

  const validity = $derived.by(() => {
    if (!conceptA.trim())
      return { ok: false, reason: "concept a is required" } as const;
    if (mode === "pairwise" && !conceptB.trim())
      return {
        ok: false,
        reason: "pairwise mode needs both concepts",
      } as const;
    return { ok: true, reason: null } as const;
  });

  /** Sorted (other, cosine, layers) rows for the ranked mode.  Filters
   * out the self-row and any null entries (which appear when no
   * overlapping layers exist).  Sort: |cosine| desc, then by name. */
  interface RankedRow {
    other: string;
    cosine: number;
    layersShared: number;
  }
  const rankedRows: RankedRow[] = $derived.by(() => {
    if (mode !== "ranked" || !result) return [];
    const a = conceptA.trim();
    const row = result.matrix[a];
    if (!row) return [];
    const out: RankedRow[] = [];
    for (const [other, val] of Object.entries(row)) {
      if (other === a) continue;
      if (val === null || val === undefined) continue;
      const pairKey = pairKeyOf(a, other);
      out.push({
        other,
        cosine: val,
        layersShared: result.layers_shared[pairKey] ?? 0,
      });
    }
    out.sort((x, y) => {
      const dx = Math.abs(y.cosine) - Math.abs(x.cosine);
      if (dx !== 0) return dx;
      return x.other.localeCompare(y.other);
    });
    return out;
  });

  const pairwiseValue: { cosine: number | null; layersShared: number } | null =
    $derived.by(() => {
      if (mode !== "pairwise" || !result) return null;
      const a = conceptA.trim();
      const b = conceptB.trim();
      const row = result.matrix[a];
      if (!row) return null;
      const v = row[b];
      const lk = pairKeyOf(a, b);
      return {
        cosine: typeof v === "number" ? v : null,
        layersShared: result.layers_shared[lk] ?? 0,
      };
    });

  function pairKeyOf(a: string, b: string): string {
    return [a, b].sort().join("|");
  }

  async function run(): Promise<void> {
    if (!validity.ok || busy) return;
    busy = true;
    errorMsg = null;
    result = null;
    try {
      // Refresh the rack so newly-extracted vectors are in the lookup
      // before we ask the server for correlations — cheap idempotent.
      try {
        await refreshVectorList();
      } catch {
        /* non-fatal */
      }
      const names =
        mode === "pairwise"
          ? [conceptA.trim(), conceptB.trim()]
          : null;
      const data = await apiVectors.correlation(names);
      result = data;
    } catch (e) {
      if (e instanceof ApiError) {
        const detail =
          e.body && typeof e.body === "object" && "detail" in (e.body as object)
            ? String((e.body as { detail: unknown }).detail)
            : e.message;
        errorMsg = `${e.status}: ${detail}`;
      } else {
        errorMsg = e instanceof Error ? e.message : String(e);
      }
    } finally {
      busy = false;
    }
  }

  function clamp(v: number, lo: number, hi: number): number {
    return Math.min(Math.max(v, lo), hi);
  }

  function barWidthPct(cosine: number): number {
    return clamp(Math.abs(cosine) * 100, 0, 100);
  }

  function barColor(cosine: number): string {
    return cosine >= 0 ? "var(--accent-green)" : "var(--accent-red)";
  }
</script>

<section class="drawer-shell" aria-label="Compare drawer">
  <header class="header">
    <span class="title">compare vectors</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button
    >
  </header>

  <div class="body">
    <div class="mode-row" role="radiogroup" aria-label="Compare mode">
      <label class="mode-opt">
        <input
          type="radio"
          name="cmp-mode"
          value="ranked"
          checked={mode === "ranked"}
          onchange={() => (mode = "ranked")}
        />
        <span>ranked vs all</span>
      </label>
      <label class="mode-opt">
        <input
          type="radio"
          name="cmp-mode"
          value="pairwise"
          checked={mode === "pairwise"}
          onchange={() => (mode = "pairwise")}
        />
        <span>pairwise</span>
      </label>
    </div>

    <datalist id="cmp-rack-names">
      {#each rackNames as n (n)}
        <option value={n}></option>
      {/each}
    </datalist>

    <form
      class="form"
      onsubmit={(ev) => {
        ev.preventDefault();
        void run();
      }}
    >
      <label class="field">
        <span class="label">concept a</span>
        <input
          type="text"
          class="input"
          bind:value={conceptA}
          list="cmp-rack-names"
          disabled={busy}
          placeholder="e.g. honest"
          autocomplete="off"
          spellcheck="false"
        />
      </label>

      {#if mode === "pairwise"}
        <label class="field">
          <span class="label">concept b</span>
          <input
            type="text"
            class="input"
            bind:value={conceptB}
            list="cmp-rack-names"
            disabled={busy}
            placeholder="e.g. warm"
            autocomplete="off"
            spellcheck="false"
          />
        </label>
      {/if}

      {#if !validity.ok}
        <p class="validation">{validity.reason}</p>
      {/if}
    </form>

    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}

    {#if mode === "pairwise" && pairwiseValue}
      <div class="result-box">
        {#if pairwiseValue.cosine === null}
          <p class="result-line">
            cosine: <span class="dim">— (no shared layers)</span>
          </p>
        {:else}
          <p class="result-line">
            cosine
            <span
              class="value"
              style="color: {barColor(pairwiseValue.cosine)}"
            >{pairwiseValue.cosine.toFixed(4)}</span>
          </p>
        {/if}
        <p class="result-line">
          layers shared: <span class="value">{pairwiseValue.layersShared}</span>
        </p>
      </div>
    {/if}

    {#if mode === "ranked" && rankedRows.length > 0}
      <div class="ranked-list" aria-label="Ranked cosine list">
        {#each rankedRows as row (row.other)}
          <div class="rrow">
            <span class="rname">{row.other}</span>
            <div class="rbar-track">
              <div
                class="rbar-fill"
                style="width: {barWidthPct(row.cosine)}%; background: {barColor(row.cosine)};"
              ></div>
            </div>
            <span
              class="rval"
              style="color: {barColor(row.cosine)}"
            >{row.cosine >= 0 ? "+" : ""}{row.cosine.toFixed(3)}</span>
            <span class="rlayers">L{row.layersShared}</span>
          </div>
        {/each}
      </div>
    {:else if mode === "ranked" && result && rankedRows.length === 0}
      <p class="dim small">no comparable concepts (no shared layers)</p>
    {/if}
  </div>

  <footer class="footer">
    <button type="button" class="btn" onclick={closeDrawer}>close</button>
    <button
      type="button"
      class="btn primary"
      onclick={run}
      disabled={!validity.ok || busy}
    >{busy ? "comparing…" : "run"}</button>
  </footer>
</section>

<style>
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-6);
    border-bottom: 1px solid var(--border);
  }
  .title {
    color: var(--accent-blue);
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    cursor: pointer;
    padding: var(--space-2) var(--space-2);
  }
  .close:hover {
    color: var(--accent-red);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    min-height: 0;
  }
  .mode-row {
    display: flex;
    gap: var(--space-6);
  }
  .mode-opt {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--fg-strong);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .form {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    text-transform: lowercase;
  }
  .input {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
  }
  .input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .validation {
    color: var(--accent-yellow);
    margin: 0;
    font-size: var(--text-sm);
  }
  .error {
    color: var(--accent-error);
    margin: 0;
    font-size: var(--text-sm);
    word-break: break-word;
  }
  .result-box {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    padding: var(--space-4) var(--space-5);
  }
  .result-line {
    margin: var(--space-1) 0;
    color: var(--fg-strong);
  }
  .value {
    color: var(--accent-green);
    font-weight: var(--weight-medium);
  }
  .dim {
    color: var(--fg-muted);
  }
  .small {
    font-size: var(--text-sm);
  }
  .ranked-list {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    background: var(--bg-deep);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-4);
  }
  .rrow {
    display: grid;
    grid-template-columns: minmax(8em, 14em) 1fr 4em 3em;
    align-items: center;
    gap: var(--space-3);
    font-size: var(--text-sm);
  }
  .rname {
    color: var(--fg-strong);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .rbar-track {
    background: var(--bg-elev);
    height: 6px;
    border-radius: var(--radius);
    position: relative;
    overflow: hidden;
  }
  .rbar-fill {
    height: 100%;
    background: var(--accent-green);
  }
  .rval {
    text-align: right;
    color: var(--fg-strong);
  }
  .rlayers {
    color: var(--fg-muted);
    text-align: right;
    font-size: var(--text-xs);
  }
  .footer {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
    padding: var(--space-6);
    border-top: 1px solid var(--border);
  }
  .btn {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: var(--space-2) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
  }
  .btn:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
  .btn:disabled {
    color: var(--fg-muted);
    border-color: var(--border);
    cursor: not-allowed;
  }
  .btn.primary {
    background: var(--accent);
    color: var(--text-on-accent);
    border-color: var(--accent);
  }
  .btn.primary:hover:not(:disabled) {
    background: var(--accent-light);
    border-color: var(--accent-light);
  }
  .btn.primary:disabled {
    background: var(--bg-elev);
  }
</style>
