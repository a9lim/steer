<script lang="ts">
  // Cross-branch diff drawer — phase 5.  Renders a side-by-side
  // word-level diff between two (or more) assistant nodes, plus the
  // sorted readings-delta table and the per-token alignment for the
  // primary pair.
  //
  // Drawer params: ``{node_ids: string[], parent_id?: string}``.
  // Surfaces:
  //   * 2 ids — single side-by-side comparison.
  //   * 3+ ids — N-column rendering, one column per id, diffs computed
  //     against the first id as anchor.

  import { onMount } from "svelte";
  import { apiTree, ApiError } from "../lib/api";
  import {
    clearNodeSelection,
    closeDrawer,
    drawerState,
    loomTree,
  } from "../lib/stores.svelte";
  import type {
    DiffReadingDeltaJSON,
    DiffTextSpanJSON,
    DiffTokenSpanJSON,
    NodeDiffJSON,
  } from "../lib/types";

  // --------------------------------------------------------- props ---

  interface Params {
    node_ids?: string[];
    parent_id?: string | null;
  }

  let { params }: { params: unknown } = $props();
  const ids = $derived.by<string[]>(() => {
    const p = (params ?? drawerState.params ?? {}) as Params;
    return Array.isArray(p.node_ids) ? p.node_ids.filter(Boolean) : [];
  });
  const anchorId = $derived(ids[0] ?? null);

  // --------------------------------------------------------- diffs ---

  // For each non-anchor id, fetch the diff against the anchor.  When
  // there are N>=2 ids, we fetch N-1 diffs.  Single fetch when N=2
  // (the common case).  Re-runs whenever ``ids`` changes.

  type DiffOrError =
    | { kind: "ok"; diff: NodeDiffJSON }
    | { kind: "err"; message: string };

  let diffs: DiffOrError[] = $state([]);
  let loading = $state(false);
  let hoveredAnchorIdx: number | null = $state(null);

  /** Layout helper — ``"unified"`` stacks the diff vertically for
   * narrow screens; ``"side-by-side"`` is the default. */
  let layout: "side-by-side" | "unified" = $state("side-by-side");
  let sortBy: "magnitude" | "name" = $state("magnitude");

  async function fetchAll(): Promise<void> {
    const list = ids;
    if (list.length < 2 || !anchorId) {
      diffs = [];
      return;
    }
    loading = true;
    const out: DiffOrError[] = [];
    try {
      for (let i = 1; i < list.length; i++) {
        try {
          const r = await apiTree.diff(anchorId, list[i]);
          out.push({ kind: "ok", diff: r });
        } catch (e) {
          if (e instanceof ApiError) {
            const detail =
              e.body && typeof e.body === "object" && "detail" in (e.body as object)
                ? String((e.body as { detail: unknown }).detail)
                : e.message;
            out.push({ kind: "err", message: `${e.status}: ${detail}` });
          } else {
            out.push({
              kind: "err",
              message: e instanceof Error ? e.message : String(e),
            });
          }
        }
      }
      diffs = out;
    } finally {
      loading = false;
    }
  }

  // Refetch whenever the id list changes.
  $effect(() => {
    void ids;
    void fetchAll();
  });

  onMount(() => {
    // Clear the multi-select once the drawer's data is in flight so the
    // sidebar's selection-bar disappears (user-feel: "ok, I've moved
    // them into the drawer").
    return () => {
      clearNodeSelection();
    };
  });

  // ----------------------------------------------- node previews ---

  function nodePreview(id: string): string {
    const n = loomTree.nodes.get(id);
    const t = (n?.text ?? "").replace(/\s+/g, " ").trim();
    if (!t) return "(empty)";
    return t.length > 80 ? t.slice(0, 80) + "…" : t;
  }

  // ---------------------------------------- readings sort + top-N --

  function sortedReadings(rs: DiffReadingDeltaJSON[]): DiffReadingDeltaJSON[] {
    const arr = [...rs];
    if (sortBy === "magnitude") {
      arr.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
    } else {
      arr.sort((a, b) => a.name.localeCompare(b.name));
    }
    return arr;
  }

  function topKSet(rs: DiffReadingDeltaJSON[], k = 5): Set<string> {
    const sorted = [...rs].sort(
      (a, b) => Math.abs(b.delta) - Math.abs(a.delta),
    );
    return new Set(sorted.slice(0, k).map((r) => r.name));
  }

  // ------------------------------------- per-token hover alignment --

  /** Map from anchor-side token index to the aligned counterpart on
   * the primary diff (the 2nd column for N>=2 ids).  Used to
   * cross-highlight tokens between the two columns. */
  function alignByA(spans: DiffTokenSpanJSON[]): Map<number, DiffTokenSpanJSON> {
    const m = new Map<number, DiffTokenSpanJSON>();
    for (const sp of spans) {
      if (sp.a_index >= 0) m.set(sp.a_index, sp);
    }
    return m;
  }
  function alignByB(spans: DiffTokenSpanJSON[]): Map<number, DiffTokenSpanJSON> {
    const m = new Map<number, DiffTokenSpanJSON>();
    for (const sp of spans) {
      if (sp.b_index >= 0) m.set(sp.b_index, sp);
    }
    return m;
  }

  // For each diff we'll precompute alignment maps.  The primary diff
  // (index 0) drives the cross-column highlight.
  const primaryAlignA = $derived.by<Map<number, DiffTokenSpanJSON>>(() => {
    const d = diffs[0];
    return d?.kind === "ok"
      ? alignByA(d.diff.per_token)
      : new Map<number, DiffTokenSpanJSON>();
  });
  const primaryAlignB = $derived.by<Map<number, DiffTokenSpanJSON>>(() => {
    const d = diffs[0];
    return d?.kind === "ok"
      ? alignByB(d.diff.per_token)
      : new Map<number, DiffTokenSpanJSON>();
  });

  function formatReadingDelta(r: DiffReadingDeltaJSON): string {
    const sign = r.delta >= 0 ? "+" : "";
    return `${sign}${r.delta.toFixed(3)}`;
  }

  function deltaColor(delta: number): string {
    if (delta === 0) return "var(--fg-muted)";
    return delta >= 0 ? "var(--accent-green)" : "var(--accent-red)";
  }

  function spanColor(state: DiffTextSpanJSON["state"]): string {
    if (state === "insert") return "rgba(126, 231, 135, 0.18)";
    if (state === "delete") return "rgba(248, 81, 73, 0.18)";
    return "transparent";
  }

  function tokensFor(text: string): string[] {
    // For per-token rendering we re-split the text by the same shape
    // the engine uses (whitespace-token boundaries).  This isn't the
    // actual model tokenization but it gives a per-word hover affordance
    // matching the text-diff granularity.
    return text.split(/(\s+)/);
  }
</script>

<section class="drawer-shell" aria-label="Cross-branch diff drawer">
  <header class="header">
    <span class="title">compare branches</span>
    <div class="header-controls">
      <label class="header-ctl">
        <span>layout</span>
        <select bind:value={layout} aria-label="Layout">
          <option value="side-by-side">side-by-side</option>
          <option value="unified">unified</option>
        </select>
      </label>
      <label class="header-ctl">
        <span>sort by</span>
        <select bind:value={sortBy} aria-label="Sort readings by">
          <option value="magnitude">|Δ| desc</option>
          <option value="name">name</option>
        </select>
      </label>
    </div>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button>
  </header>

  <div class="body">
    {#if ids.length < 2}
      <p class="empty">
        Pick at least two assistant nodes via the sidebar's "select for
        compare" right-click action — or right-click a user node with
        ≥2 assistant children and pick "compare children".
      </p>
    {:else if loading && diffs.length === 0}
      <p class="empty">computing diff…</p>
    {:else}
      <!-- Column headers: anchor + each diff target. -->
      <div class="columns" class:unified={layout === "unified"}>
        <div class="col anchor-col">
          <header class="col-header">
            <code class="col-id">{anchorId?.slice(0, 12) ?? ""}</code>
            <span class="col-tag">anchor (A)</span>
          </header>
          <p class="col-preview">{nodePreview(anchorId ?? "")}</p>
        </div>
        {#each ids.slice(1) as otherId, otherIdx (otherId)}
          <div class="col">
            <header class="col-header">
              <code class="col-id">{otherId.slice(0, 12)}</code>
              <span class="col-tag">B{ids.length > 2 ? otherIdx + 1 : ""}</span>
            </header>
            <p class="col-preview">{nodePreview(otherId)}</p>
          </div>
        {/each}
      </div>

      {#each diffs as result, diffIdx (diffIdx)}
        {#if result.kind === "err"}
          <p class="error" role="alert">{result.message}</p>
        {:else}
          {@const d = result.diff}
          {@const topDeltas = topKSet(d.readings, 5)}
          {@const sortedRs = sortedReadings(d.readings)}

          <section class="diff-block">
            <header class="diff-header">
              <span class="diff-title">
                {ids.length === 2 ? "A vs B" : `A vs B${diffIdx + 1}`}
              </span>
              {#if d.parent_applied_steering !== null || d.steering_delta}
                <code class="recipe-delta" title="steering delta A → B">
                  Δ steering: {d.steering_delta || "(none)"}
                </code>
              {/if}
            </header>

            <!-- Text diff: side-by-side or unified. -->
            {#if layout === "side-by-side"}
              <div class="text-grid">
                <div class="text-pane">
                  <span class="pane-label">A</span>
                  <div class="text-body">
                    {#each tokensFor(d.a_text) as part, idx (idx)}
                      {@const aligned = primaryAlignA.get(idx)}
                      <!-- svelte-ignore a11y_no_static_element_interactions -->
                      <span
                        class="tok"
                        class:highlight-anchor={hoveredAnchorIdx === idx}
                        title={aligned && aligned.reading_deltas.length > 0
                          ? aligned.reading_deltas
                              .slice(0, 3)
                              .map((r) => `${r.name} ${formatReadingDelta(r)}`)
                              .join(" · ")
                          : ""}
                        onmouseenter={() => (hoveredAnchorIdx = idx)}
                        onmouseleave={() => (hoveredAnchorIdx = null)}
                      >{part}</span>
                    {/each}
                  </div>
                </div>
                <div class="text-pane">
                  <span class="pane-label">B{ids.length > 2 ? diffIdx + 1 : ""}</span>
                  <div class="text-body">
                    {#each tokensFor(d.b_text) as part, idx (idx)}
                      {@const aligned = primaryAlignB.get(idx)}
                      {@const matched =
                        diffIdx === 0 &&
                        hoveredAnchorIdx !== null &&
                        aligned?.a_index === hoveredAnchorIdx}
                      <span
                        class="tok"
                        class:highlight-target={matched}
                        title={aligned && aligned.reading_deltas.length > 0
                          ? aligned.reading_deltas
                              .slice(0, 3)
                              .map((r) => `${r.name} ${formatReadingDelta(r)}`)
                              .join(" · ")
                          : ""}
                      >{part}</span>
                    {/each}
                  </div>
                </div>
              </div>
            {:else}
              <div class="unified-body">
                {#each d.text as span, idx (idx)}
                  <span
                    class="tok-span"
                    class:span-equal={span.state === "equal"}
                    class:span-insert={span.state === "insert"}
                    class:span-delete={span.state === "delete"}
                    style={`background-color: ${spanColor(span.state)}`}
                  >
                    {#if span.state === "insert"}<span class="span-sign">+</span
                      >{:else if span.state === "delete"}<span class="span-sign"
                        >−</span
                      >{/if}{span.text}{" "}
                  </span>
                {/each}
              </div>
            {/if}

            <!-- Readings delta table. -->
            {#if sortedRs.length > 0}
              <div class="readings-table">
                <span class="readings-label">readings Δ (B − A)</span>
                <div class="readings-grid">
                  {#each sortedRs as r (r.name)}
                    {@const top = topDeltas.has(r.name)}
                    <div class="reading-row" class:top-delta={top}>
                      <span class="r-name">{r.name}</span>
                      <span class="r-vals">
                        <span class="r-side">{r.a_value.toFixed(3)}</span>
                        <span class="r-arrow">→</span>
                        <span class="r-side">{r.b_value.toFixed(3)}</span>
                      </span>
                      <div class="r-bar-track">
                        <div
                          class="r-bar-fill"
                          style={`width: ${Math.min(100, Math.abs(r.delta) * 100)}%; background: ${deltaColor(r.delta)}`}
                        ></div>
                      </div>
                      <span class="r-delta" style={`color: ${deltaColor(r.delta)}`}>
                        {formatReadingDelta(r)}
                      </span>
                    </div>
                  {/each}
                </div>
              </div>
            {:else}
              <p class="dim small">
                no readings recorded — either node has empty
                ``aggregate_readings``.
              </p>
            {/if}
          </section>
        {/if}
      {/each}
    {/if}
  </div>

  <footer class="footer">
    <button type="button" class="btn" onclick={closeDrawer}>close</button>
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
    font-size: var(--font-size-base);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6em;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
  }
  .title {
    color: var(--accent-blue);
    text-transform: lowercase;
    letter-spacing: 0.04em;
    flex: 0 0 auto;
  }
  .header-controls {
    display: flex;
    gap: 0.6em;
    flex: 1 1 auto;
    justify-content: center;
    color: var(--fg-muted);
    font-size: var(--font-size-small);
  }
  .header-ctl {
    display: inline-flex;
    align-items: center;
    gap: 0.35em;
  }
  .header-ctl select {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.15em 0.4em;
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--font-size-tiny);
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    cursor: pointer;
    padding: 0.25em 0.4em;
    font-size: 1em;
    line-height: 1;
  }
  .close:hover {
    color: var(--accent-red);
  }

  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 0.7em;
    min-height: 0;
  }
  .empty {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    margin: 0;
  }
  .error {
    color: var(--accent-red);
    font-size: var(--font-size-small);
    margin: 0;
  }

  .columns {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.6em;
  }
  .col {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    padding: 0.5em 0.6em;
  }
  .col-header {
    display: flex;
    justify-content: space-between;
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
    margin-bottom: 0.3em;
  }
  .col-id {
    color: var(--accent-yellow);
  }
  .col-tag {
    color: var(--accent-blue);
  }
  .col-preview {
    margin: 0;
    color: var(--fg);
    font-size: var(--font-size-small);
    line-height: 1.4;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
  }

  .diff-block {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    padding: 0.6em 0.8em;
    display: flex;
    flex-direction: column;
    gap: 0.6em;
  }
  .diff-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6em;
    flex-wrap: wrap;
  }
  .diff-title {
    color: var(--accent-blue);
    font-size: var(--font-size-small);
  }
  .recipe-delta {
    color: var(--accent-yellow);
    font-size: var(--font-size-tiny);
    background: var(--bg-elev);
    padding: 0.2em 0.4em;
    border-radius: 2px;
  }

  .text-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5em;
  }
  .columns.unified,
  .columns.unified > .col {
    grid-template-columns: 1fr;
  }
  .text-pane {
    background: var(--bg-alt);
    border: 1px solid var(--border);
    padding: 0.4em 0.5em;
    min-height: 5em;
    max-height: 30em;
    overflow-y: auto;
  }
  .pane-label {
    color: var(--accent-blue);
    font-size: var(--font-size-tiny);
    text-transform: lowercase;
    letter-spacing: 0.04em;
    display: block;
    margin-bottom: 0.2em;
  }
  .text-body {
    color: var(--fg);
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: var(--font-size-small);
  }
  .tok {
    cursor: default;
    transition: background-color 0.1s ease;
  }
  .tok:hover {
    background: rgba(88, 166, 255, 0.10);
  }
  .tok.highlight-anchor {
    background: rgba(88, 166, 255, 0.22);
  }
  .tok.highlight-target {
    background: rgba(210, 153, 34, 0.28);
  }

  .unified-body {
    background: var(--bg-alt);
    border: 1px solid var(--border);
    padding: 0.5em 0.6em;
    line-height: 1.5;
    font-size: var(--font-size-small);
    word-break: break-word;
    white-space: normal;
  }
  .tok-span {
    padding: 0 0.05em;
  }
  .span-sign {
    margin-right: 0.15em;
    font-weight: 600;
  }
  .span-insert .span-sign {
    color: var(--accent-green);
  }
  .span-delete .span-sign {
    color: var(--accent-red);
  }
  .span-equal {
    color: var(--fg-strong);
  }

  .readings-table {
    display: flex;
    flex-direction: column;
    gap: 0.3em;
  }
  .readings-label {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
    text-transform: lowercase;
    letter-spacing: 0.04em;
  }
  .readings-grid {
    display: flex;
    flex-direction: column;
    gap: 0.2em;
  }
  .reading-row {
    display: grid;
    grid-template-columns: minmax(8em, 12em) minmax(6em, 9em) 1fr 4em;
    align-items: center;
    gap: 0.4em;
    font-size: var(--font-size-small);
  }
  .reading-row.top-delta .r-name {
    color: var(--accent-yellow);
    font-weight: 600;
  }
  .r-name {
    color: var(--fg-strong);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .r-vals {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
    display: inline-flex;
    gap: 0.25em;
    align-items: center;
  }
  .r-arrow {
    color: var(--fg-subtle);
  }
  .r-bar-track {
    background: var(--bg-elev);
    height: 6px;
    border-radius: 1px;
    position: relative;
    overflow: hidden;
  }
  .r-bar-fill {
    height: 100%;
  }
  .r-delta {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .dim {
    color: var(--fg-muted);
  }
  .small {
    font-size: var(--font-size-small);
  }

  .footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.5em;
    padding: 12px 16px;
    border-top: 1px solid var(--border);
  }
  .btn {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.4em 0.9em;
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
  }
  .btn:hover {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
</style>
