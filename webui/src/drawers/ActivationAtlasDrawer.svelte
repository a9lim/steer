<script lang="ts">
  import { closeDrawer, chatLog, highlightState, probeRack } from "../lib/stores.svelte";
  import type { TokenScore } from "../lib/types";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let selectedIdx = $state(0);

  interface AtlasToken {
    idx: number;
    turn: number;
    token: number;
    text: string;
    score: TokenScore;
  }

  const tokens = $derived.by(() => {
    const out: AtlasToken[] = [];
    for (let turn = 0; turn < chatLog.turns.length; turn++) {
      const t = chatLog.turns[turn];
      if (t.role !== "assistant") continue;
      for (let token = 0; token < (t.tokens ?? []).length; token++) {
        const score = t.tokens![token];
        out.push({ idx: out.length, turn, token, text: score.text, score });
      }
    }
    return out;
  });

  const selected = $derived(tokens[selectedIdx] ?? tokens[0] ?? null);
  const layerKeys = $derived.by(() =>
    Object.keys(selected?.score.perLayerScores ?? {}).sort((a, b) => Number(a) - Number(b)),
  );
  const probeKeys = $derived.by(() => {
    const names = new Set<string>(probeRack.active);
    for (const row of Object.values(selected?.score.perLayerScores ?? {})) {
      for (const k of Object.keys(row)) names.add(k);
    }
    return [...names].sort();
  });

  const logprobStats = $derived.by(() => {
    const vals = tokens
      .map((t) => t.score.logprob)
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    if (vals.length === 0) return null;
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    return { count: vals.length, mean, min: Math.min(...vals), max: Math.max(...vals) };
  });

  function cell(layer: string, probe: string): number | null {
    const v = selected?.score.perLayerScores?.[layer]?.[probe];
    return typeof v === "number" ? v : null;
  }

  function heat(v: number | null): string {
    if (v === null) return "background: rgba(255,255,255,0.025);";
    const t = Math.max(-1, Math.min(1, v));
    const color = t >= 0 ? "69, 211, 211" : "255, 118, 117";
    return `background: rgba(${color}, ${0.12 + Math.abs(t) * 0.58});`;
  }

  function surpriseWidth(tok: TokenScore): string {
    if (typeof tok.logprob !== "number") return "width: 4%;";
    const surprise = Math.min(1, Math.max(0, -tok.logprob / 12));
    return `width: ${Math.max(4, surprise * 100)}%;`;
  }
</script>

<section class="drawer-shell" aria-label="Activation atlas drawer">
  <header class="header">
    <div>
      <span class="title">activation atlas</span>
      <p>token × layer × probe inspection across the active conversation</p>
    </div>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}>×</button>
  </header>

  <div class="body">
    <section class="timeline">
      <div class="section-head">
        <h3>token timeline</h3>
        <span>{tokens.length} response tokens</span>
      </div>
      {#if tokens.length === 0}
        <div class="empty">generate with probes or top alternatives enabled to populate the atlas</div>
      {:else}
        <div class="token-grid">
          {#each tokens as tok (tok.idx)}
            <button
              type="button"
              class:selected={tok.idx === selectedIdx}
              onclick={() => (selectedIdx = tok.idx)}
              title={`turn ${tok.turn}, token ${tok.token}, logprob ${tok.score.logprob ?? "—"}`}
            >
              <span class="spark" style={surpriseWidth(tok.score)}></span>
              <code>{tok.text}</code>
            </button>
          {/each}
        </div>
      {/if}
    </section>

    <section class="summary">
      <div class="stat">
        <span>highlight</span>
        <strong>{highlightState.target ?? "off"}</strong>
      </div>
      <div class="stat">
        <span>probes</span>
        <strong>{probeRack.active.length}</strong>
      </div>
      <div class="stat">
        <span>logprob mean</span>
        <strong>{logprobStats ? logprobStats.mean.toFixed(2) : "—"}</strong>
      </div>
      <div class="stat">
        <span>selected</span>
        <strong>{selected ? `T${selected.turn}:${selected.token}` : "—"}</strong>
      </div>
    </section>

    <section class="heatmap-panel">
      <div class="section-head">
        <h3>layer/probe heatmap</h3>
        <span>{selected ? JSON.stringify(selected.text) : "no token selected"}</span>
      </div>
      {#if selected && layerKeys.length > 0 && probeKeys.length > 0}
        <div class="heatmap-wrap">
          <table class="heatmap">
            <thead>
              <tr>
                <th>layer</th>
                {#each probeKeys as probe (probe)}
                  <th title={probe}>{probe}</th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each layerKeys as layer (layer)}
                <tr>
                  <th>L{layer}</th>
                  {#each probeKeys as probe (probe)}
                    {@const v = cell(layer, probe)}
                    <td style={heat(v)} title={`${probe} L${layer}: ${v ?? "—"}`}>
                      {v === null ? "—" : v.toFixed(2)}
                    </td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {:else}
        <div class="empty">no per-layer probe scores on the selected token</div>
      {/if}
    </section>

    <section class="logits">
      <div class="section-head">
        <h3>distribution lens</h3>
        <span>{selected?.score.topAlts?.length ?? 0} alternatives</span>
      </div>
      {#if selected?.score.topAlts?.length}
        <div class="alts">
          {#each selected.score.topAlts as alt (alt.id)}
            <div class="alt-row">
              <code>{JSON.stringify(alt.text)}</code>
              <span>#{alt.id}</span>
              <strong>{alt.logprob.toFixed(3)}</strong>
            </div>
          {/each}
        </div>
      {:else}
        <div class="empty">enable top alternatives in advanced sampling to see token counterfactuals</div>
      {/if}
    </section>
  </div>
</section>

<style>
  .drawer-shell { display: flex; flex-direction: column; min-height: 0; background: var(--bg-alt); }
  .header { display: flex; justify-content: space-between; gap: var(--space-6); padding: var(--space-6) var(--space-6); border-bottom: 1px solid var(--border); background: var(--surface); }
  .title { color: var(--accent); text-transform: uppercase; letter-spacing: 0; font-size: var(--text-xs); font-weight: var(--weight-bold); }
  .header p { margin: var(--space-2) 0 0; color: var(--fg-muted); }
  .close { background: transparent; border: 0; color: var(--fg-muted); font-size: var(--text-md); }
  .body { display: grid; grid-template-columns: minmax(18rem, 0.72fr) minmax(24rem, 1fr); grid-template-rows: auto 1fr; gap: var(--space-4); padding: var(--space-6); overflow: auto; }
  .timeline, .summary, .heatmap-panel, .logits { border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); padding: var(--space-5); min-width: 0; }
  .timeline { grid-row: span 2; display: grid; grid-template-rows: auto minmax(0, 1fr); min-height: 0; }
  .summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--space-3); }
  .logits { grid-column: 2; }
  .section-head { display: flex; align-items: center; justify-content: space-between; gap: var(--space-4); margin-bottom: var(--space-4); }
  h3 { margin: 0; color: var(--fg); font-size: var(--text); letter-spacing: 0; }
  .section-head span, .stat span { color: var(--fg-muted); font-size: var(--text-xs); text-transform: uppercase; letter-spacing: 0; }
  .token-grid { display: flex; flex-wrap: wrap; gap: var(--space-2); align-content: flex-start; overflow: auto; padding-right: var(--space-2); }
  .token-grid button { position: relative; max-width: 11rem; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-elev); color: var(--fg); padding: var(--space-2) var(--space-3); overflow: hidden; }
  .token-grid button.selected { border-color: var(--accent); background: var(--accent-subtle); }
  .spark { position: absolute; inset: auto auto 0 0; height: 2px; background: var(--accent-amber); opacity: 0.8; }
  code { position: relative; font-family: var(--font-mono); white-space: pre-wrap; }
  .stat { border: 1px solid var(--border); border-radius: var(--radius); padding: var(--space-4); display: grid; gap: var(--space-1); background: var(--bg-elev); }
  .stat strong { color: var(--accent); font-family: var(--font-mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .heatmap-wrap { overflow: auto; border: 1px solid var(--border); border-radius: var(--radius); max-height: 27rem; }
  .heatmap { width: 100%; border-collapse: collapse; min-width: 30rem; }
  th, td { border: 1px solid var(--border); padding: var(--space-3) var(--space-3); text-align: right; font-family: var(--font-mono); font-size: var(--text-xs); }
  th { color: var(--fg-muted); background: var(--bg-elev); text-transform: uppercase; letter-spacing: 0; }
  td { color: var(--fg); }
  .alts { display: grid; gap: var(--space-2); max-height: 14rem; overflow: auto; }
  .alt-row { display: grid; grid-template-columns: 1fr auto auto; gap: var(--space-4); align-items: center; border: 1px solid var(--border); border-radius: var(--radius); padding: var(--space-3) var(--space-3); }
  .alt-row span { color: var(--fg-muted); }
  .alt-row strong { color: var(--accent-amber); font-family: var(--font-mono); }
  .empty { display: grid; place-items: center; min-height: 9rem; color: var(--fg-muted); text-align: center; border: 1px solid var(--border); border-radius: var(--radius); padding: var(--space-6); }
</style>
