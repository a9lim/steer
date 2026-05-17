<script lang="ts">
  import SamplingStrip from "./SamplingStrip.svelte";
  import SteeringRack from "./SteeringRack.svelte";
  import ProbeRack from "./ProbeRack.svelte";
  import {
    genStatus,
    geometricMeanPpl,
    loomTree,
    sessionState,
  } from "../lib/stores.svelte";

  const nodeCount = $derived(loomTree.nodes.size);
  const ppl = $derived(geometricMeanPpl(genStatus));
  const model = $derived(sessionState.info?.model_id ?? "no session");
  const device = $derived(
    sessionState.info
      ? `${sessionState.info.device}/${sessionState.info.dtype}`
      : "offline",
  );
</script>

<aside class="inspector" aria-label="Saklas inspector">
  <section class="overview">
    <div>
      <p class="eyebrow">active workbench</p>
      <h2 title={model}>{model}</h2>
      <p class="sub">{device}</p>
    </div>
    <div class="meters" aria-label="runtime meters">
      <div class="meter">
        <span>tok/s</span>
        <strong>{genStatus.tokPerSec ? genStatus.tokPerSec.toFixed(1) : "—"}</strong>
      </div>
      <div class="meter">
        <span>ppl</span>
        <strong>{ppl === null ? "—" : ppl.toFixed(2)}</strong>
      </div>
      <div class="meter">
        <span>tree</span>
        <strong>{nodeCount || "—"}</strong>
      </div>
    </div>
  </section>

  <div class="sampling-wrap">
    <SamplingStrip />
  </div>

  <div class="rack-grid">
    <SteeringRack />
    <ProbeRack />
  </div>
</aside>

<style>
  .inspector {
    display: grid;
    grid-template-rows: auto auto minmax(0, 1fr);
    gap: 0.7rem;
    padding: 0.75rem;
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
    background: var(--bg-alt);
  }

  .overview {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 0.75rem;
    padding: 0.75rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
  }

  .eyebrow,
  .sub,
  .meter span {
    margin: 0;
    color: var(--fg-muted);
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0;
  }

  h2 {
    margin: 0.15rem 0;
    font-family: var(--font-ui);
    font-size: 0.95rem;
    line-height: 1.2;
    color: var(--fg);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    letter-spacing: 0;
  }

  .meters {
    display: grid;
    grid-template-columns: repeat(3, minmax(3.1rem, auto));
    gap: 0.35rem;
  }

  .meter {
    display: grid;
    gap: 0.1rem;
    min-width: 3.1rem;
    padding: 0.45rem 0.5rem;
    border: 1px solid var(--border-dim);
    border-radius: var(--radius);
    background: rgba(255, 255, 255, 0.025);
  }

  .meter strong {
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 0.9rem;
  }

  .sampling-wrap {
    min-width: 0;
  }

  .rack-grid {
    display: grid;
    grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
    gap: 0.65rem;
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
  }

  :global(.sampling-wrap .sampling-strip) {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    padding: 0.6rem;
  }
</style>
