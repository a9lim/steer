<script lang="ts">
  // Connector between a parent node and a child node.  Phase 5: when
  // ``parentId`` and ``childId`` are provided, fetch the canonical
  // steering-delta label lazily from the server (cached in the store
  // per `parent|child` key) and render it inline.

  import { fetchEdgeLabel, loomTree } from "../../lib/stores.svelte";

  interface Props {
    /** Active-path membership — bold the line so the user can trace
     *  the chosen branch downward. */
    active?: boolean;
    /** Dead branch — drop opacity in lockstep with the child. */
    dead?: boolean;
    /** Steering-delta label endpoints — when both are set the
     *  component fetches the label from the server lazily and
     *  caches the result. */
    parentId?: string | null;
    childId?: string | null;
    /** Manual label override — phase-3 callers can pass a static
     *  string; phase-5 fetches lazily when this is unset. */
    label?: string | null;
    /** Logit-pass: child's per-turn ``mean_logprob``.  Null when capture
     *  wasn't live; the edge then renders flat regardless of mode. */
    weight?: number | null;
    /** Logit-pass: ``"none"`` / ``"confidence"`` / ``"surprise"`` — see
     *  ``loomUiState.weightMode``. */
    weightMode?: "none" | "confidence" | "surprise";
  }

  let {
    active = false,
    dead = false,
    parentId = null,
    childId = null,
    label = null,
    weight = null,
    weightMode = "none",
  }: Props = $props();

  /** Logit-pass: map ``mean_logprob`` (≤ 0) to a [0, 1] intensity for
   *  edge stroke-width / opacity scaling.
   *
   *  confidence = 1 / (1 - logprob)   # logprob → 0 ⇒ 1; logprob → -∞ ⇒ 0
   *  surprise   = 1 - confidence
   *
   *  Both branches return ``null`` when the input is missing — caller
   *  then renders flat (today's shape). */
  const intensity = $derived.by<number | null>(() => {
    if (weightMode === "none") return null;
    if (weight == null || !Number.isFinite(weight) || weight > 0) return null;
    const conf = 1 / (1 - weight);
    return weightMode === "confidence" ? conf : 1 - conf;
  });

  /** Scale the line's CSS variables.  Width grows from 1px → 3px,
   *  opacity from 0.35 → 1.  Active edges still get the green-accent
   *  override below from the class; weighting just modulates magnitude. */
  const lineStyle = $derived.by<string>(() => {
    if (intensity === null) return "";
    const width = 1 + 2 * intensity;
    const opacity = 0.35 + 0.65 * intensity;
    return `width: ${width.toFixed(2)}px; opacity: ${opacity.toFixed(3)};`;
  });

  // Lazy fetch on mount + whenever the tree revision changes (cache
  // gets invalidated, so we need to re-request).  Skip when the
  // caller already passed an explicit ``label`` or when either
  // endpoint is missing.
  //
  // The fetched steering-delta label is *rendered by LoomNode* (as a
  // trailing chip in the node row) — not here.  An absolutely-positioned
  // label on this 1ch-wide edge column overlapped the node text.  This
  // component still owns the fetch so the cache stays populated; the
  // sidebar reads ``edgeLabelCache`` and forwards the label to the node.
  $effect(() => {
    void loomTree.rev;
    if (label !== null) return;
    if (!parentId || !childId) return;
    // Only fetch when the parent has ≥2 children — single-child
    // edges have no delta to render and we save a request.
    const siblings = loomTree.children_of.get(parentId) ?? [];
    if (siblings.length < 2) return;
    fetchEdgeLabel(parentId, childId);
  });
</script>

<div
  class="edge"
  class:active
  class:dead
  class:weighted={intensity !== null}
  aria-hidden="true"
>
  <span class="line" style={lineStyle}></span>
</div>

<style>
  .edge {
    position: relative;
    width: 1ch;
    flex: 0 0 auto;
    display: flex;
    align-items: stretch;
    justify-content: center;
    pointer-events: none;
  }
  .line {
    width: 1px;
    background: var(--border);
    height: 100%;
  }
  .edge.active .line {
    background: var(--accent-green);
    width: 2px;
  }
  .edge.dead {
    opacity: 0.3;
  }
</style>
