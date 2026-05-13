<script lang="ts">
  // Connector between a parent node and a child node.  Phase 5: when
  // ``parentId`` and ``childId`` are provided, fetch the canonical
  // steering-delta label lazily from the server (cached in the store
  // per `parent|child` key) and render it inline.

  import {
    edgeLabelCache,
    fetchEdgeLabel,
    loomTree,
  } from "../../lib/stores.svelte";

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
  }

  let {
    active = false,
    dead = false,
    parentId = null,
    childId = null,
    label = null,
  }: Props = $props();

  // Lazy fetch on mount + whenever the tree revision changes (cache
  // gets invalidated, so we need to re-request).  Skip when the
  // caller already passed an explicit ``label`` or when either
  // endpoint is missing.
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

  const resolvedLabel = $derived.by(() => {
    if (label !== null) return label;
    if (!parentId || !childId) return null;
    const key = `${parentId}|${childId}`;
    return edgeLabelCache.get(key) ?? null;
  });
</script>

<div
  class="edge"
  class:active
  class:dead
  aria-hidden="true"
>
  <span class="line"></span>
  {#if resolvedLabel}
    <span class="label" title="steering delta">{resolvedLabel}</span>
  {/if}
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
  .label {
    position: absolute;
    left: 1.2ch;
    top: 50%;
    transform: translateY(-50%);
    font-size: var(--font-size-tiny);
    color: var(--accent-yellow);
    white-space: nowrap;
    pointer-events: auto;
  }
</style>
