<script lang="ts">
  // Inspector — the right-hand control rack: the steering and probe
  // racks split the full column height.  The sampling strip moved to
  // the bottom of the threads column (below the workbench card), giving
  // the racks this column's whole budget.
  import SteeringRack from "./SteeringRack.svelte";
  import ProbeRack from "./ProbeRack.svelte";
</script>

<aside class="inspector" aria-label="Saklas inspector">
  <div class="rack-grid">
    <SteeringRack />
    <ProbeRack />
  </div>
</aside>

<style>
  .inspector {
    display: grid;
    grid-template-rows: minmax(0, 1fr);
    padding: 0.75rem;
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
    background: var(--bg-alt);
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
</style>
