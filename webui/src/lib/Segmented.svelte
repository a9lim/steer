<script lang="ts">
  // Segmented control — a small shared mode toggle matching the a9l.im
  // site's animated-indicator toggle.  An absolutely-positioned bar slides
  // under the active segment on the --ease-out curve.
  //
  // Generic over string-valued options; consumers pass ``onChange`` and
  // read the new value there (the indicator tracks ``value``).

  interface Option {
    value: string;
    label: string;
    title?: string;
  }

  interface Props {
    options: Option[];
    /** Currently-selected option value. */
    value: string;
    onChange?: (value: string) => void;
    disabled?: boolean;
    ariaLabel?: string;
  }

  let {
    options,
    value = $bindable(),
    onChange,
    disabled = false,
    ariaLabel,
  }: Props = $props();

  const activeIdx = $derived(
    Math.max(0, options.findIndex((o) => o.value === value)),
  );

  function pick(v: string): void {
    if (disabled || v === value) return;
    value = v;
    onChange?.(v);
  }
</script>

<div
  class="seg"
  class:disabled
  role="radiogroup"
  aria-label={ariaLabel}
  style:--seg-count={options.length}
>
  <span
    class="indicator"
    aria-hidden="true"
    style:transform="translateX({activeIdx * 100}%)"
  ></span>
  {#each options as opt (opt.value)}
    <button
      type="button"
      role="radio"
      aria-checked={opt.value === value}
      class:active={opt.value === value}
      {disabled}
      title={opt.title}
      onclick={() => pick(opt.value)}
    >
      {opt.label}
    </button>
  {/each}
</div>

<style>
  .seg {
    position: relative;
    display: inline-grid;
    grid-auto-flow: column;
    grid-auto-columns: 1fr;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg);
    padding: 2px;
    isolation: isolate;
  }
  .seg.disabled {
    opacity: 0.55;
  }

  /* The sliding indicator — width is one segment, position tracks the
   * active index on the site's --ease-out curve. */
  .indicator {
    position: absolute;
    z-index: -1;
    top: 2px;
    bottom: 2px;
    left: 2px;
    width: calc((100% - 4px) / var(--seg-count));
    background: var(--secondary-subtle);
    border: 1px solid var(--accent-blue);
    border-radius: var(--radius);
    transition: transform var(--dur) var(--ease-out);
  }

  .seg button {
    background: transparent;
    border: 0;
    padding: 0.2em 0.7em;
    color: var(--fg-dim);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    line-height: 1.4;
    white-space: nowrap;
    cursor: pointer;
    transition: color var(--dur) var(--ease-out);
  }
  .seg button:hover:not(:disabled):not(.active) {
    color: var(--fg-strong);
  }
  .seg button.active {
    color: var(--accent-blue);
  }
  .seg button:disabled {
    cursor: not-allowed;
  }
</style>
