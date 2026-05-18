<script lang="ts">
  // The steering rack — section header, one VectorStrip per loaded
  // vector (alphabetized for stable ordering across re-renders), the
  // +extract / +load action buttons, and the canonical steering
  // expression with click-to-copy + paste-edit modes.
  //
  // The rack Map is the source of truth; the expression below is a
  // derived view that round-trips through serialize/parseExpression.
  // Paste-edit replaces the rack wholesale with what the user typed —
  // any vectors not in the new expression are dropped from the rack
  // (they remain registered server-side and can be re-added via
  // +extract/+load).

  import VectorStrip from "./VectorStrip.svelte";
  import {
    vectorRack,
    currentSteeringExpression,
    openDrawer,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import {
    parseExpression,
    ExpressionParseError,
  } from "../lib/expression";

  // Reactive entries — sorted alphabetically by name for stable order.
  // The Map iteration order tracks insertion which makes the rack
  // visually jump around as vectors land out of order; sorting fixes it.
  const sortedEntries = $derived.by(() => {
    const arr = [...vectorRack.entries.entries()];
    arr.sort((a, b) => a[0].localeCompare(b[0]));
    return arr;
  });

  const count = $derived(sortedEntries.length);

  // ------------ canonical expression display ------------

  const expression = $derived(currentSteeringExpression());

  let copied = $state(false);
  let copyTimer: number | null = null;

  async function copyExpression(): Promise<void> {
    if (!expression) return;
    try {
      await navigator.clipboard.writeText(expression);
      copied = true;
      if (copyTimer !== null) window.clearTimeout(copyTimer);
      copyTimer = window.setTimeout(() => {
        copied = false;
        copyTimer = null;
      }, 1200);
    } catch {
      // Some browsers block clipboard from non-HTTPS / non-gesture
      // contexts — surface an in-app toast rather than a native prompt.
      pushToast("clipboard blocked — select the expression to copy", {
        kind: "warning",
      });
    }
  }

  // ------------ paste-edit ------------

  let editing = $state(false);
  let draft = $state("");
  let parseError = $state<string | null>(null);
  let textareaRef: HTMLTextAreaElement | null = $state(null);

  function startEdit(): void {
    draft = expression;
    parseError = null;
    editing = true;
    // Focus + select after the textarea mounts so the user can start
    // typing or paste over the current value immediately.
    queueMicrotask(() => {
      if (textareaRef) {
        textareaRef.focus();
        textareaRef.select();
      }
    });
  }

  function cancelEdit(): void {
    editing = false;
    draft = "";
    parseError = null;
  }

  /** Replace the rack wholesale from the typed expression.  Empty input
   * clears the rack — that's the canonical "no steering" state and
   * matches the unsteered serialize output. */
  function commitEdit(): void {
    const text = draft.trim();
    if (text === "") {
      vectorRack.entries.clear();
      editing = false;
      draft = "";
      parseError = null;
      return;
    }
    try {
      const next = parseExpression(text);
      // Replace in place so any reactive consumers tracking the same
      // Map reference re-render coherently.  Clearing first keeps the
      // semantics "expression is the rack" — vectors not in the new
      // expression vanish.
      vectorRack.entries.clear();
      for (const [name, entry] of next) {
        vectorRack.entries.set(name, entry);
      }
      editing = false;
      draft = "";
      parseError = null;
    } catch (e) {
      if (e instanceof ExpressionParseError) {
        parseError = e.message;
      } else if (e instanceof Error) {
        parseError = e.message;
      } else {
        parseError = String(e);
      }
    }
  }

  function onTextareaKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
      ev.preventDefault();
      commitEdit();
    } else if (ev.key === "Escape") {
      ev.preventDefault();
      cancelEdit();
    }
  }

  /** Blur fires before click on a sibling button — if the user is
   * heading for the cancel button, defer commit so cancel wins.  We
   * inspect ``relatedTarget`` (the element receiving focus) and bail
   * when it's still inside our editor block. */
  function onTextareaBlur(ev: FocusEvent): void {
    const next = ev.relatedTarget as Node | null;
    if (next instanceof Element && next.closest(".expression-block")) {
      // The user is moving focus to a sibling control inside the
      // expression block (e.g. the cancel button) — let that handler
      // decide what to do.
      return;
    }
    commitEdit();
  }
</script>

<section class="rack" aria-label="Steering rack">
  <header class="header">
    <div class="header-text">
      <span class="title">STEERING</span>
      <span class="subtitle">shape the response</span>
    </div>
    <span class="count" aria-live="polite">
      {count} vector{count === 1 ? "" : "s"}
    </span>
  </header>

  <div class="strips" class:is-empty={count === 0}>
    {#if count === 0}
      <div class="empty">
        <p class="empty-copy">
          Steering shapes how the model responds.
          Add a concept to begin.
        </p>
        <button
          type="button"
          class="add-steering"
          onclick={() => openDrawer("vector_picker")}
        >
          + add steering
        </button>
      </div>
    {:else}
      {#each sortedEntries as [name, entry] (name)}
        <VectorStrip {name} {entry} />
      {/each}
    {/if}
  </div>

  {#if count > 0}
    <div class="actions">
      <button
        type="button"
        class="add-steering"
        onclick={() => openDrawer("vector_picker")}
        title="Browse the concept catalog or extract a custom vector"
      >
        + add steering
      </button>
    </div>
  {/if}

  <div class="expression-block">
    <div class="expr-header">
      <span class="expr-label">active steering</span>
      {#if !editing}
        <button
          type="button"
          class="pencil"
          onclick={startEdit}
          aria-label="edit steering expression"
          title="edit / paste a steering expression"
        >
          ✎
        </button>
      {/if}
    </div>

    {#if editing}
      <textarea
        bind:this={textareaRef}
        class="expr-edit"
        bind:value={draft}
        onblur={onTextareaBlur}
        onkeydown={onTextareaKeydown}
        rows="2"
        spellcheck="false"
        autocomplete="off"
        placeholder="0.3 honest + 0.4 warm@after - 0.2 sycophantic|honest"
        aria-label="paste-edit steering expression"
      ></textarea>
      <div class="edit-hints">
        <span class="hint">⌘/Ctrl-Enter to apply · Esc to cancel · blur applies</span>
        <button type="button" class="cancel" onclick={cancelEdit}>cancel</button>
      </div>
      {#if parseError}
        <div class="error" role="alert">{parseError}</div>
      {/if}
    {:else if expression}
      <button
        type="button"
        class="expr-code"
        onclick={copyExpression}
        title={copied ? "copied" : "click to copy"}
        aria-label="copy steering expression"
      >
        <code>{expression}</code>
        <span class="copy-hint" class:active={copied}>
          {copied ? "copied" : "copy"}
        </span>
      </button>
    {:else}
      <div class="expr-empty">
        <code>(no active steering)</code>
      </div>
    {/if}
  </div>
</section>

<style>
  /* A flat section of the inspector panel — no border box, no own
   * background; the only chrome is the border-bottom hairline dividing
   * it from the probe section below.  Fixed chrome + one scrollable
   * middle.  This deliberately uses flex instead of grid: the generated
   * ``.strips`` element was able to grow past the rack in some viewport
   * sizes, hiding the apply-vector controls. */
  .rack {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-5);
    background: transparent;
    border-bottom: 1px solid var(--border);
    height: 100%;
    min-height: 0;
    max-height: 100%;
    overflow: hidden;
  }

  .header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-3);
    border-bottom: 1px solid var(--border);
    padding-bottom: var(--space-3);
  }
  .header-text {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
  }
  .title {
    font-weight: var(--weight-bold);
    color: var(--accent-blue);
    letter-spacing: 0;
    font-size: var(--text-sm);
    text-transform: uppercase;
  }
  .subtitle {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .count {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    flex: 0 0 auto;
  }

  /* Strips own the scroll — overflow at the rack level would push the
   * actions + expression block off-screen when vectors pile up. */
  .strips {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    flex: 1 1 0;
    min-height: 2.4rem;
    max-height: 100%;
    overflow-y: auto;
    padding-right: var(--space-1);
  }
  .strips.is-empty {
    align-items: center;
    justify-content: center;
  }
  /* First-run teaching state — one line of plain copy above the primary
   * action.  Replaces the bare "no active steering vectors". */
  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-5) var(--space-4);
    text-align: center;
  }
  .empty-copy {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.5;
    max-width: 28ch;
  }

  /* Anchored at the bottom of the rack (above the expression block).
   * Border-top mirrors the probe rack's actions row for visual symmetry. */
  .actions {
    flex: 0 0 auto;
    border-top: 1px solid var(--border);
    padding-top: var(--space-4);
  }
  /* Primary entry point — the one obvious way to add a steering vector. */
  .add-steering {
    width: 100%;
    background: var(--accent-subtle);
    color: var(--accent-blue);
    border: 1px solid var(--border);
    min-height: 2.1rem;
    padding: var(--space-4) var(--space-5);
    border-radius: var(--radius);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  .empty .add-steering {
    width: auto;
    min-width: 14em;
  }
  .add-steering:hover {
    background: var(--accent-glow);
  }

  .expression-block {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    border-top: 1px solid var(--border);
    padding-top: var(--space-4);
  }
  .expr-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .expr-label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    text-transform: uppercase;
    letter-spacing: 0;
  }
  .pencil {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: var(--text-sm);
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    line-height: 1;
  }
  .pencil:hover {
    color: var(--accent-blue);
    background: var(--bg-elev);
  }

  .expr-code {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    width: 100%;
    text-align: left;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-4) var(--space-3);
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
  }
  .expr-code:hover {
    border-color: var(--accent);
  }
  .expr-code code {
    flex: 1 1 auto;
    overflow-wrap: anywhere;
    word-break: break-all;
    color: var(--fg-strong);
    background: transparent;
  }
  .copy-hint {
    flex: 0 0 auto;
    font-size: var(--text-sm);
    color: var(--fg-muted);
  }
  .copy-hint.active {
    color: var(--accent-green);
  }

  .expr-empty {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-4) var(--space-3);
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .expr-empty code {
    background: transparent;
    color: var(--fg-muted);
  }

  .expr-edit {
    width: 100%;
    background: var(--bg-elev);
    color: var(--fg);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    padding: var(--space-4) var(--space-3);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    resize: vertical;
    min-height: 3em;
  }
  .expr-edit:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
  }

  .edit-hints {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .hint {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .cancel {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-muted);
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius);
    font-size: var(--text-sm);
  }
  .cancel:hover {
    color: var(--fg-strong);
    border-color: var(--border);
  }

  .error {
    color: var(--accent-error);
    background: rgba(248, 81, 73, 0.08);
    border: 1px solid var(--accent-red);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font-size: var(--text-sm);
    font-family: var(--font-mono);
  }
</style>
