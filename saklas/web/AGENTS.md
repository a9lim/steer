# web/

Static Svelte 5 + Vite dashboard mounted at `/` by `saklas serve`. CLI default is on (`--no-web` opts out); `create_app(..., web=False)` is the library default so embedded API surfaces don't pick up the dashboard.

## Layout

```
saklas/web/
  __init__.py        # re-exports: register_web_routes, dist_path
  routes.py          # mount logic + SPA fallback
  dist/              # COMMITTED build artifact, ships in the wheel
    index.html
    assets/index.css
    assets/saklas.js
```

The Svelte source lives at the repo's `webui/` directory (peer of `saklas/`). `cd webui && npm run build` emits straight to `saklas/web/dist/` — no intermediate copy. The committed bundle is the source of truth; CI source-vs-dist drift gating is wired but disabled by default.

## Mount

`register_web_routes(app)` mounts `/assets/*` on `StaticFiles` (content-hashed, safe to cache hard), registers `GET /` → `index.html`, and a catch-all `GET /{full_path:path}` that serves allowlisted top-level dist files (favicon, etc.) and otherwise falls back to `index.html` for SPA routing. The catch-all is registered last by `create_app` so it never shadows `/v1/*`, `/api/*`, `/saklas/v1/*`. `full_path` is only ever used as a dict key, never a path component — `..` traversal is structurally impossible.

`dist_path()` resolves through `importlib.resources` (editable + wheel). `WebUINotBuilt` is raised on mount when the dist directory is empty — only fires in source installs that haven't run `npm run build`.

## Wire protocol

The dashboard speaks the native `/saklas/v1/*` API (`saklas/server/saklas_api.py`):

- **WS `/saklas/v1/sessions/{id}/stream`** — token + probe co-stream. With probes loaded, the `token` event carries `scores` (`dict[str, float]`, the magnitude-weighted `score_single_token` aggregate the TUI also tints with), `per_layer_scores` (`dict[str, dict[str, float]]`, string-keyed, feeds the token-drilldown heatmap), and `raw_index` (decode-step index into the backing node's `raw_token_ids`). `done` carries `per_token_probes`. A `generate` message with `fork_node_id`/`fork_raw_index`/`fork_alt_token_id` is the **logit fork** — replays the node's raw decode prefix with one token swapped, resampling the continuation as a sibling. A `generate` with `prefill_node_id`/`prefill_text` is **answer-prefill** — seeds an assistant reply under a *user* node.
- **GET `/sessions/{id}/correlation[?names=…]`** — N×N magnitude-weighted cosine matrix; default pool unions steering vectors + active probes.
- **GET `/sessions/{id}/vectors/{name}/diagnostics`** — 16-bucket layer-magnitude histogram + per-layer magnitudes; falls back to monitor profiles when `name` is a probe.
- **GET `/packs`**, **GET `/packs/search?q=…`**, **POST `/packs`** — installed packs, HF Hub search proxy, install.
- **POST `/sessions/{id}/vectors/merge`**, **POST `/sessions/{id}/vectors/clone`** (SSE progress on `Accept: text/event-stream`), **POST `/sessions/{id}/extract`** (SSE).
- **POST `/sessions/{id}/experiments/fan`** — alpha grid as loom siblings, JSON `RunSet` summary.
- **Loom tree** under `/sessions/{id}/tree` — `tree`/`tree/active` GETs; `navigate`/`edit`/`branch`/`delete`/`star`/`note`/`reset` mutations; `edge_label`, `filter`, `diff`, `joint_logprobs`; `transcript` export/import.
- **GET `/sessions/{id}/traits/stream`** — live per-token probe SSE.

## Source layout

```
webui/src/
  main.ts                     # bootstrap: mounts <App /> via Svelte 5 mount()
  App.svelte                  # shell + drawer switch; NARROW_DRAWERS size class
  lib/
    api.ts                    # typed REST + WS + SSE clients
    stores.svelte.ts          # runes-based shared state + cross-cutting WS/tree state
    stores/                   # split slices: drawers, inputHistory, toasts (.svelte.ts)
    types.ts                  # shared interfaces; DrawerName union
    expression.ts             # parse/serialize the steering grammar
    concepts.ts               # concept-catalog helpers (category / poles / recommended α)
    tokens.ts                 # HIGHLIGHT_SAT + scoreToRgb + twoStripeStyle
    charts.ts                 # bucketize() port of saklas.core.histogram
    charts/{Bar,Sparkline,Histogram,HeatmapCell}.svelte
    Segmented.svelte          # shared segmented control
    Slider.svelte             # shared range slider
    Toaster.svelte            # toast host (bottom-right, TTL-dismissed)
    style/{tokens.css,global.css}
  panels/
    Topbar.svelte             # brand + session status + pending-actions badge
    WorkspaceRail.svelte      # left rail: loom toggle + category fly-outs
    BranchCanvas.svelte       # active path + sibling/child lanes + fan/compare controls
    InspectorPanel.svelte     # runtime meters + sampling + steering/probe racks
    StatusFooter.svelte       # gen progress · t/s · elapsed · ppl
    Chat.svelte               # thinking-collapsible, probe-tinted tokens, inline actions
    SamplingStrip.svelte      # T / P / K / max / seed / thinking + apply-mode
    SteeringRack.svelte       # vector strips + "+ add steering" + canonical expression
    VectorStrip.svelte        # enable + α slider + trigger + variant + projection modal
    ProbeRack.svelte          # probe strips + sort + "+ add probe"
    ProbeStrip.svelte         # select-for-highlight + sparkline + per-layer reading strip
    loom/{LoomSidebar,LoomNode,LoomEdge}.svelte
  drawers/
    {Load,SaveConversation,LoadConversation,Compare,SystemPrompt,ModelInfo,
     Help,Export,Pack,Merge,Clone,VectorPicker,ProbePicker,TokenDrilldown,
     ExperimentLab,ActivationAtlas,RecipeBuilder,AdvancedSampling,Health,
     SessionAdmin,Correlation,LayerNorms,NodeCompare,Transcript}Drawer.svelte
    _SearchableConceptList.svelte  # shared categorized catalog for both pickers
    index.ts                  # barrel re-exports for App.svelte's switch
```

(`lib/stores.ts` is a dead legacy file — not imported anywhere; ignore it.)

Adding a panel: write the `.svelte`, wire state into the smallest matching `lib/stores/` slice (or `stores.svelte.ts` for cross-cutting WS/tree/chat state), mount from `App.svelte`, `npm run build`, commit the regenerated `dist/`. Adding a drawer: write it under `drawers/`, add the name to the `DrawerName` union in `lib/types.ts` (and to `NARROW_DRAWERS` in `App.svelte` for forms/pickers), add an `App.svelte` switch branch, re-export from `drawers/index.ts`, and add it to a `WorkspaceRail.svelte` category fly-out.

## Reactivity gotcha

Svelte 5's `$state` does NOT track `Map.set` / `Set.add` / inner-object property writes inside collections. Cross-component collections in `stores.svelte.ts` use `SvelteMap` / `SvelteSet` from `svelte/reactivity`. Inner-object mutations on map values are still untracked, so every rack mutator reassigns: `entries.set(name, {...e, alpha})` — vectorRack and probeRack alike. `updateProbeFromScores` (driven by every WS `token` event) is the hot path here — a bare `entry.current = val` would freeze probe sparklines at zero through a whole generation.

## Persistence

The server loom tree is authoritative. The browser keeps a first-paint cache of the latest `LoomTreeJSON` plus `highlightState` in `localStorage` under `saklas.chat.v2.<model_id>`; v1 flat `ChatTurn[]` logs auto-migrate. Saves are debounced ~250 ms after mutations. `refreshLoomTree()` overwrites the cache with server state once the tree endpoint responds. `schedulePersist` measures payload size against a 5 MB soft budget and fires a once-per-session advisory toast (suggesting transcript export + tree clear) above it — the write isn't hard-stopped. `pendingIndex` is force-cleared on restore so an in-flight turn from a killed tab can't ghost the UI.

## Per-token highlighting

Highlighting lives on the chat token spans, driven by a single highlight-probe dropdown in the chat header with an optional two-stripe compare-two mode. It tints **live** as tokens stream: the WS `token` event's `scores` aggregate feeds the same `scoreToRgb` ramp the post-generation pass uses, so streaming and finalized tints match (and match the TUI). Clicking any token opens the `token_drilldown` drawer with the per-layer × per-probe heatmap regardless of whether a highlight probe is selected.

## Out of scope

- True multi-session switching — server URL-paths support it; the client still assumes `default`. `SessionAdminDrawer` inspects the collection and sets an in-memory bearer key but is not a session router.
- Persistent credential management — the bearer key stays in memory for the page session, never written to `localStorage`.
- Mobile / touch-first layout — desktop research tool, min-width 1280px.
- Combobox autocomplete on the projection-target picker (free-form name input).
- Pagination on HF pack search (capped at 20 results).
