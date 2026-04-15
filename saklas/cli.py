"""CLI entry point for saklas.

Top-level shape (v2 hard break):

    saklas tui <model> [...]
    saklas serve <model> [...]
    saklas pack {install,refresh,clear,rm,ls,search,merge,push,export,clone,extract} ...
    saklas config {show,validate} ...

There is no bare-TUI mode. ``saklas`` with no arguments prints help.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:multiprocessing.resource_tracker")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Model-loading args shared between `tui` and `serve`."""
    p.add_argument(
        "model",
        help="HuggingFace model ID or local path (e.g. google/gemma-2-9b-it)",
    )
    p.add_argument(
        "-q", "--quantize",
        choices=["4bit", "8bit"],
        default=None,
        help="Quantization mode (default: bf16/fp16)",
    )
    p.add_argument(
        "-d", "--device",
        default="auto",
        help="Device: auto (detect), cuda, mps, or cpu (default: auto)",
    )
    p.add_argument(
        "-p", "--probes",
        nargs="*",
        default=None,
        help="Probe categories: all, none, affect, epistemic, alignment, register, social_stance, cultural (default: all)",
    )


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", action="append", default=None, metavar="PATH",
                   help="Load setup YAML (repeatable; later overrides earlier)")
    p.add_argument("-s", "--strict", action="store_true",
                   help="With -c: fail hard on missing vectors")


def _resolve_probes(raw: list[str] | None) -> list[str]:
    from saklas.session import PROBE_CATEGORIES
    if raw is None or raw == ["all"]:
        return list(PROBE_CATEGORIES)
    if raw == ["none"] or raw == []:
        return []
    return raw


def _make_session(args: argparse.Namespace):
    from saklas.session import SaklasSession
    probe_categories = _resolve_probes(args.probes)
    return SaklasSession.from_pretrained(
        args.model, device=args.device, quantize=args.quantize,
        probes=probe_categories,
        system_prompt=getattr(args, "system_prompt", None),
        max_tokens=getattr(args, "max_tokens", 1024),
    )


def _print_model_info(session) -> None:
    info = session.model_info
    print(f"Architecture: {info['model_type']}")
    print(f"Layers: {info['num_layers']}, Hidden dim: {info['hidden_dim']}")
    print(f"VRAM: {info['vram_used_gb']:.1f} GB")
    print(f"Loaded {len(session.probes)} probes")


def _load_effective_config(args: argparse.Namespace):
    """Compose ~/.saklas/config.yaml + any -c files and stamp args in place.

    Returns the composed ConfigFile (poles pre-resolved). Sets:
      args.config_vectors, args.temperature, args.top_p, args.thinking,
      args.system_prompt, args.max_tokens, and args.model (if YAML supplied it).
    """
    from saklas.config_file import (
        ConfigFile, apply_flag_overrides, ensure_vectors_installed,
    )
    extras = [Path(p) for p in (getattr(args, "config", None) or [])]
    composed = ConfigFile.effective(extras, include_default=True)
    composed = apply_flag_overrides(
        composed,
        model=getattr(args, "model", None),
        temperature=None,
        top_p=None,
        max_tokens=None,
        system_prompt=None,
    )
    composed = composed.resolve_poles()
    if getattr(args, "model", None) is None:
        args.model = composed.model
    args.temperature = composed.temperature
    args.top_p = composed.top_p
    args.thinking = composed.thinking
    args.system_prompt = composed.system_prompt
    args.max_tokens = composed.max_tokens if composed.max_tokens is not None else 1024
    args.config_vectors = composed.vectors
    ensure_vectors_installed(composed, strict=getattr(args, "strict", False))
    return composed


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_PACK_VERBS: list[tuple[str, str]] = [
    ("install",   "Install a concept pack from HF or a local folder"),
    ("refresh",   "Re-pull concept(s) from their source"),
    ("clear",     "Delete per-model tensors for matched concepts"),
    ("rm",        "Fully remove a concept folder"),
    ("ls",        "List locally installed concept packs"),
    ("search",    "Search the HuggingFace hub for concept packs"),
    ("merge",     "Merge existing vectors into a new pack"),
    ("push",      "Push a concept pack to HF as a model repo"),
    ("export",    "Export a pack to an interchange format (gguf)"),
    ("clone",     "Clone a persona from a text corpus"),
    ("extract",   "Extract a steering vector for a concept"),
]


def _build_tui_parser(parser: argparse.ArgumentParser) -> None:
    # When a model supplies -c/--config pointing at a YAML with model: set,
    # the positional can be omitted. Handled in _run_tui via composed config.
    parser.add_argument("model", nargs="?", default=None,
                        help="HuggingFace model ID or local path")
    parser.add_argument("-q", "--quantize", choices=["4bit", "8bit"], default=None,
                        help="Quantization mode (default: bf16/fp16)")
    parser.add_argument("-d", "--device", default="auto",
                        help="Device: auto (detect), cuda, mps, or cpu")
    parser.add_argument("-p", "--probes", nargs="*", default=None,
                        help="Probe categories (default: all)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Default max generation tokens")
    _add_config_args(parser)


def _build_serve_parser(parser: argparse.ArgumentParser) -> None:
    _add_common_args(parser)
    parser.add_argument("-H", "--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("-P", "--port", type=int, default=8000, help="Bind port")
    parser.add_argument("-S", "--steer", action="append", default=[], metavar="NAME[:ALPHA]",
                        help="Pre-load a steering vector (repeatable)")
    parser.add_argument("-C", "--cors", action="append", default=[], metavar="ORIGIN",
                        help="CORS allowed origin (repeatable)")
    parser.add_argument("-k", "--api-key", default=None, metavar="KEY",
                        help="Require Bearer token auth; falls back to $SAKLAS_API_KEY")
    _add_config_args(parser)


# --- pack subtree --------------------------------------------------------

def _build_pack_install(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", help="<ns>/<concept>[@revision] or path to a concept folder")
    p.add_argument("-s", "--statements-only", action="store_true",
                   help="Keep statements.json only; drop any bundled tensors")
    p.add_argument("-a", "--as", dest="as_target", default=None, metavar="NS/NAME",
                   help="Relocate the installed pack under a different namespace/name")
    p.add_argument("-f", "--force", action="store_true",
                   help="Overwrite an existing installation")


def _build_pack_refresh(p: argparse.ArgumentParser) -> None:
    p.add_argument("selector", help="Selector or the literal 'neutrals'")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID",
                   help="Scope to one model's tensors")


def _build_pack_clear(p: argparse.ArgumentParser) -> None:
    p.add_argument("selector", help="Selector (name, tag:x, namespace:x, default, all)")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID",
                   help="Scope to one model's tensors only (default: all models)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip confirmation prompt on broad selectors")


def _build_pack_rm(p: argparse.ArgumentParser) -> None:
    p.add_argument("selector", help="Selector (name, tag:x, namespace:x, default, all)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Required for broad selectors (all, namespace:)")


def _build_pack_ls(p: argparse.ArgumentParser) -> None:
    p.add_argument("selector", nargs="?", default=None,
                   help="Optional selector (name, tag:x, namespace:x, default, all)")
    p.add_argument("-j", "--json", dest="json_output", action="store_true",
                   help="Emit machine-readable JSON instead of a table")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Include descriptions in the table output")


def _build_pack_search(p: argparse.ArgumentParser) -> None:
    p.add_argument("query", nargs="?", default="",
                   help="Search text (matched against HF model ids)")
    p.add_argument("-j", "--json", dest="json_output", action="store_true",
                   help="Emit machine-readable JSON instead of a table")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Include descriptions in the table output")


def _build_pack_push(p: argparse.ArgumentParser) -> None:
    p.add_argument("selector", help="Single concept selector (name or ns/name)")
    p.add_argument("-a", "--as", dest="as_target", default=None, metavar="OWNER/NAME")
    p.add_argument("-p", "--private", action="store_true")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    p.add_argument("-s", "--statements-only", action="store_true")
    p.add_argument("-n", "--no-statements", action="store_true")
    p.add_argument("-t", "--tag-version", action="store_true")
    p.add_argument("-d", "--dry-run", action="store_true")
    p.add_argument("-f", "--force", action="store_true")


def _build_pack_export(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="format", required=True)
    g = sub.add_parser("gguf", help="Export baked tensors to llama.cpp GGUF")
    g.add_argument("selector", help="Single concept selector (name or ns/name)")
    g.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    g.add_argument("-o", "--output", default=None, metavar="PATH")
    g.add_argument("--model-hint", default=None, metavar="HINT")


def _build_pack_clone(p: argparse.ArgumentParser) -> None:
    p.add_argument("corpus_path", help="Path to a UTF-8 text file, one utterance per line")
    p.add_argument("-N", "--name", required=True, help="Persona identifier (stored under local/<name>)")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    p.add_argument("-n", "--n-pairs", dest="n_pairs", type=int, default=90)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-f", "--force", action="store_true")
    p.set_defaults(quantize=None, device="auto", probes=None)


def _build_pack_extract(p: argparse.ArgumentParser) -> None:
    p.add_argument("concept", nargs="+",
                   help="Either one concept (e.g. 'happy.sad') or two poles (e.g. 'happy' 'sad')")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID")
    p.add_argument("-f", "--force", action="store_true")
    p.set_defaults(quantize=None, device="auto", probes=None)


def _build_pack_merge(p: argparse.ArgumentParser) -> None:
    p.add_argument("name", help="New pack name (written under local/)")
    p.add_argument("components", help="Comma-separated components: ns/a:0.3,ns/b:0.4")
    p.add_argument("-f", "--force", action="store_true")
    p.add_argument("-s", "--strict", action="store_true")
    p.add_argument("-m", "--model", default=None, metavar="MODEL_ID")


_PACK_BUILDERS = {
    "install": _build_pack_install,
    "refresh": _build_pack_refresh,
    "clear":   _build_pack_clear,
    "rm":      _build_pack_rm,
    "ls":      _build_pack_ls,
    "search":  _build_pack_search,
    "merge":   _build_pack_merge,
    "push":    _build_pack_push,
    "export":  _build_pack_export,
    "clone":   _build_pack_clone,
    "extract": _build_pack_extract,
}


def _build_pack_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="pack_cmd", required=False, metavar="VERB")
    for verb, desc in _PACK_VERBS:
        child = sub.add_parser(verb, help=desc, description=desc)
        _PACK_BUILDERS[verb](child)


# --- config subtree ------------------------------------------------------

def _build_config_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="config_cmd", required=False, metavar="VERB")

    show = sub.add_parser("show", help="Print the effective merged config")
    show.add_argument("-c", "--config", action="append", default=None, metavar="PATH",
                      help="Extra YAML files to compose on top of ~/.saklas/config.yaml")
    show.add_argument("-m", "--model", default=None,
                      help="Override model field in output")
    show.add_argument("--no-default", action="store_true",
                      help="Skip loading ~/.saklas/config.yaml")

    validate = sub.add_parser("validate", help="Validate a config file (CI hook)")
    validate.add_argument("file", help="Path to YAML config file")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _build_root_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="saklas",
        description="Activation steering + trait monitoring for local HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "top-level verbs:\n"
            "  tui      Launch the interactive TUI (requires <model>)\n"
            "  serve    Start the OpenAI + Ollama compatible API server\n"
            "  pack     Manage concept packs (install/ls/search/extract/...)\n"
            "  config   Inspect and validate saklas config files\n"
            "\n"
            "Run `saklas <verb> -h` for verb-specific options."
        ),
    )
    sub = root.add_subparsers(dest="command", required=False, metavar="VERB")

    tui = sub.add_parser("tui", help="Launch the interactive TUI", description="Launch the interactive TUI")
    _build_tui_parser(tui)

    serve = sub.add_parser("serve", help="Start the API server", description="Start the API server")
    _build_serve_parser(serve)

    pack = sub.add_parser("pack", help="Manage concept packs", description="Manage concept packs")
    _build_pack_parser(pack)

    cfg = sub.add_parser("config", help="Inspect/validate config", description="Inspect/validate config")
    _build_config_parser(cfg)

    return root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_root_parser()
    # Zero-arg: print help+hint and exit 0 (not argparse's exit 2).
    if not argv:
        parser.print_help()
        print()
        print("try 'saklas tui <model_id>' or 'saklas --help'")
        sys.exit(0)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _print_startup(args: argparse.Namespace) -> None:
    print(f"Loading model: {args.model}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")


def _setup_steering_vectors(
    session,
    vector_specs,
    *,
    verbose: bool = False,
) -> dict[str, float]:
    """Resolve pole aliases, extract profiles, register with session.

    vector_specs is either a dict[coord, alpha] or an iterable of
    ``(raw, alpha)`` tuples. Returns ``dict[registry_key, effective_alpha]``.
    """
    from saklas.cli_selectors import resolve_pole, AmbiguousSelectorError

    if isinstance(vector_specs, dict):
        items = [
            (coord.split("/", 1)[-1] if "/" in coord else coord,
             coord.split("/", 1)[0] if "/" in coord else None,
             coord, alpha)
            for coord, alpha in vector_specs.items()
        ]
    else:
        items = [(raw, None, raw, alpha) for raw, alpha in vector_specs]

    default_alphas: dict[str, float] = {}
    for raw_name, ns, display, alpha in items:
        try:
            canonical, sign, _match = resolve_pole(raw_name, namespace=ns)
        except AmbiguousSelectorError as e:
            if verbose:
                print(f"  Failed to resolve '{raw_name}': {e}", file=sys.stderr)
                sys.exit(1)
            print(f"  Failed to register '{display}': {e}")
            continue
        effective_alpha = alpha * sign
        try:
            if verbose:
                print(
                    f"Extracting steering vector: {canonical}"
                    + (f" (negated from '{raw_name}')" if sign < 0 else "")
                )
                _, profile = session.extract(
                    canonical, on_progress=lambda m: print(f"  {m}")
                )
            else:
                _, profile = session.extract(canonical)
        except Exception as e:
            if verbose:
                raise
            print(f"  Failed to register '{display}': {e}")
            continue
        registry_key = f"{ns}/{canonical}" if ns else canonical
        session.steer(registry_key, profile)
        default_alphas[registry_key] = effective_alpha
        print(f"  Registered '{registry_key}' (alpha={effective_alpha})"
              if not verbose else
              f"  Registered '{registry_key}' (default alpha={effective_alpha})")
    return default_alphas


def _run_tui(args: argparse.Namespace) -> None:
    _load_effective_config(args)
    if not args.model:
        print(
            "saklas tui: model required. Pass a HuggingFace repo id (e.g.\n"
            "  saklas tui google/gemma-2-2b-it\n"
            "or supply it via -c setup.yaml with a `model:` field.",
            file=sys.stderr,
        )
        sys.exit(2)

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    _setup_steering_vectors(session, getattr(args, "config_vectors", {}) or {})

    from saklas.tui.app import SaklasApp
    app = SaklasApp(session=session)
    app.run()


def _parse_steer_flag(raw: str) -> tuple[str, float]:
    if ":" in raw:
        name, alpha_s = raw.rsplit(":", 1)
        return name, float(alpha_s)
    return raw, 0.0


def _run_serve(args: argparse.Namespace) -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn
    except ImportError:
        print(
            "Server dependencies not installed. Run:\n"
            "  pip install saklas[serve]",
            file=sys.stderr,
        )
        sys.exit(1)

    _load_effective_config(args)

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    # Config-file vectors first, then explicit --steer flags on top.
    config_specs = getattr(args, "config_vectors", {}) or {}
    if config_specs:
        _setup_steering_vectors(session, config_specs, verbose=True)
    steer_specs = [_parse_steer_flag(spec) for spec in args.steer]
    default_alphas = _setup_steering_vectors(session, steer_specs, verbose=True)

    from saklas.server import create_app
    app = create_app(session, default_alphas=default_alphas,
                     cors_origins=args.cors or None,
                     api_key=getattr(args, "api_key", None))

    _warmup_session(session)

    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"OpenAI-compatible:  http://{args.host}:{args.port}/v1")
    print(f"Ollama-compatible:  http://{args.host}:{args.port}/api")
    print(f"API docs:           http://{args.host}:{args.port}/docs")
    if args.port != 11434:
        print("Tip: for drop-in Ollama compatibility, run with `--port 11434`.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _warmup_session(session) -> None:
    """Run a tiny stateless generation so the first real request is fast."""
    import time as _time
    from saklas.sampling import SamplingConfig
    print("Warming up generation kernels...", flush=True)
    try:
        start = _time.monotonic()
        session.generate(
            "Hi",
            sampling=SamplingConfig(max_tokens=1),
            stateless=True,
        )
        print(f"  warmed in {_time.monotonic() - start:.1f}s")
    except Exception as e:
        print(f"  warm-up skipped: {e}")


# --- pack runners --------------------------------------------------------

def _run_pack(args: argparse.Namespace) -> None:
    pack_cmd = getattr(args, "pack_cmd", None)
    if pack_cmd is None:
        print("usage: saklas pack <verb> [...]")
        print()
        width = max(len(v) for v, _ in _PACK_VERBS)
        for v, desc in _PACK_VERBS:
            print(f"  {v:<{width}}  {desc}")
        print()
        print("Run `saklas pack <verb> -h` for verb-specific options.")
        sys.exit(0)
    runner = _PACK_RUNNERS[pack_cmd]
    runner(args)


def _run_install(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    cache_ops.install(
        args.target,
        as_=args.as_target,
        force=args.force,
        statements_only=args.statements_only,
    )
    suffix = " (statements only)" if args.statements_only else ""
    print(f"Installed {args.target}{suffix}")


def _run_refresh(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    from saklas.cli_selectors import parse as sel_parse

    if args.selector == "neutrals":
        if args.model is not None:
            print("warning: --model has no effect with `refresh neutrals`", file=sys.stderr)
        dst = cache_ops.refresh_neutrals()
        print(f"Refreshed {dst}")
        return

    selector = sel_parse(args.selector)
    n = cache_ops.refresh(selector, model_scope=args.model)
    print(f"Refreshed {n} concept(s)")


def _run_clear(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    from saklas.cli_selectors import parse as sel_parse

    selector = sel_parse(args.selector)
    if selector.kind in {"all", "namespace"} and not args.yes:
        print(
            f"refusing to clear a broad selector ({selector.kind}); pass --yes to confirm",
            file=sys.stderr,
        )
        sys.exit(2)
    n = cache_ops.delete_tensors(selector, args.model)
    print(f"Deleted {n} files")


def _run_rm(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    from saklas.cli_selectors import parse as sel_parse

    selector = sel_parse(args.selector)
    try:
        n = cache_ops.uninstall(selector, yes=args.yes)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    print(f"Uninstalled {n} concept(s)")


def _run_ls(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    from saklas.cli_selectors import parse as sel_parse

    selector = sel_parse(args.selector) if args.selector else None
    cache_ops.list_local_packs(
        selector,
        json_output=args.json_output,
        verbose=args.verbose,
    )


def _run_search(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    cache_ops.search_remote_packs(
        args.query,
        json_output=args.json_output,
        verbose=args.verbose,
    )


def _run_export(args: argparse.Namespace) -> None:
    if args.format != "gguf":
        print(f"Unknown export format: {args.format}", file=sys.stderr)
        sys.exit(2)
    from saklas import cache_ops
    from saklas.cli_selectors import parse as sel_parse
    selector = sel_parse(args.selector)
    written = cache_ops.export_gguf(
        selector,
        model_scope=args.model,
        output=args.output,
        model_hint=args.model_hint,
    )
    for p in written:
        print(f"Wrote {p}")


def _run_merge(args: argparse.Namespace) -> None:
    from saklas import merge as merge_mod
    components = merge_mod.parse_components(args.components)
    dst = merge_mod.merge_into_pack(
        args.name, components, model=args.model,
        force=args.force, strict=args.strict,
    )
    print(f"Merged pack written to {dst}")


def _run_push(args: argparse.Namespace) -> None:
    from saklas import cache_ops
    from saklas.cli_selectors import parse as sel_parse

    selector = sel_parse(args.selector)
    try:
        coord, url, sha = cache_ops.push(
            selector,
            as_=args.as_target,
            private=args.private,
            model_scope=args.model,
            statements_only=args.statements_only,
            no_statements=args.no_statements,
            tag_version=args.tag_version,
            dry_run=args.dry_run,
            force=args.force,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    if sha:
        print(f"Pushed {coord} -> {url} @ {sha[:12]}")
    elif args.dry_run:
        print(f"Dry-run: would push {coord} -> {url}")
    else:
        print(f"Pushed {coord} -> {url}")


def _require_model(args: argparse.Namespace) -> None:
    if not args.model:
        print(f"{args.pack_cmd}: -m/--model is required", file=sys.stderr)
        sys.exit(2)


def _run_clone(args: argparse.Namespace) -> None:
    _require_model(args)
    from saklas.cloning import (
        CorpusTooShortError, CorpusTooLongError, InsufficientPairsError,
    )
    from saklas.cli_selectors import _all_concepts

    for c in _all_concepts():
        if c.name == args.name and c.namespace != "local":
            print(
                f"warning: '{args.name}' exists in namespace '{c.namespace}'; "
                f"reference this as 'local/{args.name}' to disambiguate",
                file=sys.stderr,
            )
            break

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    try:
        canonical, _profile = session.clone_from_corpus(
            args.corpus_path,
            name=args.name,
            n_pairs=args.n_pairs,
            seed=args.seed,
            force=args.force,
        )
    except (CorpusTooShortError, CorpusTooLongError, InsufficientPairsError) as e:
        print(f"clone failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Cloned persona -> local/{canonical}")


def _run_extract(args: argparse.Namespace) -> None:
    _require_model(args)
    from saklas.session import canonical_concept_name

    if len(args.concept) == 1:
        raw = args.concept[0]
        baseline = None
    elif len(args.concept) == 2:
        raw = args.concept[0]
        baseline = args.concept[1]
    else:
        print(
            "extract: expected 1 or 2 positional arguments "
            f"(got {len(args.concept)})",
            file=sys.stderr,
        )
        sys.exit(2)

    _print_startup(args)
    session = _make_session(args)
    _print_model_info(session)

    canonical = canonical_concept_name(raw, baseline)

    import pathlib
    from saklas.paths import safe_model_id
    from saklas.cli_selectors import _all_concepts
    candidate_folders = [c.folder for c in _all_concepts() if c.name == canonical]
    candidate_folders.append(session._local_concept_folder(canonical))
    sid = safe_model_id(session.model_id)
    candidate_paths = [
        pathlib.Path(folder) / f"{sid}.safetensors" for folder in candidate_folders
    ]
    existing = next((p for p in candidate_paths if p.exists()), None)

    if existing is not None and not args.force:
        print(f"already extracted at {existing}")
        sys.exit(0)

    if args.force:
        for p in candidate_paths:
            if p.exists():
                p.unlink()

    try:
        if baseline is not None:
            canonical, _profile = session.extract(raw, baseline=baseline)
        else:
            canonical, _profile = session.extract(raw)
    except Exception as e:
        print(f"extract failed: {e}", file=sys.stderr)
        sys.exit(1)

    final_path = next((p for p in candidate_paths if p.exists()), None)
    if final_path is None:
        final_path = (
            pathlib.Path(session._local_concept_folder(canonical)) / f"{sid}.safetensors"
        )
    print(f"extracted {canonical} -> {final_path}")


_PACK_RUNNERS = {
    "install": _run_install,
    "refresh": _run_refresh,
    "clear":   _run_clear,
    "rm":      _run_rm,
    "ls":      _run_ls,
    "search":  _run_search,
    "merge":   _run_merge,
    "push":    _run_push,
    "export":  _run_export,
    "clone":   _run_clone,
    "extract": _run_extract,
}


# --- config runners ------------------------------------------------------

def _run_config(args: argparse.Namespace) -> None:
    cmd = getattr(args, "config_cmd", None)
    if cmd == "show":
        _run_config_show(args)
    elif cmd == "validate":
        _run_config_validate(args)
    else:
        print("usage: saklas config {show,validate}")
        print()
        print("  show      Print the effective merged config")
        print("  validate  Validate a config file (exit 0 valid, 2 invalid)")
        sys.exit(0)


def _run_config_show(args: argparse.Namespace) -> None:
    from saklas import __version__
    from saklas.config_file import ConfigFile, apply_flag_overrides
    extras = [Path(p) for p in (args.config or [])]
    composed = ConfigFile.effective(extras, include_default=not args.no_default)
    if args.model is not None:
        composed = apply_flag_overrides(composed, model=args.model)
    composed = composed.resolve_poles()
    header = f"# effective merged config for saklas {__version__}"
    sys.stdout.write(composed.to_yaml(header=header))


def _run_config_validate(args: argparse.Namespace) -> None:
    from saklas.config_file import ConfigFile, ConfigFileError, ensure_vectors_installed
    p = Path(args.file)
    if not p.exists():
        print(f"config validate: {p}: file not found", file=sys.stderr)
        sys.exit(2)
    try:
        cfg = ConfigFile.load(p)
        cfg = cfg.resolve_poles()
        # Dry-run: don't install, just check resolvability.
        from saklas.cli_selectors import _all_concepts
        installed = {(c.namespace, c.name) for c in _all_concepts()}
        installed_names = {c.name for c in _all_concepts()}
        missing: list[str] = []
        for coord in cfg.vectors:
            if "/" in coord:
                ns, name = coord.split("/", 1)
                if ns == "default" or (ns, name) in installed:
                    continue
                if ns == "local":
                    missing.append(coord)
                    continue
                # HF namespace — we assume install would succeed; don't probe.
                continue
            else:
                if coord in installed_names:
                    continue
                missing.append(coord)
        if missing:
            raise ConfigFileError(
                f"unresolvable vectors (not installed and no namespace to install from): {missing}"
            )
    except ConfigFileError as e:
        print(f"config validate: {p}: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"config validate: {p}: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"{p}: ok")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

_COMMAND_RUNNERS = {
    "tui":    _run_tui,
    "serve":  _run_serve,
    "pack":   _run_pack,
    "config": _run_config,
}


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cmd = getattr(args, "command", None)
    if cmd is None:
        # argparse with required=False returns None when no verb given but
        # argv was non-empty (shouldn't happen — parse_args catches []).
        _build_root_parser().print_help()
        sys.exit(0)
    _COMMAND_RUNNERS[cmd](args)
