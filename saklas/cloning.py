"""Persona cloning via corpus-driven contrastive extraction.

Reuses the same contrastive-PCA pipeline as saklas.session.extract().
Instead of curated or LLM-synthesized pairs from a concept *name*,
pairs are `(persona_line, model_neutralized_rewrite)` — the model
flattens the persona's voice into plain prose, and the delta between
the two is the persona direction.

Only the entry point is public: `clone_from_corpus`. Tests hit the
module-private helpers directly to keep them CPU-only; the full
extraction path is covered by a GPU-gated end-to-end test.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import random
import re
from typing import TYPE_CHECKING

import torch

from saklas.generation import build_chat_input
from saklas.packs import NAME_REGEX, PackFormatError, PackMetadata, hash_file
from saklas.paths import concept_dir, safe_model_id
from saklas.vectors import extract_contrastive, load_profile as _load_profile, save_profile as _save_profile

if TYPE_CHECKING:
    from saklas.session import SaklasSession

_log = logging.getLogger(__name__)

_BATCH_SIZE = 5
_MIN_WORDS = 6
_MIN_CORPUS_LINES = 10
_TOKENS_PER_REWRITE = 50

_NUMBERED_RE = re.compile(r"^\s*(\d+)[.)]\s+(.+?)\s*$")


class CorpusTooShortError(ValueError):
    """Corpus has fewer than the minimum number of usable lines."""


class CorpusTooLongError(ValueError):
    """A batch prompt plus its generation budget does not fit in context."""


class InsufficientPairsError(RuntimeError):
    """Too few pairs survived generation/parsing to extract a stable vector."""


# ---------------------------------------------------------------------------
# Pure helpers (tested directly)
# ---------------------------------------------------------------------------

def _filter_corpus(path: str | pathlib.Path) -> list[str]:
    """Load corpus from disk, filter short lines, dedupe.

    UTF-8 with errors='replace' (never raises on bad bytes). Leading
    BOM stripped. Whitespace stripped per line. Lines with fewer than
    `_MIN_WORDS` whitespace-separated tokens are dropped. Duplicates
    are collapsed, preserving first-seen order.
    """
    path = pathlib.Path(path)
    raw = path.read_bytes().decode("utf-8", errors="replace")
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
    seen: set[str] = set()
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if len(s.split()) < _MIN_WORDS:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _sample_lines(lines: list[str], n: int, rng: random.Random) -> list[str]:
    """Seeded random sample without replacement. Clamps n to len(lines)."""
    if not lines:
        raise ValueError("cannot sample from empty corpus")
    n = min(n, len(lines))
    return rng.sample(lines, n)


def _chunk(items: list[str], batch_size: int) -> list[list[str]]:
    """Split a list into batches of at most `batch_size`. Last may be short."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _build_neutralize_prompt(batch: list[str]) -> str:
    """Build a numbered-list rewrite prompt for neutralizing a batch."""
    k = len(batch)
    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(batch))
    return (
        f"Rewrite each of the following lines in a plain, neutral voice. "
        f"Preserve meaning exactly. Do not add commentary. Output exactly "
        f"{k} numbered lines, one per line, in the form \"<i>. <rewrite>\".\n\n"
        f"{numbered}"
    )


def _parse_numbered(raw: str, expected: int) -> list[str] | None:
    """Parse a numbered-list response. Returns None on any format failure.

    Tolerates leading preamble and trailing notes. Requires exactly
    `expected` numbered lines, numbered 1..expected in strict order,
    each with a non-empty rewrite.
    """
    rewrites: list[str] = []
    for line in raw.splitlines():
        m = _NUMBERED_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        text = m.group(2).strip()
        rewrites.append((idx, text))
    if len(rewrites) != expected:
        return None
    for pos, (idx, text) in enumerate(rewrites, start=1):
        if idx != pos:
            return None
        if not text:
            return None
    return [text for _, text in rewrites]


def _fit_check(
    tokenizer,
    batch: list[str],
    generation_budget: int,
    ctx_len: int,
) -> bool:
    """Rough upper-bound fit check for a single batch."""
    prompt = _build_neutralize_prompt(batch)
    try:
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    except Exception:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Add slack for chat template wrapping.
    overhead = 64
    return (len(ids) + overhead + generation_budget) <= ctx_len


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _hash_file_bytes(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _neutralize_batch(
    session: "SaklasSession",
    batch: list[str],
    seed: int | None,
) -> list[str] | None:
    """Run the neutralize prompt through the raw HF model, parse result."""
    prompt = _build_neutralize_prompt(batch)
    system_msg = (
        "You rewrite text in plain neutral prose. You follow format "
        "instructions exactly."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    input_ids = build_chat_input(
        session._tokenizer, messages, system_prompt=None,
    ).to(session._device)

    if seed is not None:
        torch.manual_seed(seed)

    pad_id = session._tokenizer.pad_token_id or session._tokenizer.eos_token_id
    max_new = len(batch) * _TOKENS_PER_REWRITE + 32
    with torch.inference_mode():
        out = session._model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=pad_id,
        )
    new_ids = out[0][input_ids.shape[-1]:]
    text = session._tokenizer.decode(new_ids, skip_special_tokens=True)
    return _parse_numbered(text, len(batch))


def clone_from_corpus(
    session: "SaklasSession",
    path: str | pathlib.Path,
    name: str,
    *,
    n_pairs: int = 45,
    seed: int | None = None,
    force: bool = False,
) -> tuple[str, dict[int, torch.Tensor]]:
    """Extract a monopolar steering vector from a persona corpus file.

    Pipeline: load + filter corpus, sample n_pairs lines, batch them,
    ask the model to rewrite each batch in neutral voice, pair each
    persona line with its rewrite, feed into the same contrastive PCA
    path as session.extract(). Returns `(canonical_name, profile)`.

    Cache: keyed on corpus sha256 + n_pairs + batch_size + seed +
    model. Pass `force=True` to bypass.
    """
    # Local imports to avoid a circular import with saklas.session.
    from saklas.session import canonical_concept_name, _N_PAIRS

    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus file not found: {path}")

    canonical = canonical_concept_name(name)
    if not NAME_REGEX.match(canonical):
        raise ValueError(
            f"invalid concept name after slugging: {canonical!r}"
        )

    folder = concept_dir("local", canonical)
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "pack.json").exists():
        PackMetadata(
            name=canonical,
            description=f"Cloned from {path.name}",
            version="1.0.0",
            license="AGPL-3.0-or-later",
            tags=["cloned"],
            recommended_alpha=0.5,
            source="local",
            files={},
        ).write(folder)

    corpus_sha = _hash_file_bytes(path)
    cache_path = folder / f"{safe_model_id(session.model_id)}.safetensors"

    # Cache hit check.
    if not force and cache_path.exists():
        try:
            meta = PackMetadata.load(folder)
            extra = getattr(meta, "extra", None) or {}
            # PackMetadata may not round-trip unknown fields; fall back to raw.
            raw_pack = json.loads((folder / "pack.json").read_text())
            cached_sha = raw_pack.get("corpus_sha256")
            cached_n = raw_pack.get("n_pairs")
            cached_bs = raw_pack.get("batch_size")
            cached_seed = raw_pack.get("seed")
            if (
                cached_sha == corpus_sha
                and cached_n is not None
                and cached_bs == _BATCH_SIZE
                and cached_seed == seed
            ):
                profile, _m = _load_profile(str(cache_path))
                profile = session._promote_profile(profile)
                print(f"cloned {canonical} (cache hit) -> {folder}")
                return canonical, profile
        except (PackFormatError, FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
            pass

    lines = _filter_corpus(path)
    if len(lines) < _MIN_CORPUS_LINES:
        raise CorpusTooShortError(
            f"corpus has {len(lines)} usable lines after filtering; "
            f"need at least {_MIN_CORPUS_LINES}"
        )

    if n_pairs > len(lines):
        _log.warning(
            "clone: requested %d pairs but corpus only has %d lines; clamping",
            n_pairs, len(lines),
        )
        n_pairs = len(lines)

    # Record effective seed so reruns are reproducible.
    effective_seed = seed if seed is not None else random.Random().randint(0, 2**31 - 1)
    _log.info("clone: effective seed = %d", effective_seed)
    rng = random.Random(effective_seed)

    sample = _sample_lines(lines, n_pairs, rng)
    batches = _chunk(sample, _BATCH_SIZE)

    # Fit check (use worst-case = longest batch). Context length via model config.
    ctx_len = getattr(session._model.config, "max_position_embeddings", None) or 4096
    longest = max(batches, key=len)
    budget = len(longest) * _TOKENS_PER_REWRITE + 32
    if not _fit_check(session._tokenizer, longest, budget, ctx_len):
        raise CorpusTooLongError(
            f"batch prompt + generation budget exceeds context length "
            f"({ctx_len}); try a smaller --n-pairs or shorter corpus lines"
        )

    # Save and clear steering state — extracted pairs must not be
    # contaminated by whatever's currently active.
    saved_vectors = dict(session._steering.vectors)
    had_hooks = bool(session._steering.hooks)
    session._steering.clear_all()

    pairs: list[tuple[str, str]] = []
    try:
        for i, batch in enumerate(batches):
            batch_seed = effective_seed + i
            rewrites = _neutralize_batch(session, batch, batch_seed)
            if rewrites is None:
                # Retry once with a perturbed seed.
                _log.warning("clone: batch %d parse failed; retrying", i)
                rewrites = _neutralize_batch(session, batch, batch_seed + 10_000)
            if rewrites is None:
                _log.warning("clone: batch %d dropped after retry", i)
                continue
            for persona_line, rewrite in zip(batch, rewrites):
                pairs.append((persona_line, rewrite))
    finally:
        # Restore prior vector registrations.
        for vname, vdata in saved_vectors.items():
            session._steering.add_vector(vname, vdata["profile"], vdata["alpha"])
        if had_hooks:
            session._steering.apply_to_model(
                session._layers, session._device, session._dtype,
            )

    if len(pairs) < _N_PAIRS // 2:
        raise InsufficientPairsError(
            f"only {len(pairs)} pairs survived generation; "
            f"need at least {_N_PAIRS // 2}"
        )

    # Persist statements.json in the same shape as bundled packs.
    stmts_path = folder / "statements.json"
    stmt_objs = [{"positive": p, "negative": n} for p, n in pairs]
    stmts_path.write_text(json.dumps(stmt_objs, indent=2))

    # Run the same contrastive PCA path as session.extract().
    profile = extract_contrastive(
        session._model, session._tokenizer, stmt_objs, layers=session._layers,
    )
    _save_profile(profile, str(cache_path), {
        "method": "contrastive_pca_cloned",
        "statements_sha256": hash_file(stmts_path),
    })

    # Update pack.json: refresh file hashes, then stamp clone metadata.
    try:
        meta = PackMetadata.load(folder)
        new_files: dict[str, str] = {}
        for entry in sorted(folder.iterdir()):
            if entry.is_file() and entry.name != "pack.json":
                new_files[entry.name] = hash_file(entry)
        meta.files = new_files
        meta.tags = sorted(set((meta.tags or []) + ["cloned"]))
        meta.description = f"Cloned from {path.name}"
        meta.source = "local"
        meta.write(folder)
    except PackFormatError:
        pass

    # Stamp cloning-specific metadata directly (PackMetadata may not
    # carry arbitrary fields on write).
    raw_pack = json.loads((folder / "pack.json").read_text())
    raw_pack["corpus_sha256"] = corpus_sha
    raw_pack["n_pairs"] = n_pairs
    raw_pack["batch_size"] = _BATCH_SIZE
    raw_pack["seed"] = seed
    (folder / "pack.json").write_text(json.dumps(raw_pack, indent=2))

    print(
        f"cloned {canonical} -> {folder}\n"
        f"corpus not included in pack; statements.json contains "
        f"model-generated pairs only"
    )
    return canonical, profile
