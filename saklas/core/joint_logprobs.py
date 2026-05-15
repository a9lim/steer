"""Cross-branch joint logprobs (Phase 5 of docs/plans/logit-pass.md).

Given two assistant LoomNodes A and B that share a parent, run one
forward pass per branch and report, for each aligned assistant-token
position pair:

* ``lp_a_in_a`` / ``lp_b_in_b`` — chosen-token logprob under each
  branch's own distribution.  Mirrors what the engine already captures
  at generation time, recomputed here so the cross-evaluation and the
  self-evaluation use bit-identical math.
* ``lp_a_in_b`` / ``lp_b_in_a`` — chosen-token logprob under the *other*
  branch's distribution at the byte-aligned position.  Answers "what
  would B have given the token A picked here?" and vice versa.
* ``rank_changed`` — true iff the argmax token differs between the two
  distributions at this aligned position.  This is the canonical
  "steering shifted the head of the distribution, not just the
  argmax" signal.
* ``approx_kl`` — top-K-truncated KL(P_A || P_B), summed over the union
  of each side's top-K tokens.  The tail is unobserved, so this is
  documented as approximate signal not measurement (per Decision 5).

The route is fired lazily on NodeCompareDrawer open per Decision 9;
results cache on the session keyed by sorted ``(a_id, b_id)`` for the
session lifetime.  Tree mutations that rename / delete the involved
nodes invalidate the entries (see ``SaklasSession`` cache wiring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from saklas.core.generation import build_chat_input
from saklas.core.loom_diff import per_token_diff

if TYPE_CHECKING:  # avoid a hard import cycle at module load
    from saklas.core.session import SaklasSession


# Truncation budget for the approximate KL.  ~32 covers the practical
# mass at typical sampler temperatures; we don't try to estimate the
# tail because the engine isn't shipping it.
_KL_TOP_K = 32


@dataclass(frozen=True)
class JointLogprobRow:
    """One aligned-position record in a :class:`JointLogprobs` result.

    Indices are positions in each branch's *assistant-only* token list
    (i.e. relative to the divergence point), so they line up with the
    drawer's per-token row rendering.  Text fields carry the decoded
    token strings, ready for display without re-tokenization.
    """

    a_index: int
    b_index: int
    a_text: str
    b_text: str
    aligned: bool
    lp_a_in_a: float | None
    lp_b_in_b: float | None
    lp_a_in_b: float | None
    lp_b_in_a: float | None
    rank_changed: bool
    approx_kl: float | None


@dataclass(frozen=True)
class JointLogprobs:
    a_id: str
    b_id: str
    parent_id: str | None
    rows: tuple[JointLogprobRow, ...]
    n_rank1_changed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_id": self.a_id,
            "b_id": self.b_id,
            "parent_id": self.parent_id,
            "n_rank1_changed": self.n_rank1_changed,
            "rows": [
                {
                    "a_index": r.a_index,
                    "b_index": r.b_index,
                    "a_text": r.a_text,
                    "b_text": r.b_text,
                    "aligned": r.aligned,
                    "lp_a_in_a": r.lp_a_in_a,
                    "lp_b_in_b": r.lp_b_in_b,
                    "lp_a_in_b": r.lp_a_in_b,
                    "lp_b_in_a": r.lp_b_in_a,
                    "rank_changed": r.rank_changed,
                    "approx_kl": r.approx_kl,
                }
                for r in self.rows
            ],
        }


# ---------------------------------------------------------------------------
# Pure-math core (tested in isolation; see tests/test_joint_logprobs.py)
# ---------------------------------------------------------------------------


def _approx_kl_topk(
    logp_a: torch.Tensor,  # [V] fp32 log-softmax
    logp_b: torch.Tensor,  # [V] fp32 log-softmax
    top_k: int,
) -> float:
    """Truncated KL(P_A || P_B) over the union of each side's top-K.

    Within the union, contributions to ``Σ P_A(t) (log P_A(t) − log P_B(t))``
    are summed exactly; the tail (tokens neither side ranked in its top-K)
    is dropped — that's the "approx" in ``approx_kl``.  For typical
    post-sampler distributions K=32 captures the bulk; the residual is
    well under 0.1 nats in practice.
    """
    vocab = logp_a.shape[-1]
    k = min(top_k, vocab)
    top_a = torch.topk(logp_a, k).indices
    top_b = torch.topk(logp_b, k).indices
    union = torch.unique(torch.cat([top_a, top_b]))
    pa = logp_a.index_select(0, union).exp()
    diff = logp_a.index_select(0, union) - logp_b.index_select(0, union)
    return float((pa * diff).sum().item())


def _compute_rows(
    logp_a: torch.Tensor,        # [T_a, V] log-softmax of A's logits
    logp_b: torch.Tensor,        # [T_b, V]
    token_ids_a: list[int],      # full sequence (prefix + assistant) for A
    token_ids_b: list[int],
    token_strs_a: list[str],     # decoded text per id (display) for A's full seq
    token_strs_b: list[str],
    prefix_len: int,             # shared prefix length (in tokens)
    *,
    kl_top_k: int = _KL_TOP_K,
) -> list[JointLogprobRow]:
    """Pure-tensor inner loop — no session / tokenizer / IO.

    Aligns A's assistant tail and B's assistant tail via the shared
    :func:`per_token_diff` byte-offset walker, then looks up logprobs
    from the precomputed log-softmax tables.  Position ``prefix_len + i``
    in the full sequence is *predicted* by the logits at position
    ``prefix_len + i - 1`` — that's the index we read from on each row.
    """
    assistant_ids_a = token_ids_a[prefix_len:]
    assistant_ids_b = token_ids_b[prefix_len:]
    assistant_strs_a = token_strs_a[prefix_len:]
    assistant_strs_b = token_strs_b[prefix_len:]

    # ``per_token_diff`` walks byte-offset alignment over the per-token
    # display strings; we feed it the assistant tail so ``a_index`` /
    # ``b_index`` come back in the same space we'll surface to the UI.
    spans = per_token_diff(assistant_strs_a, assistant_strs_b)

    rows: list[JointLogprobRow] = []
    for sp in spans:
        a_idx = sp.a_index
        b_idx = sp.b_index
        # Logits *at* full-sequence position k predict the token at k+1,
        # so to score the token at full-position prefix_len+i we read
        # logp[prefix_len + i - 1].  ``max(0, …)`` guards the (degenerate)
        # case where prefix_len is 0 and i is 0 — fall back to position
        # 0's logits, which are conditioned on nothing and will produce
        # the unigram-like prior.
        pa_pos = max(0, prefix_len + a_idx - 1)
        pb_pos = max(0, prefix_len + b_idx - 1)

        # Self-evaluation: chosen logprob under own distribution.
        lp_a_in_a: float | None = None
        lp_b_in_b: float | None = None
        if 0 <= a_idx < len(assistant_ids_a):
            lp_a_in_a = float(logp_a[pa_pos, assistant_ids_a[a_idx]].item())
        if 0 <= b_idx < len(assistant_ids_b):
            lp_b_in_b = float(logp_b[pb_pos, assistant_ids_b[b_idx]].item())

        # Cross-evaluation: only meaningful when the positions actually
        # align (byte-equal context up to here).  On divergent rows the
        # cross-prob is ambiguous (which prior position do we score
        # against?) so we leave it null.
        lp_a_in_b: float | None = None
        lp_b_in_a: float | None = None
        rank_changed = False
        approx_kl: float | None = None
        if sp.aligned and 0 <= a_idx < len(assistant_ids_a) and 0 <= b_idx < len(assistant_ids_b):
            lp_a_in_b = float(logp_b[pb_pos, assistant_ids_a[a_idx]].item())
            lp_b_in_a = float(logp_a[pa_pos, assistant_ids_b[b_idx]].item())
            # Rank-1 change: does the argmax differ at this aligned
            # position?  Cheap signal — one ``argmax`` per side.
            argmax_a = int(logp_a[pa_pos].argmax().item())
            argmax_b = int(logp_b[pb_pos].argmax().item())
            rank_changed = argmax_a != argmax_b
            approx_kl = _approx_kl_topk(
                logp_a[pa_pos], logp_b[pb_pos], kl_top_k,
            )

        rows.append(JointLogprobRow(
            a_index=a_idx,
            b_index=b_idx,
            a_text=sp.a_text,
            b_text=sp.b_text,
            aligned=sp.aligned,
            lp_a_in_a=lp_a_in_a,
            lp_b_in_b=lp_b_in_b,
            lp_a_in_b=lp_a_in_b,
            lp_b_in_a=lp_b_in_a,
            rank_changed=rank_changed,
            approx_kl=approx_kl,
        ))
    return rows


# ---------------------------------------------------------------------------
# IO wrapper — talks to the session, tokenizer, model
# ---------------------------------------------------------------------------


def _shared_prefix_len(ids_a: list[int], ids_b: list[int]) -> int:
    """Longest common prefix length between two token-id lists.

    Both branches share their parent's chat-template prefix verbatim,
    so this lands at the divergence point — exactly where the assistant
    content starts to differ.  Used instead of re-tokenizing the parent
    prefix separately to dodge template-boundary surprises.
    """
    n = min(len(ids_a), len(ids_b))
    i = 0
    while i < n and ids_a[i] == ids_b[i]:
        i += 1
    return i


def compute_joint_logprobs(
    session: "SaklasSession",
    a_id: str,
    b_id: str,
) -> JointLogprobs:
    """Run cross-evaluation between two assistant sibling nodes.

    Builds each branch's full input through the chat template, runs one
    forward pass per branch in ``inference_mode``, and assembles per-
    aligned-position records.  Caller is responsible for holding
    ``session.lock`` — model forwards must serialize against any
    concurrent generation on the same session.

    Raises ``KeyError`` when either node id is unknown to the tree.
    Returns an empty-rows :class:`JointLogprobs` when the branches share
    no divergent assistant tokens (e.g. one node is empty), which is the
    least-surprising shape for the drawer to render.
    """
    tree = session.tree
    a_node = tree.nodes[a_id]
    b_node = tree.nodes[b_id]
    parent_id = a_node.parent_id if a_node.parent_id == b_node.parent_id else None

    tokenizer = session.tokenizer
    model = session._model
    device = next(model.parameters()).device

    # Build full token sequence for each branch.  ``add_generation_prompt
    # =False`` because the assistant content *is* the last message —
    # we don't want the template to start a new assistant turn after it.
    messages_a = tree.messages_for(a_id)
    messages_b = tree.messages_for(b_id)
    system_prompt = session.config.system_prompt or None
    input_a = build_chat_input(
        tokenizer, messages_a, system_prompt=system_prompt,
        thinking=False, add_generation_prompt=False,
    )
    input_b = build_chat_input(
        tokenizer, messages_b, system_prompt=system_prompt,
        thinking=False, add_generation_prompt=False,
    )
    # Both shapes are ``[1, T]``; squeeze the batch dim for our ops and
    # restore it inside the forward.
    ids_a = input_a[0].tolist()
    ids_b = input_b[0].tolist()
    prefix_len = _shared_prefix_len(ids_a, ids_b)

    # If neither side has any assistant tokens past the prefix, return
    # an empty result — nothing to align.
    if prefix_len >= len(ids_a) and prefix_len >= len(ids_b):
        return JointLogprobs(
            a_id=a_id, b_id=b_id, parent_id=parent_id,
            rows=(), n_rank1_changed=0,
        )

    # Per-token display text — ``tokenizer.decode([id])`` gives the
    # human-readable rendering (space-prefixed where appropriate),
    # which byte-aligns cleanly under ``per_token_diff`` because both
    # branches tokenize the shared prefix identically.
    def _decode_each(ids: list[int]) -> list[str]:
        # Batch-decode-per-id rather than ``decode(ids)`` to keep the
        # per-position list aligned 1:1 with the id list.  Cost is
        # ~O(T) tokenizer calls; small enough for a single drawer
        # open.  Coerce to ``str`` because some tokenizer's ``decode``
        # signature is typed ``list[str] | str``.
        return [str(tokenizer.decode([tid])) for tid in ids]

    strs_a = _decode_each(ids_a)
    strs_b = _decode_each(ids_b)

    # Run the two forward passes.  ``use_cache=False`` — we don't reuse
    # the kv across the two branches, and a single one-shot forward is
    # what fastest on every backend.  ``inference_mode`` matches the
    # generation path's discipline.
    input_a_dev = input_a.to(device)
    input_b_dev = input_b.to(device)
    with torch.inference_mode():
        out_a = model(input_ids=input_a_dev, use_cache=False)
        out_b = model(input_ids=input_b_dev, use_cache=False)
    # Cast to fp32 for the log-softmax to dodge fp16 overflow at large
    # vocabs.  Move to CPU so the downstream ``.item()`` reads don't
    # bounce through device sync per cell.
    logp_a = torch.log_softmax(out_a.logits[0].float(), dim=-1).cpu()
    logp_b = torch.log_softmax(out_b.logits[0].float(), dim=-1).cpu()

    rows = _compute_rows(
        logp_a, logp_b, ids_a, ids_b, strs_a, strs_b, prefix_len,
    )
    n_changed = sum(1 for r in rows if r.aligned and r.rank_changed)
    return JointLogprobs(
        a_id=a_id, b_id=b_id, parent_id=parent_id,
        rows=tuple(rows), n_rank1_changed=n_changed,
    )


# ---------------------------------------------------------------------------
# Cache helpers — symmetric key so (a, b) and (b, a) dedupe
# ---------------------------------------------------------------------------


def _cache_key(a_id: str, b_id: str) -> tuple[str, str]:
    """Symmetric (a, b) ↔ (b, a) cache key.

    Sorted so the drawer hits the same entry regardless of which node
    the user right-clicked first.  Result layout still respects the
    caller's ``(a, b)`` orientation — see :func:`reorient_for_request`.
    """
    return (a_id, b_id) if a_id <= b_id else (b_id, a_id)


def reorient_for_request(
    result: JointLogprobs, requested_a_id: str, requested_b_id: str,
) -> JointLogprobs:
    """Flip the result's a/b labelling to match the caller's request.

    Cache stores under the sorted key, so a request for ``(B, A)`` after
    ``(A, B)`` was already computed needs the columns swapped before
    the drawer renders them.  Pure metadata work — no recomputation.
    """
    if (result.a_id, result.b_id) == (requested_a_id, requested_b_id):
        return result
    swapped_rows = tuple(
        JointLogprobRow(
            a_index=r.b_index,
            b_index=r.a_index,
            a_text=r.b_text,
            b_text=r.a_text,
            aligned=r.aligned,
            lp_a_in_a=r.lp_b_in_b,
            lp_b_in_b=r.lp_a_in_a,
            lp_a_in_b=r.lp_b_in_a,
            lp_b_in_a=r.lp_a_in_b,
            rank_changed=r.rank_changed,
            approx_kl=r.approx_kl,
        )
        for r in result.rows
    )
    return JointLogprobs(
        a_id=requested_a_id,
        b_id=requested_b_id,
        parent_id=result.parent_id,
        rows=swapped_rows,
        n_rank1_changed=result.n_rank1_changed,
    )


__all__ = [
    "JointLogprobRow",
    "JointLogprobs",
    "compute_joint_logprobs",
    "_compute_rows",
    "_approx_kl_topk",
    "_shared_prefix_len",
    "_cache_key",
    "reorient_for_request",
    "_KL_TOP_K",
]
