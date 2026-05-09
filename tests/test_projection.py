"""Runtime projection: unit-level project_profile + session materialization.

Pure tensor math is tested in ``TestProjectProfile``; the session-level
integration rides on the same ``_Stub`` pattern used by
``test_steering_context.py`` — a ``SaklasSession`` that bypasses the
model-loading machinery and pre-registers profiles directly.
"""
from __future__ import annotations

import pytest
import torch

from saklas.core.events import EventBus
from saklas.core.session import (
    SaklasSession, VectorNotRegisteredError,
)
from saklas.core.steering_expr import parse_expr
from saklas.core.triggers import Trigger
from saklas.core.vectors import project_profile


# ------------------------------------------------------- project_profile ---

class TestProjectProfile:
    def test_orthogonal_to_parallel_is_zero(self):
        base = {0: torch.tensor([1.0, 0.0])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out = project_profile(base, onto, "|")
        assert torch.allclose(out[0], torch.zeros(2), atol=1e-6)

    def test_onto_of_parallel_is_base(self):
        base = {0: torch.tensor([2.0, 0.0])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out = project_profile(base, onto, "~")
        assert torch.allclose(out[0], torch.tensor([2.0, 0.0]), atol=1e-6)

    def test_orthogonal_drops_shared_axis(self):
        # base = [1, 1]; onto = [1, 0].  Projection onto = [1, 0]; orthogonal = [0, 1].
        base = {0: torch.tensor([1.0, 1.0])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out_ortho = project_profile(base, onto, "|")
        assert torch.allclose(out_ortho[0], torch.tensor([0.0, 1.0]), atol=1e-6)

    def test_onto_keeps_shared_axis(self):
        base = {0: torch.tensor([1.0, 1.0])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out_onto = project_profile(base, onto, "~")
        assert torch.allclose(out_onto[0], torch.tensor([1.0, 0.0]), atol=1e-6)

    def test_multi_layer(self):
        base = {
            0: torch.tensor([1.0, 1.0]),
            1: torch.tensor([2.0, 2.0]),
        }
        onto = {
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([0.0, 1.0]),
        }
        out = project_profile(base, onto, "|")
        assert torch.allclose(out[0], torch.tensor([0.0, 1.0]), atol=1e-6)
        assert torch.allclose(out[1], torch.tensor([2.0, 0.0]), atol=1e-6)

    def test_missing_onto_layer_passes_through_for_ortho(self):
        base = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.5, 0.5])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out = project_profile(base, onto, "|")
        assert 1 in out
        assert torch.allclose(out[1], torch.tensor([0.5, 0.5]), atol=1e-6)
        assert torch.allclose(out[0], torch.zeros(2), atol=1e-6)

    def test_missing_onto_layer_drops_for_onto(self):
        base = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.5, 0.5])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out = project_profile(base, onto, "~")
        assert 1 not in out
        assert 0 in out

    def test_near_zero_onto_passes_base_for_ortho(self):
        base = {0: torch.tensor([1.0, 0.0])}
        onto = {0: torch.tensor([1e-20, 0.0])}
        out = project_profile(base, onto, "|")
        assert torch.allclose(out[0], torch.tensor([1.0, 0.0]), atol=1e-6)

    def test_near_zero_onto_drops_for_onto_operator(self):
        base = {0: torch.tensor([1.0, 0.0])}
        onto = {0: torch.tensor([1e-20, 0.0])}
        with pytest.raises(ValueError):
            project_profile(base, onto, "~")

    def test_unknown_operator_raises(self):
        base = {0: torch.tensor([1.0])}
        onto = {0: torch.tensor([1.0])}
        with pytest.raises(ValueError):
            project_profile(base, onto, "@")

    def test_empty_intersection_raises_for_ortho(self):
        base = {0: torch.tensor([1.0])}
        onto = {7: torch.tensor([1.0])}
        # "|" passes through layer 0 (not in onto), so the result has
        # layer 0 — non-empty.
        out = project_profile(base, onto, "|")
        assert set(out.keys()) == {0}

    def test_empty_intersection_raises_for_onto_operator(self):
        base = {0: torch.tensor([1.0])}
        onto = {7: torch.tensor([1.0])}
        with pytest.raises(ValueError):
            project_profile(base, onto, "~")

    def test_result_dtype_matches_base(self):
        base = {0: torch.tensor([1.0, 1.0], dtype=torch.float16)}
        onto = {0: torch.tensor([1.0, 0.0], dtype=torch.float16)}
        out = project_profile(base, onto, "|")
        assert out[0].dtype == torch.float16


# ---------------------------------------------- session-level integration ---

class _Stub(SaklasSession):
    """SaklasSession without real model/tokenizer, mirrors test_steering_context."""
    def __init__(self, profiles: dict) -> None:  # type: ignore[override]
        self._profiles = dict(profiles)
        self._steering_stack = []
        self._steering_override_stack = []
        # v2.2 session-level defaults consulted by ``_resolve_*``
        # helpers when the override LIFO has no entries.
        from saklas.core.hooks import DEFAULT_THETA_MAX as _DTM
        self._injection_mode = "angular"
        self._theta_max = _DTM
        self._projection_metric = "mahalanobis"
        self._whitener = None
        self._layer_means = {}
        self.events = EventBus()
        self._rebuild_calls: list[dict[str, float]] = []
        self._rebuild_entries: list[dict[str, tuple[float, Trigger]]] = []

    @property
    def whitener(self) -> None:  # type: ignore[override]
        # No model in stub mode — return None so
        # ``_materialize_projections`` falls back to Euclidean
        # per-layer (the path real sessions hit when neutrals aren't
        # cached yet).
        return None

    def _rebuild_steering_hooks(self) -> None:  # type: ignore[override]
        flat = self._flatten_steering_stack()
        for name in flat:
            if name not in self._profiles:
                raise VectorNotRegisteredError(f"No vector registered for '{name}'")
        self._rebuild_entries.append(dict(flat))
        self._rebuild_calls.append(
            {name: alpha for name, (alpha, _trig) in flat.items()},
        )

    def _resolve_pole_aliases(self, entries):  # type: ignore[override]
        return {k: (float(v[0]), v[1]) for k, v in entries.items()}


def _profile_a():
    # "a" direction, layer 0 only.
    return {0: torch.tensor([1.0, 1.0])}


def _profile_b():
    # "b" direction along x-axis.
    return {0: torch.tensor([1.0, 0.0])}


class TestSessionProjection:
    def test_parses_and_registers_synthetic_key(self):
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        steering = parse_expr("0.5 a|b")
        with s.steering(steering):
            assert "a|b" in s._profiles
            # Registered projection is orthogonal: [0, 1] (the y-component).
            assert torch.allclose(
                s._profiles["a|b"][0], torch.tensor([0.0, 1.0]), atol=1e-6,
            )
            assert s._rebuild_calls[-1] == {"a|b": 0.5}

    def test_onto_operator_registers_projected(self):
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        steering = parse_expr("0.5 a~b")
        with s.steering(steering):
            assert "a~b" in s._profiles
            assert torch.allclose(
                s._profiles["a~b"][0], torch.tensor([1.0, 0.0]), atol=1e-6,
            )

    def test_mixed_plain_and_projection(self):
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        steering = parse_expr("0.3 a + 0.5 a|b")
        with s.steering(steering):
            flat = s._rebuild_calls[-1]
            assert flat == {"a": 0.3, "a|b": 0.5}

    def test_projection_trigger_propagates(self):
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        steering = parse_expr("0.5 a|b@after")
        with s.steering(steering):
            entries = s._rebuild_entries[-1]
            assert entries["a|b"] == (0.5, Trigger.AFTER_THINKING)

    def test_projection_missing_base_raises(self):
        s = _Stub({"b": _profile_b()})  # no "a" registered
        steering = parse_expr("0.5 a|b")
        with pytest.raises(VectorNotRegisteredError) as ei:
            with s.steering(steering):
                pass
        assert "a" in str(ei.value) or "projection" in str(ei.value)

    def test_projection_missing_onto_raises(self):
        s = _Stub({"a": _profile_a()})  # no "b"
        steering = parse_expr("0.5 a|b")
        with pytest.raises(VectorNotRegisteredError):
            with s.steering(steering):
                pass

    def test_base_direction_differs_from_orthogonal(self):
        # Sanity: steering "a" and steering "a|b" should register
        # measurably different tensors in _profiles.
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        with s.steering(parse_expr("1.0 a|b")):
            projected = s._profiles["a|b"][0].clone()
        # Baseline is a = [1, 1]; orthogonal-to-b strips x-component -> [0, 1].
        assert not torch.allclose(
            projected, _profile_a()[0], atol=1e-3,
        )
        assert torch.allclose(projected, torch.tensor([0.0, 1.0]), atol=1e-6)

    def test_nested_projection_scopes(self):
        # Keys "a" and "a|b" don't collide, so nesting leaves both active
        # in the flattened head; exiting the inner scope restores the outer.
        s = _Stub({"a": _profile_a(), "b": _profile_b()})
        with s.steering(parse_expr("0.3 a")):
            assert s._rebuild_calls[-1] == {"a": 0.3}
            with s.steering(parse_expr("0.5 a|b")):
                assert s._rebuild_calls[-1] == {"a": 0.3, "a|b": 0.5}
            assert s._rebuild_calls[-1] == {"a": 0.3}
        assert s._rebuild_calls[-1] == {}


# ---------------------------------------- v2.2 metric-default integration ---

class _MetricStub(_Stub):
    """Variant of ``_Stub`` that lets tests select a metric and stand-in
    whitener.  ``project_profile`` is patched per-test to record kwarg
    shape rather than do real math.
    """

    def __init__(
        self,
        profiles: dict,
        *,
        projection_metric: str = "mahalanobis",
        whitener_value: object = "WHITENER",
    ) -> None:
        super().__init__(profiles)
        self._projection_metric = projection_metric
        # Sentinel — patched ``project_profile`` only checks ``is None``.
        self._whitener_sentinel = whitener_value

    @property
    def whitener(self) -> object:  # type: ignore[override]
        return self._whitener_sentinel


class TestProjectionMetricDefault:
    """The v2.2 default flips runtime ``~`` / ``|`` to Mahalanobis.

    Verifies that ``_materialize_projections`` passes ``self.whitener``
    to ``project_profile`` under the default metric, and ``None``
    under ``"euclidean"`` (the ``--legacy`` path).
    """

    def _patch_project(self, monkeypatch, calls: list) -> None:
        # ``_materialize_projections`` imports ``project_profile``
        # lazily inside the method (``from saklas.core.vectors import
        # project_profile``), so we patch it on the source module.
        from saklas.core import vectors as vectors_mod

        def _spy(base, onto, operator, *, whitener=None):
            calls.append((operator, whitener is None))
            return {0: torch.tensor([0.0, 1.0])}

        monkeypatch.setattr(vectors_mod, "project_profile", _spy)

    def test_default_passes_whitener(self, monkeypatch):
        calls: list = []
        s = _MetricStub({"a": _profile_a(), "b": _profile_b()})
        self._patch_project(monkeypatch, calls)
        with s.steering(parse_expr("0.5 a|b")):
            pass
        assert calls == [("|", False)], (
            "default session should hand session.whitener to project_profile"
        )

    def test_legacy_metric_passes_none(self, monkeypatch):
        calls: list = []
        s = _MetricStub(
            {"a": _profile_a(), "b": _profile_b()},
            projection_metric="euclidean",
        )
        self._patch_project(monkeypatch, calls)
        with s.steering(parse_expr("0.5 a|b")):
            pass
        assert calls == [("|", True)], (
            "euclidean session should pass whitener=None"
        )

    def test_per_call_override_flips_metric(self, monkeypatch):
        from saklas.core.steering import Steering

        calls: list = []
        s = _MetricStub({"a": _profile_a(), "b": _profile_b()})
        self._patch_project(monkeypatch, calls)
        # First scope inherits the session default ("mahalanobis").
        with s.steering(parse_expr("0.5 a|b")):
            pass
        # Second scope overrides to euclidean via Steering.projection_metric.
        # parse_expr doesn't surface this field — programmatic-only.
        base = parse_expr("0.5 a|b")
        override = Steering(
            alphas=base.alphas,
            thinking=base.thinking,
            trigger=base.trigger,
            projection_metric="euclidean",
        )
        with s.steering(override):
            pass
        assert calls == [("|", False), ("|", True)]

    def test_per_call_override_inherits_to_inner_scope(self, monkeypatch):
        from saklas.core.steering import Steering

        calls: list = []
        s = _MetricStub({"a": _profile_a(), "b": _profile_b()})
        self._patch_project(monkeypatch, calls)
        # Outer scope sets "euclidean"; inner scope ``None`` inherits it.
        outer = Steering(
            alphas=parse_expr("0.5 a|b").alphas,
            projection_metric="euclidean",
        )
        with s.steering(outer):
            with s.steering(parse_expr("0.5 a~b")):
                pass
        # Both calls should run with whitener=None.
        assert calls == [("|", True), ("~", True)], (
            "inner scope without override should inherit outer's "
            "euclidean choice via _resolve_projection_metric"
        )

    def test_invalid_metric_rejected(self, monkeypatch):
        from saklas.core.steering import Steering

        calls: list = []
        s = _MetricStub({"a": _profile_a(), "b": _profile_b()})
        self._patch_project(monkeypatch, calls)
        bad = Steering(
            alphas=parse_expr("0.5 a|b").alphas,
            projection_metric="bogus",
        )
        with pytest.raises(ValueError, match="projection_metric"):
            with s.steering(bad):
                pass
