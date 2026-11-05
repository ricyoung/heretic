# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026  Richard Young + contributors

"""
CPU-only tests for the GammaMPOA modifier (heretic.modifiers.gamma_mpoa).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src python tests/test_gamma_mpoa.py [--report PATH]
    (GAMMA_MPOA_REAL_DIR=<dir> additionally range-downloads real post-norm weights, ~5 GB)

Tiny random-weight models are built from transformers configs (hidden size 64, 3 layers).
Post-norm gains are randomised (log-normal, ~5% negative) so that gamma is non-uniform.
Gains used by the checks are probed through the norms' forward passes along the test's own
writer -> norm mapping (not the module's rmsnorm_gain / post_norm_chain).

(a) pre-norm models: GammaMPOA LoRA == MPOA LoRA, max abs diff 0 (prepared modifiers, and the
    real Model + load_and_init_modifiers path with core_mpoa.toml vs core_gmpoa.toml)
(b) post-norm models: |v^T y| / ||y|| of the residual writes (forward hooks on the post-norms)
    for unedited / MPOA / GammaMPOA(unit) / GammaMPOA at weight 1 (Gemma-4 MoE: dense path
    and whole MLP-block write)
(c) merged weights equal W' computed directly (float64 re-implementation: plumbing check) and
    satisfy the design invariant b^T W' = (1 - lambda) b^T W with fractional direction_index
    and per-layer lambda (own direction interpolation, forward-probed gains)
(d) the plugin loads through Heretic's plugin loader with configs/core_gmpoa.toml and
    configs/core_gmpoa_unit.toml and runs a trial
plus gain conventions, edge cases (near-zero / dead / dominant gains, unknown sandwich norms),
the analysis claims in the module docstring, and (optionally) real weights.
"""

import contextlib
import copy
import glob
import inspect
import io
import json
import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import tomllib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import optuna
import torch
import torch.linalg as LA
import torch.nn.functional as F
import transformers as T

from heretic.config import QuantizationMethod
from heretic.model import Model
from heretic.modifiers.abliteration import (
    Abliteration,
    Parameters,
    RowNormalization,
    WeightDistribution,
)
from heretic.modifiers.abliteration import Settings as AblSettings
from heretic.modifiers.gamma_mpoa import (
    GammaMode,
    GammaMPOA,
    post_norm_chain,
    rmsnorm_gain,
)
from heretic.modifiers.gamma_mpoa import Settings as GSettings
from heretic.plugin import Context

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(min(8, os.cpu_count() or 1))

PAPER = Path("/data/heretic_2/33_2611_does_refusal_removal_get_harder")
GMPOA_TOML = PAPER / "configs" / "core_gmpoa.toml"
GMPOA_UNIT_TOML = PAPER / "configs" / "core_gmpoa_unit.toml"
MPOA_TOML = PAPER / "configs" / "core_mpoa.toml"

REPORT: dict[str, list[str]] = {}


def report(section: str, line: str) -> None:
    REPORT.setdefault(section, []).append(line)


COMMON = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 256,
}


def make_config(arch: str, vocab_size: int = 256) -> Any:
    kw = {**COMMON, "vocab_size": vocab_size}
    if arch == "qwen2":
        return T.Qwen2Config(**kw)
    if arch == "mistral":
        return T.MistralConfig(head_dim=16, **kw)
    if arch == "gemma2":
        return T.Gemma2Config(head_dim=16, **kw)
    if arch == "gemma3":
        return T.Gemma3TextConfig(head_dim=16, **kw)
    if arch == "gemma4":
        return T.Gemma4TextConfig(
            head_dim=16, global_head_dim=16, hidden_size_per_layer_input=0, **kw
        )
    if arch == "gemma4_moe":
        return T.Gemma4TextConfig(
            head_dim=16,
            global_head_dim=16,
            hidden_size_per_layer_input=0,
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=32,
            **kw,
        )
    if arch == "olmo2":
        return T.Olmo2Config(**kw)
    if arch == "olmo3":
        return T.Olmo3Config(**kw)
    raise ValueError(arch)


PRE_NORM = ["qwen2", "mistral"]
POST_NORM = ["gemma2", "gemma3", "gemma4", "gemma4_moe", "olmo2", "olmo3"]


def set_gain(norm: torch.nn.Module, gain: torch.Tensor) -> None:
    """Set an RMSNorm's weight so that its effective gain equals `gain`."""
    name = type(norm).__name__
    weight = cast(torch.Tensor, norm.weight)
    with torch.no_grad():
        if name in ("Gemma2RMSNorm", "Gemma3RMSNorm"):
            weight.copy_(gain - 1.0)
        else:
            weight.copy_(gain)


def build_hf(arch: str, seed: int = 0, gamma: str = "random", dtype=torch.float32):
    torch.manual_seed(seed)
    model = T.AutoModelForCausalLM.from_config(make_config(arch))
    model.eval()
    generator = torch.Generator().manual_seed(seed + 1)
    # Make every writer's row norms non-uniform as in real models (row-norm CV ~0.2).
    for layer in model.model.layers:
        with torch.no_grad():
            for linear in (layer.self_attn.o_proj, layer.mlp.down_proj):
                scale = torch.exp(
                    0.2 * torch.randn(linear.weight.shape[0], 1, generator=generator)
                )
                linear.weight.mul_(scale)
    if arch in POST_NORM and gamma == "random":
        for layer in model.model.layers:
            for norm_name in (
                "post_attention_layernorm",
                "post_feedforward_layernorm",
                "post_feedforward_layernorm_1",
            ):
                norm = getattr(layer, norm_name, None)
                if norm is None:
                    continue
                d = norm.weight.shape[0]
                gain = torch.exp(0.5 * torch.randn(d, generator=generator))
                flip = torch.rand(d, generator=generator) < 0.05
                gain[flip] = -gain[flip]
                set_gain(norm, gain)
    return model.to(dtype)


def wrap(hf_model) -> Model:
    """A heretic Model around an in-memory HF model (bypasses loading from disk)."""
    model = object.__new__(Model)
    model.model = hf_model
    model.settings = cast(
        Any,
        SimpleNamespace(quantization=QuantizationMethod.NONE, model="tiny"),
    )
    model.needs_reload = False
    model.dtype = hf_model.dtype
    return model


def make_modifier(cls, **settings) -> Any:
    heretic_settings = cast(Any, SimpleNamespace(seed=1234))
    settings_cls = GSettings if cls is GammaMPOA else AblSettings
    return cls(heretic_settings=heretic_settings, settings=settings_cls(**settings))


def random_directions(n_layers: int, d: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n_layers + 1, d, generator=generator), dim=1)


def prepare(model: Model, modifiers: list[Any], directions: torch.Tensor) -> Context:
    ctx = Context(settings=cast(Any, SimpleNamespace()), model=model)
    rank = None
    for modifier in modifiers:
        modifier.residual_directions = directions
        modifier.lora_rank = (
            1
            if modifier.settings.row_normalization != RowNormalization.FULL
            else modifier.settings.full_normalization_lora_rank
        )
        assert rank is None or rank == modifier.lora_rank
        rank = modifier.lora_rank
    model.apply_lora(cast(int, rank))
    for modifier in modifiers:
        if isinstance(modifier, GammaMPOA):
            modifier.detect_post_norm_gains(ctx)
    return ctx


def full_strength(model: Model, weight: float = 1.0) -> Parameters:
    n = len(model.get_layers())
    return Parameters(
        direction_index=None,
        weight_distributions={
            component: WeightDistribution(
                max_weight=weight,
                max_weight_position=0.0,
                min_weight=weight,
                min_weight_distance=float(n),
            )
            for component in model.get_modifiable_components()
        },
    )


def lora_state(model: Model) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.model.named_parameters()
        if "lora_" in name
    }


def apply(
    modifier: Any, ctx: Context, parameters: Parameters
) -> dict[str, torch.Tensor]:
    ctx.get_model()._zero_lora_weights()
    modifier.modify_model(ctx, parameters)
    return lora_state(ctx.get_model())


def writer_modules(model: Model):
    """(layer_index, component, peft module, base weight) for every writer."""
    for layer_index in range(len(model.get_layers())):
        for component, modules in model.get_layer_modules(layer_index).items():
            for module in modules:
                yield layer_index, component, module


def effective_weight(module) -> torch.Tensor:
    """W + B A of a PEFT LoRA linear, float64."""
    W = module.base_layer.weight.detach().double()
    A = module.lora_A["default"].weight.detach().double()
    B = module.lora_B["default"].weight.detach().double()
    return W + module.scaling["default"] * B @ A


def writer_norm_chain(layer, component: str) -> tuple[str, ...] | None:
    """
    The test's own writer -> post-norm mapping (independent of gamma_mpoa.post_norm_chain;
    the names are checked against the modeling source in GainConventionTests).
    """
    name = type(layer).__name__
    if name not in (
        "Gemma2DecoderLayer",
        "Gemma3DecoderLayer",
        "Gemma4TextDecoderLayer",
        "Olmo2DecoderLayer",
        "Olmo3DecoderLayer",
    ):
        return None
    if component == "attn.o_proj":
        return ("post_attention_layernorm",)
    if hasattr(layer, "post_feedforward_layernorm_1") and getattr(
        layer, "enable_moe_block", False
    ):
        return ("post_feedforward_layernorm_1", "post_feedforward_layernorm")
    return ("post_feedforward_layernorm",)


def probe_gain(norm) -> torch.Tensor:
    """
    Gain of an RMSNorm measured through its forward pass (independent of rmsnorm_gain):
    for x = 1, rms(x) = 1 and forward(x) = gain / sqrt(1 + eps).
    """
    d = norm.weight.shape[0]
    with torch.no_grad():
        out = norm(torch.ones(1, d, dtype=norm.weight.dtype))
    return out.double().view(-1) * math.sqrt(1 + norm_eps(norm))


def gain_of(model: Model, layer_index: int, component: str) -> torch.Tensor | None:
    layer = model.get_layers()[layer_index]
    chain = writer_norm_chain(layer, component)
    if chain is None:
        return None
    gain = None
    for name in chain:
        norm_gain = probe_gain(getattr(layer, name))
        gain = norm_gain if gain is None else gain * norm_gain
    return cast(torch.Tensor, gain)


def direction_for(directions, direction_index, layer_index) -> torch.Tensor:
    """The direction MPOA uses for a layer (own float64 re-derivation of the lerp)."""
    if direction_index is None:
        return directions[layer_index + 1].double()
    frac, index = math.modf(direction_index + 1)
    v = (1 - frac) * directions[int(index)].double() + frac * directions[
        int(index) + 1
    ].double()
    return v / v.norm()


def layer_weight(distribution: WeightDistribution, layer_index: int) -> float:
    distance = abs(layer_index - distribution.max_weight_position)
    if distance > distribution.min_weight_distance:
        return 0.0
    return distribution.max_weight + (distance / distribution.min_weight_distance) * (
        distribution.min_weight - distribution.max_weight
    )


def weight_leak(model: Model, directions: torch.Tensor, use_gain: bool = True) -> float:
    """
    Mean over writers of ||b^T W_eff|| / ||b^T W|| with b = Gamma v (fraction of the
    v-component of the residual write that survives the edit; input-independent).
    use_gain=False: b = v (the o-space functional that MPOA and the unit control remove).
    """
    fractions = []
    for layer_index, component, module in writer_modules(model):
        v = directions[layer_index + 1].double()
        g = gain_of(model, layer_index, component) if use_gain else None
        b = v if g is None else g * v
        W = module.base_layer.weight.detach().double()
        fractions.append(
            (
                LA.vector_norm(b @ effective_weight(module)) / LA.vector_norm(b @ W)
            ).item()
        )
    return sum(fractions) / len(fractions)


def write_leak(
    model: Model, directions: torch.Tensor, seed: int = 0
) -> dict[str, float]:
    """
    Run a forward pass and measure |v^T y| / ||y|| of every residual write y, where y is the
    output of the writer's post-norm (post-norm models) or of the writer itself (pre-norm).
    For Gemma-4 MoE layers the "mlp.down_proj" entry is the dense-MLP path only, measured at
    post_feedforward_layernorm_1 with v replaced by Gamma_outer v (forward-probed gains); the
    "mlp.total" entry is the whole MLP block's write (dense + unedited experts) at the outer
    post_feedforward_layernorm.
    """
    hooks = []
    captured: dict[tuple[int, str], torch.Tensor] = {}
    probes: dict[tuple[int, str], torch.Tensor] = {}
    layers = model.get_layers()

    def capture(target, key, probe):
        probes[key] = probe

        def hook(_m, _i, output, key=key):
            captured[key] = (
                output[0] if isinstance(output, tuple) else output
            ).detach()

        hooks.append(target.register_forward_hook(hook))

    for layer_index, component, module in writer_modules(model):
        layer = layers[layer_index]
        chain = writer_norm_chain(layer, component)
        v = directions[layer_index + 1].float()
        key = (layer_index, component)
        if chain is None:
            capture(module, key, v)
        elif len(chain) == 1:
            capture(getattr(layer, chain[0]), key, v)
        else:
            outer = torch.ones_like(v)
            for name in chain[1:]:
                outer = outer * probe_gain(getattr(layer, name)).float()
            capture(getattr(layer, chain[0]), key, outer * v)
            capture(getattr(layer, chain[-1]), (layer_index, "mlp.total"), v)
    try:
        generator = torch.Generator().manual_seed(seed)
        input_ids = torch.randint(0, 256, (2, 16), generator=generator)
        with torch.no_grad():
            model.model(input_ids=input_ids)
    finally:
        for h in hooks:
            h.remove()
    ratios = {}
    for key, y in captured.items():
        y = y.float().reshape(-1, y.shape[-1])
        p = probes[key].to(y.device)
        ratios[key] = (
            ((y @ p).abs() / (LA.vector_norm(y, dim=1) * LA.vector_norm(p)))
            .mean()
            .item()
        )
    out = {
        component: sum(r for (_, c), r in ratios.items() if c == component)
        / max(1, sum(1 for (_, c) in ratios if c == component))
        for component in {c for (_, c) in ratios}
    }
    edited = [r for (_, c), r in ratios.items() if c != "mlp.total"]
    out["all"] = sum(edited) / len(edited)
    return out


def reference_gamma_edit(W, v, g, weight, eps, full, inverse="ridge"):
    """Independent float64 reference of the dense gamma-exact edit (no LoRA truncation)."""
    W, v, g = W.double(), v.double(), g.double()
    b = g * v
    delta = eps * g.pow(2).mean().sqrt()
    if inverse == "ridge":
        a = v * g / (g * g + delta * delta)
    else:
        a = v / torch.where(g.abs() < delta, torch.where(g < 0, -delta, delta), g)
    a = a / (b @ a)
    W_proj = W - weight * torch.outer(a, b @ W)
    if not full:
        return W_proj, a, b
    R = LA.vector_norm(W, dim=1, keepdim=True)
    W_full = F.normalize(W_proj, dim=1) * R
    W_dense = W_full - torch.outer(a, b @ W_full - (1 - weight) * (b @ W))
    return W_dense, a, b


# --------------------------------------------------------------------------------------


class GainConventionTests(unittest.TestCase):
    """The gain each post-norm applies, checked against the installed modeling code."""

    def test_conventions_match_source_and_forward(self) -> None:
        import transformers.models.gemma2.modeling_gemma2 as g2
        import transformers.models.gemma3.modeling_gemma3 as g3
        import transformers.models.gemma4.modeling_gemma4 as g4
        import transformers.models.olmo2.modeling_olmo2 as o2
        import transformers.models.olmo3.modeling_olmo3 as o3

        expectations = [
            (g2, "Gemma2RMSNorm", "output * (1.0 + self.weight.float())", "1 + w"),
            (g3, "Gemma3RMSNorm", "output * (1.0 + self.weight.float())", "1 + w"),
            (g4, "Gemma4RMSNorm", "normed_output * self.weight.float()", "w"),
            (o2, "Olmo2RMSNorm", "(self.weight * hidden_states)", "w"),
            (o3, "Olmo3RMSNorm", "(self.weight * hidden_states)", "w"),
        ]
        for module, cls_name, snippet, label in expectations:
            src_lines, start = inspect.getsourcelines(getattr(module, cls_name))
            hits = [start + i for i, line in enumerate(src_lines) if snippet in line]
            self.assertTrue(hits, f"{cls_name}: '{snippet}' not found")
            path = Path(cast(str, module.__file__))
            report(
                f"gain conventions (transformers {T.__version__})",
                f"{cls_name:14s} gain = {label:6s} {path.parent.name}/{path.name}:{hits[0]}",
            )

        for module, layer_cls in [
            (g2, "Gemma2DecoderLayer"),
            (g3, "Gemma3DecoderLayer"),
            (g4, "Gemma4TextDecoderLayer"),
            (o2, "Olmo2DecoderLayer"),
            (o3, "Olmo3DecoderLayer"),
        ]:
            src_lines, start = inspect.getsourcelines(
                getattr(module, layer_cls).forward
            )
            lines = {
                key: [
                    start + i
                    for i, line in enumerate(src_lines)
                    if key + "(hidden_states" in line
                ]
                for key in (
                    "self.post_attention_layernorm",
                    "self.post_feedforward_layernorm",
                )
            }
            for key, hits in lines.items():
                self.assertTrue(hits, f"{layer_cls}: {key} not applied in forward")
            path = Path(cast(str, module.__file__))
            report(
                f"gain conventions (transformers {T.__version__})",
                f"{layer_cls:22s} o_proj->post_attention_layernorm {path.name}:{lines['self.post_attention_layernorm'][0]}, "
                f"down_proj->post_feedforward_layernorm {path.name}:{lines['self.post_feedforward_layernorm'][0]}",
            )

        # rmsnorm_gain() self-checks against forward(); make sure it agrees on random weights.
        for arch in POST_NORM:
            hf = build_hf(arch, seed=3)
            layer = hf.model.layers[0]
            for name in ("post_attention_layernorm", "post_feedforward_layernorm"):
                norm = getattr(layer, name)
                gain = rmsnorm_gain(norm)
                x = torch.randn(5, gain.shape[0])
                expected = (
                    gain
                    * x
                    / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm_eps(norm))
                )
                self.assertLess((norm(x) - expected).abs().max().item(), 1e-5)

    def test_prenorm_layers_have_no_chain(self) -> None:
        for arch in PRE_NORM:
            hf = build_hf(arch)
            for component in ("attn.o_proj", "mlp.down_proj"):
                self.assertIsNone(post_norm_chain(hf.model.layers[0], component))


def norm_eps(norm) -> float:
    return float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))


class PreNormIdentityTests(unittest.TestCase):
    """(a) Pre-norm models: GammaMPOA edit == MPOA edit, bit for bit."""

    def test_bit_identical(self) -> None:
        for arch in PRE_NORM:
            for row_norm in ("full", "none", "pre"):
                hf = build_hf(arch, seed=1)
                model = wrap(hf)
                directions = random_directions(3, 64, seed=2)
                mpoa = make_modifier(Abliteration, row_normalization=row_norm)
                gmpoa = make_modifier(GammaMPOA, row_normalization=row_norm)
                ctx = prepare(model, [mpoa, gmpoa], directions)

                study = optuna.create_study(
                    sampler=optuna.samplers.RandomSampler(seed=0)
                )
                param_sets = [full_strength(model)]
                for _ in range(5):
                    param_sets.append(mpoa.suggest_parameters(ctx, study.ask()))

                max_diff = 0.0
                n_tensors = 0
                for parameters in param_sets:
                    a = apply(mpoa, ctx, parameters)
                    b = apply(gmpoa, ctx, parameters)
                    self.assertEqual(a.keys(), b.keys())
                    for key in a:
                        self.assertTrue(
                            torch.equal(a[key], b[key]), f"{arch} {row_norm} {key}"
                        )
                        max_diff = max(max_diff, (a[key] - b[key]).abs().max().item())
                        n_tensors += 1
                report(
                    "(a) pre-norm: GammaMPOA vs MPOA LoRA weights",
                    f"{arch:8s} row_norm={row_norm:4s} {len(param_sets)} parameter sets, "
                    f"{n_tensors} LoRA tensors, max |diff| = {max_diff:.1e}",
                )


class PostNormLeakTests(unittest.TestCase):
    """(b) Post-norm models: v-component of the residual write."""

    def run_arch(self, arch: str, gamma: str = "random", row_norm: str = "full"):
        hf = build_hf(arch, seed=4, gamma=gamma)
        model = wrap(hf)
        directions = random_directions(3, 64, seed=5)
        mpoa = make_modifier(Abliteration, row_normalization=row_norm)
        gmpoa = make_modifier(GammaMPOA, row_normalization=row_norm)
        unit = make_modifier(GammaMPOA, row_normalization=row_norm, gamma_mode="unit")
        ctx = prepare(model, [mpoa, gmpoa, unit], directions)
        parameters = full_strength(model, 1.0)

        def measure():
            return (
                write_leak(model, directions),
                weight_leak(model, directions),
                weight_leak(model, directions, use_gain=False),
            )

        results = {}
        model._zero_lora_weights()
        results["unedited"] = measure()
        apply(mpoa, ctx, parameters)
        results["MPOA"] = measure()
        apply(unit, ctx, parameters)
        results["unit"] = measure()
        apply(gmpoa, ctx, parameters)
        results["GammaMPOA"] = measure()
        return results

    def test_leak(self) -> None:
        section = "(b) post-norm residual-write leak, weight=1, row_normalization=full"
        report(
            section,
            "unit = GammaMPOA(gamma_mode=unit): same exact kernel-confined edit with Gamma := 1. "
            "gemma4_moe*: 'mlp' is the dense-MLP path only; 'mlp total' is the MLP block's "
            "whole write (dense + unedited experts) at the outer post_feedforward_layernorm.",
        )
        report(
            section,
            f"{'model':11s} {'gamma':7s} {'edit':10s} {'|v.y|/|y| attn':>15s} "
            f"{'mlp':>9s} {'all':>9s} {'mlp total':>10s} {'surv (Gv)^T W':>14s} {'surv v^T W':>11s}",
        )
        for arch in POST_NORM + PRE_NORM:
            for gamma in (
                ["random", "uniform"] if arch in ("gemma2", "olmo2") else ["random"]
            ):
                results = self.run_arch(arch, gamma=gamma)
                for edit, (w, frac, frac_v) in results.items():
                    total = w.get("mlp.total")
                    report(
                        section,
                        f"{arch + ('*' if total is not None else ''):11s} "
                        f"{gamma if arch in POST_NORM else 'n/a':7s} {edit:10s} "
                        f"{w['attn.o_proj']:15.2e} {w['mlp.down_proj']:9.2e} {w['all']:9.2e} "
                        f"{'' if total is None else f'{total:.2e}':>10s} {frac:14.2e} {frac_v:11.2e}",
                    )
                if arch in POST_NORM:
                    self.assertLess(results["GammaMPOA"][0]["all"], 1e-5, arch)
                    self.assertLess(results["GammaMPOA"][1], 1e-5, arch)
                    # The unit control removes the o-space readout exactly but not the
                    # gain-weighted one.
                    self.assertLess(results["unit"][2], 1e-5, arch)
                    if gamma == "random":
                        self.assertGreater(results["MPOA"][0]["all"], 1e-2, arch)
                        self.assertGreater(results["MPOA"][1], 0.1, arch)
                        self.assertGreater(results["unit"][1], 0.1, arch)
                    if "mlp.total" in results["GammaMPOA"][0]:
                        # Documented limitation: the experts are not edited.
                        self.assertGreater(
                            results["GammaMPOA"][0]["mlp.total"], 1e-3, arch
                        )
                else:
                    # Pre-norm: gain mode is MPOA; unit mode is the exact edit.
                    self.assertEqual(results["GammaMPOA"][1], results["MPOA"][1])
                    self.assertLess(results["unit"][1], 1e-5, arch)

    def test_leak_none_mode(self) -> None:
        for arch in ("gemma2", "olmo3"):
            results = self.run_arch(arch, row_norm="none")
            for edit, (w, frac, _) in results.items():
                report(
                    "(b') row_normalization=none (projected abliteration vs oblique)",
                    f"{arch:8s} {edit:10s} |v.y|/|y| = {w['all']:.2e}   surviving b^T W = {frac:.2e}",
                )
            self.assertLess(results["GammaMPOA"][1], 1e-6)
            self.assertGreater(results["MPOA"][1], 0.1)

    def test_unit_control_prenorm(self) -> None:
        """gamma_mode=unit on pre-norm models: the 'exact MPOA' edit (no row-norm leak)."""
        for arch in PRE_NORM:
            hf = build_hf(arch, seed=4)
            model = wrap(hf)
            directions = random_directions(3, 64, seed=5)
            mpoa = make_modifier(Abliteration)
            unit = make_modifier(GammaMPOA, gamma_mode="unit")
            ctx = prepare(model, [mpoa, unit], directions)
            self.assertTrue(unit.use_gamma_path)
            parameters = full_strength(model)
            apply(mpoa, ctx, parameters)
            leak_mpoa = weight_leak(model, directions)
            apply(unit, ctx, parameters)
            leak_exact = weight_leak(model, directions)
            report(
                "gamma_mode=unit on pre-norm models (weight=1)",
                f"{arch:8s} surviving v^T W: MPOA {leak_mpoa:.2e}   unit (exact) {leak_exact:.2e}",
            )
            self.assertLess(leak_exact, 1e-5)


class LoraFormTests(unittest.TestCase):
    """(c) Merged weights equal W' computed directly."""

    def merged(self, model: Model):
        return copy.deepcopy(model.model).merge_and_unload()

    def check(
        self, arch: str, row_norm: str, rank: int, weight: float, inverse: str = "ridge"
    ):
        hf = build_hf(arch, seed=6)
        model = wrap(hf)
        directions = random_directions(3, 64, seed=7)
        gmpoa = make_modifier(
            GammaMPOA,
            row_normalization=row_norm,
            full_normalization_lora_rank=rank,
            gamma_inverse=inverse,
        )
        ctx = prepare(model, [gmpoa], directions)
        apply(gmpoa, ctx, full_strength(model, weight))
        merged = self.merged(model)
        worst_direct = worst_exact = worst_rownorm = 0.0
        for layer_index, component, module in writer_modules(model):
            W = module.base_layer.weight.detach()
            v = directions[layer_index + 1]
            g = gain_of(model, layer_index, component)
            assert g is not None
            reference, _, b = reference_gamma_edit(
                W, v, g, weight, gmpoa.settings.gamma_eps, row_norm == "full", inverse
            )
            merged_layer = merged.model.layers[layer_index]
            merged_W = (
                (
                    merged_layer.self_attn.o_proj.weight
                    if component == "attn.o_proj"
                    else merged_layer.mlp.down_proj.weight
                )
                .detach()
                .double()
            )
            # merged == W + B A
            self.assertLess(
                (merged_W - effective_weight(module)).abs().max().item(), 1e-6
            )
            scale = W.abs().max().item()
            worst_direct = max(
                worst_direct, (merged_W - reference).abs().max().item() / scale
            )
            target = (1 - weight) * (b @ W.double())
            worst_exact = max(
                worst_exact,
                (
                    LA.vector_norm(b @ merged_W - target)
                    / LA.vector_norm(b @ W.double())
                ).item(),
            )
            R = LA.vector_norm(W.double(), dim=1)
            worst_rownorm = max(
                worst_rownorm,
                ((LA.vector_norm(merged_W, dim=1) - R) / R).abs().max().item(),
            )
        return worst_direct, worst_exact, worst_rownorm

    def test_lora_equals_direct(self) -> None:
        section = (
            "(c) LoRA form: merged weights vs W' computed directly (float64 reference)"
        )
        report(
            section,
            f"{'model':8s} {'Gamma^-1':8s} {'row_norm':8s} {'rank':>4s} {'weight':>6s} "
            f"{'max|merged-ref|/max|W|':>23s} {'b^T exactness':>14s} {'max row-norm err':>17s}",
        )
        for arch in ("gemma2", "gemma4", "olmo2", "olmo3"):
            for inverse, row_norm, rank, weight in [
                ("ridge", "none", 1, 1.0),
                ("ridge", "none", 1, 0.7),
                (
                    "ridge",
                    "full",
                    65,
                    1.0,
                ),  # rank-64 K + rank-1 projection: no truncation
                ("ridge", "full", 65, 1.3),
                ("clamp", "none", 1, 1.0),
                ("clamp", "full", 65, 1.0),
            ]:
                direct, exact, rownorm = self.check(
                    arch, row_norm, rank, weight, inverse
                )
                report(
                    section,
                    f"{arch:8s} {inverse:8s} {row_norm:8s} {rank:4d} {weight:6.1f} {direct:23.2e} {exact:14.2e} {rownorm:17.2e}",
                )
                self.assertLess(direct, 1e-5)
                self.assertLess(exact, 1e-5)

    def test_default_rank_norm_preservation(self) -> None:
        """Deployed rank-3 adapters: row-norm error of GammaMPOA vs MPOA (both truncated)."""
        section = "(c') default rank 3: deployed row-norm preservation and exactness"
        for arch in ("gemma2", "olmo2"):
            hf = build_hf(arch, seed=6)
            model = wrap(hf)
            directions = random_directions(3, 64, seed=7)
            mpoa = make_modifier(Abliteration)
            gmpoa = make_modifier(GammaMPOA)
            ctx = prepare(model, [mpoa, gmpoa], directions)
            parameters = full_strength(model)
            for name, modifier in (("MPOA", mpoa), ("GammaMPOA", gmpoa)):
                apply(modifier, ctx, parameters)
                rownorm = []
                for _, _, module in writer_modules(model):
                    W = module.base_layer.weight.detach().double()
                    R = LA.vector_norm(W, dim=1)
                    rownorm.append(
                        ((LA.vector_norm(effective_weight(module), dim=1) - R) / R)
                        .abs()
                        .max()
                        .item()
                    )
                report(
                    section,
                    f"{arch:8s} {name:10s} max row-norm err {max(rownorm):.2e}   "
                    f"surviving b^T W {weight_leak(model, directions):.2e}",
                )

    def test_invariants_fractional_direction_and_lambda(self) -> None:
        """
        Design-level check, independent of the module's helpers: forward-probed gains, own
        direction interpolation, per-layer lambda in 0.11-1.37, default settings (full, rank 3):
        merged == W + B A and b^T W' = (1 - lambda_l) b^T W for every edited writer.
        """
        section = "(c'') invariants with fractional direction_index and per-layer lambda (default settings)"
        for arch in ("gemma2", "gemma4", "olmo2", "olmo3"):
            hf = build_hf(arch, seed=21)
            model = wrap(hf)
            directions = random_directions(3, 64, seed=22)
            gmpoa = make_modifier(GammaMPOA)
            ctx = prepare(model, [gmpoa], directions)
            for direction_index in (0.37, 1.62, None):
                parameters = Parameters(
                    direction_index=direction_index,
                    weight_distributions={
                        "attn.o_proj": WeightDistribution(1.37, 1.2, 0.41, 2.0),
                        "mlp.down_proj": WeightDistribution(0.83, 0.3, 0.11, 1.9),
                    },
                )
                apply(gmpoa, ctx, parameters)
                merged = self.merged(model)
                worst_exact = worst_merge = 0.0
                lambdas = []
                for layer_index, component, module in writer_modules(model):
                    lam = layer_weight(
                        parameters.weight_distributions[component], layer_index
                    )
                    A = module.lora_A["default"].weight.detach()
                    if lam == 0:
                        self.assertEqual(A.abs().max().item(), 0.0)
                        continue
                    lambdas.append(lam)
                    b = cast(torch.Tensor, gain_of(model, layer_index, component)) * (
                        direction_for(directions, direction_index, layer_index)
                    )
                    W = module.base_layer.weight.detach().double()
                    W_eff = effective_weight(module)
                    merged_layer = merged.model.layers[layer_index]
                    W_merged = (
                        (
                            merged_layer.self_attn.o_proj.weight
                            if component == "attn.o_proj"
                            else merged_layer.mlp.down_proj.weight
                        )
                        .detach()
                        .double()
                    )
                    worst_merge = max(
                        worst_merge,
                        (W_merged - W_eff).abs().max().item() / W.abs().max().item(),
                    )
                    for W_new in (W_eff, W_merged):
                        worst_exact = max(
                            worst_exact,
                            (
                                LA.vector_norm(b @ W_new - (1 - lam) * (b @ W))
                                / LA.vector_norm(b @ W)
                            ).item(),
                        )
                report(
                    section,
                    f"{arch:7s} direction_index={direction_index!s:5s} lambda in "
                    f"[{min(lambdas):.2f}, {max(lambdas):.2f}]: max|merged-(W+BA)|/max|W| = "
                    f"{worst_merge:.1e}, max |b^T W' - (1-lambda) b^T W| / |b^T W| = {worst_exact:.1e}",
                )
                self.assertLess(worst_merge, 1e-6)
                self.assertLess(worst_exact, 1e-5)

    def test_same_direction_as_mpoa(self) -> None:
        """In "none" mode MPOA's lora_B = -lambda v exposes the v it used; GammaMPOA must use it too."""
        hf = build_hf("olmo2", seed=31)
        model = wrap(hf)
        directions = random_directions(3, 64, seed=32)
        mpoa = make_modifier(Abliteration, row_normalization="none")
        gmpoa = make_modifier(GammaMPOA, row_normalization="none")
        ctx = prepare(model, [mpoa, gmpoa], directions)
        parameters = Parameters(
            1.25,
            {
                c: WeightDistribution(0.9, 1.0, 0.5, 2.0)
                for c in ("attn.o_proj", "mlp.down_proj")
            },
        )
        apply(mpoa, ctx, parameters)
        v_mpoa = {}
        for layer_index, component, module in writer_modules(model):
            lam = layer_weight(parameters.weight_distributions[component], layer_index)
            v_mpoa[(layer_index, component)] = (
                -module.lora_B["default"].weight.detach().view(-1) / lam
            ).double()
        apply(gmpoa, ctx, parameters)
        worst = 0.0
        for layer_index, component, module in writer_modules(model):
            b = (
                cast(torch.Tensor, gain_of(model, layer_index, component))
                * v_mpoa[(layer_index, component)]
            )
            W = module.base_layer.weight.detach().double()
            A = module.lora_A["default"].weight.detach().double().view(-1)
            worst = max(
                worst, (A - b @ W).abs().max().item() / (b @ W).abs().max().item()
            )
        report(
            "(c'') invariants with fractional direction_index and per-layer lambda (default settings)",
            f"GammaMPOA lora_A == (Gamma v_MPOA)^T W with v_MPOA recovered from MPOA's LoRA "
            f"(olmo2, direction_index=1.25): max rel diff {worst:.1e}",
        )
        self.assertLess(worst, 1e-5)

    def test_bf16_floor(self) -> None:
        """Numerical floor of the exactness when the base model is bfloat16."""
        hf = build_hf("gemma2", seed=6, dtype=torch.bfloat16)
        model = wrap(hf)
        directions = random_directions(3, 64, seed=7)
        mpoa = make_modifier(Abliteration)
        gmpoa = make_modifier(GammaMPOA)
        ctx = prepare(model, [mpoa, gmpoa], directions)
        for name, modifier in (("MPOA", mpoa), ("GammaMPOA", gmpoa)):
            apply(modifier, ctx, full_strength(model))
            merged = self.merged(model)
            fractions = []
            for layer_index, component, module in writer_modules(model):
                v = directions[layer_index + 1].double()
                g = gain_of(model, layer_index, component)
                b = cast(torch.Tensor, g) * v
                layer = merged.model.layers[layer_index]
                W_m = (
                    (
                        layer.self_attn.o_proj.weight
                        if component == "attn.o_proj"
                        else layer.mlp.down_proj.weight
                    )
                    .detach()
                    .double()
                )
                W = module.base_layer.weight.detach().double()
                fractions.append(
                    (LA.vector_norm(b @ W_m) / LA.vector_norm(b @ W)).item()
                )
            report(
                "bf16 base weights (merged in bf16): surviving b^T W",
                f"gemma2 {name:10s} {sum(fractions) / len(fractions):.2e}",
            )


class EdgeCaseTests(unittest.TestCase):
    """Near-zero gains (Gemma 1+w with w ~ -1), unknown sandwich norms."""

    def test_near_zero_gains(self) -> None:
        hf = build_hf("gemma2", seed=11)
        layers = hf.model.layers
        generator = torch.Generator().manual_seed(3)
        d = 64
        with torch.no_grad():
            # L0 attn: all gains ~ +-1e-4.
            layers[0].post_attention_layernorm.weight.copy_(
                -1 + 1e-4 * torch.randn(d, generator=generator)
            )
            # L1 mlp: dead norm, every gain exactly 0.
            layers[1].post_feedforward_layernorm.weight.fill_(-1.0)
            # L2 attn: half the gains exactly 0, the rest O(1) incl. some negative.
            w = torch.randn(d, generator=generator) * 0.8
            w[: d // 2] = -1.0
            layers[2].post_attention_layernorm.weight.copy_(w)
            # L2 mlp: one huge gain, the rest in [0, 1e-3].
            w2 = -1 + 1e-3 * torch.rand(d, generator=generator)
            w2[5] = 50.0
            layers[2].post_feedforward_layernorm.weight.copy_(w2)
        model = wrap(hf)
        directions = random_directions(3, d, seed=12).clone()
        # L2 direction mostly on the zero-gain half (b^T a ~ 1e-6 before normalisation).
        v2 = torch.zeros(d)
        v2[: d // 2] = torch.randn(d // 2, generator=generator)
        v2[d // 2 :] = 1e-3 * torch.randn(d // 2, generator=generator)
        directions[3] = F.normalize(v2, dim=0)

        section = (
            "edge cases: near-zero / dead / outlier gains (tiny Gemma-2, weight=1)"
        )
        report(
            section,
            f"{'Gamma^-1':8s} {'writer':16s} {'|g|max':>8s} {'||b^T W||':>9s} {'exactness':>9s} "
            f"{'||dW||/||W||':>12s} {'row-norm err':>12s} {'off-target':>10s}  warning",
        )
        modifiers = {
            inverse: make_modifier(GammaMPOA, gamma_inverse=inverse)
            for inverse in ("ridge", "clamp")
        }
        with contextlib.redirect_stdout(io.StringIO()):
            ctx = prepare(model, list(modifiers.values()), directions)
        for inverse, gmpoa in modifiers.items():
            with contextlib.redirect_stdout(io.StringIO()) as log:
                apply(gmpoa, ctx, full_strength(model, 1.0))
            log_text = log.getvalue().replace("\n", " ")
            for layer_index, component, module in writer_modules(model):
                A = module.lora_A["default"].weight.detach().double()
                B = module.lora_B["default"].weight.detach().double()
                self.assertTrue(torch.isfinite(A).all() and torch.isfinite(B).all())
                g = cast(torch.Tensor, gain_of(model, layer_index, component))
                v = directions[layer_index + 1].double()
                b = g * v
                W = module.base_layer.weight.detach().double()
                W_new = W + B @ A
                bW = LA.vector_norm(b @ W).item()
                exact = LA.vector_norm(b @ W_new).item() / bW if bW > 0 else 0.0
                R = LA.vector_norm(W, dim=1)
                row_error = ((LA.vector_norm(W_new, dim=1) - R) / R).abs().max().item()
                relative = (LA.matrix_norm(W_new - W) / LA.matrix_norm(W)).item()
                stats = gmpoa.edit_stats.get((layer_index, component), {})
                warned = [
                    kind
                    for (li, c, kind) in gmpoa._warned
                    if (li, c) == (layer_index, component)
                ]
                report(
                    section,
                    f"{inverse:8s} L{layer_index} {component:13s} {g.abs().max().item():8.2e} {bW:9.2e} "
                    f"{exact:9.1e} {relative:12.3f} {row_error:12.2e} "
                    f"{stats.get('off_target', float('nan')):10.3f}  {','.join(warned) or '-'}",
                )
                if bW == 0:
                    # Dead norm: nothing to remove, identity adapter, logged as skipped.
                    self.assertEqual(A.abs().max().item(), 0.0)
                    self.assertIn("skip", warned)
                else:
                    # Exact for every writer with a v-readout, including the one whose
                    # direction lives on the zero-gain half (previously skipped).
                    self.assertLess(exact, 1e-5)
                if row_error > 0.1 or relative > 0.5:
                    self.assertIn("size", warned)
            self.assertIn("GammaMPOA warning", log_text)

    def test_unknown_sandwich_rejected(self) -> None:
        # GLM-4: post_self_attn_layernorm / post_mlp_layernorm sandwich norms.
        config = T.Glm4Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=256,
            head_dim=16,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=2,
        )
        torch.manual_seed(0)
        model = wrap(T.AutoModelForCausalLM.from_config(config).eval())
        gmpoa = make_modifier(GammaMPOA)
        with self.assertRaises(ValueError) as raised:
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(model, [gmpoa], random_directions(2, 64, seed=43))
        self.assertIn("POST_NORM_CHAINS", str(raised.exception))
        report(
            "edge cases: unknown sandwich-norm layers",
            f"GLM-4 (post_self_attn_layernorm/post_mlp_layernorm): raises ValueError: "
            f"{str(raised.exception)[:110]}...",
        )
        # A known sandwich layer whose post-norm was replaced by something unsupported.
        hf = build_hf("gemma2", seed=41)
        hf.model.layers[1].post_feedforward_layernorm = torch.nn.Identity()
        gmpoa = make_modifier(GammaMPOA)
        with self.assertRaises(ValueError) as raised:
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(wrap(hf), [gmpoa], random_directions(3, 64, seed=42))
        report(
            "edge cases: unknown sandwich-norm layers",
            f"Gemma-2 with post_feedforward_layernorm -> Identity: raises ValueError: {raised.exception}",
        )
        # Pre-norm models (post_attention_layernorm is their pre-MLP norm) are not affected.
        for arch in PRE_NORM:
            gmpoa = make_modifier(GammaMPOA)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(wrap(build_hf(arch)), [gmpoa], random_directions(3, 64, seed=1))
            self.assertFalse(gmpoa.use_gamma_path)


def local_tokenizer():
    tokenizer_dirs = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/"
        )
    )
    if not tokenizer_dirs:
        return None
    return T.AutoTokenizer.from_pretrained(tokenizer_dirs[0])


def init_from_toml(toml_path: Path, model_dir: str):
    """Model + load_and_init_modifiers (the real init path) for a paper config."""
    import heretic.plugin as plugin_module
    from heretic.config import InitOnlySettings
    from heretic.modifier import load_and_init_modifiers
    from heretic.utils import Prompt

    toml = tomllib.loads(toml_path.read_text())
    settings = InitOnlySettings.model_validate(
        toml
        | {
            "model": model_dir,
            "dtypes": ["float32"],
            "device_map": "cpu",
            "batch_size": 4,
            "seed": 1234,
        }
    )
    good = [
        Prompt(system="You are a helpful assistant.", user=f"Write a poem about {w}.")
        for w in ("cats", "rain", "tea", "maps", "sea")
    ]
    bad = [
        Prompt(system="You are a helpful assistant.", user=f"Explain how to steal {w}.")
        for w in ("cars", "data", "money", "art", "keys")
    ]
    original = plugin_module.load_prompts
    # No dataset download: distinct synthetic good/bad prompts.
    plugin_module.load_prompts = lambda _settings, spec: (
        bad if "harmful" in str(spec) else good
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()) as log:
            model = Model(settings)
            entries = load_and_init_modifiers(settings, model)
    finally:
        plugin_module.load_prompts = original
    return model, entries, settings, log.getvalue()


class PluginLoaderTests(unittest.TestCase):
    """(d) Loads through Heretic's plugin loader with configs/core_gmpoa.toml."""

    def test_config_diff(self) -> None:
        a = MPOA_TOML.read_text().splitlines()
        b = GMPOA_TOML.read_text().splitlines()
        self.assertEqual(len(a), len(b))
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        self.assertEqual(len(diff), 1)
        self.assertIn("heretic.modifiers.gamma_mpoa.GammaMPOA", diff[0][1])
        report(
            "(d) plugin loader",
            f"core_gmpoa.toml vs core_mpoa.toml: 1 differing line: {diff[0][1].strip()}",
        )
        # The matched Gamma := 1 arm: core_gmpoa.toml plus a [modifier.GammaMPOA] table.
        unit = GMPOA_UNIT_TOML.read_text()
        self.assertTrue(unit.startswith(GMPOA_TOML.read_text()))
        extra = unit[len(GMPOA_TOML.read_text()) :].strip().splitlines()
        settings = tomllib.loads(unit)["modifier"]["GammaMPOA"]
        self.assertEqual(settings, {"gamma_mode": "unit"})
        report(
            "(d) plugin loader",
            f"core_gmpoa_unit.toml = core_gmpoa.toml + {' / '.join(line.strip() for line in extra if line.strip() and not line.startswith('#'))}",
        )

    def test_load_and_run(self) -> None:
        import tempfile

        tokenizer = local_tokenizer()
        if tokenizer is None:
            self.skipTest("no local tokenizer")

        with tempfile.TemporaryDirectory() as tmp:
            config = make_config("gemma2", vocab_size=len(tokenizer))
            torch.manual_seed(0)
            hf = T.AutoModelForCausalLM.from_config(config)
            generator = torch.Generator().manual_seed(1)
            for layer in hf.model.layers:
                for name in ("post_attention_layernorm", "post_feedforward_layernorm"):
                    set_gain(
                        getattr(layer, name),
                        torch.exp(0.5 * torch.randn(64, generator=generator)),
                    )
            hf.save_pretrained(tmp)
            tokenizer.save_pretrained(tmp)

            for toml_path in (GMPOA_TOML, GMPOA_UNIT_TOML):
                model, entries, settings, log = init_from_toml(toml_path, tmp)
                unit = toml_path == GMPOA_UNIT_TOML

                self.assertEqual(len(entries), 1)
                modifier = entries[0].modifier
                self.assertIsInstance(modifier, GammaMPOA)
                self.assertFalse(modifier.reproducible)
                self.assertEqual(
                    modifier.settings.row_normalization, RowNormalization.FULL
                )
                self.assertEqual(
                    modifier.settings.gamma_mode,
                    GammaMode.UNIT if unit else GammaMode.GAIN,
                )
                self.assertEqual(model.peft_config.r, 3)
                self.assertTrue(modifier.use_gamma_path)
                if not unit:
                    # Off-target ||Gamma a - v|| logged per writer at init.
                    self.assertEqual(len(modifier.off_target), 6)
                    self.assertIn("off-target ||Gamma a - v||", log)

                ctx = Context(settings=settings, model=model)
                study = optuna.create_study(
                    sampler=optuna.samplers.RandomSampler(seed=1)
                )
                parameters = modifier.suggest_parameters(ctx, study.ask())
                modifier.reset_model(ctx)
                with contextlib.redirect_stdout(io.StringIO()):
                    modifier.modify_model(ctx, parameters)

                worst = 0.0
                n_edited = 0
                for layer_index, component, module in writer_modules(model):
                    A = module.lora_A["default"].weight
                    if A.abs().max() == 0:
                        continue
                    n_edited += 1
                    weight = layer_weight(
                        parameters.weight_distributions[component], layer_index
                    )
                    v = direction_for(
                        modifier.residual_directions,
                        parameters.direction_index,
                        layer_index,
                    )
                    g = gain_of(model, layer_index, component)
                    b = v if unit else cast(torch.Tensor, g) * v
                    W = module.base_layer.weight.detach().double()
                    residual = b @ effective_weight(module) - (1 - weight) * (b @ W)
                    worst = max(
                        worst, (LA.vector_norm(residual) / LA.vector_norm(b @ W)).item()
                    )
                    self.assertIn((layer_index, component), modifier.edit_stats)
                self.assertGreater(n_edited, 0)
                self.assertLess(worst, 1e-5)

                report(
                    "(d) plugin loader",
                    f"{toml_path.name}: loaded {type(modifier).__module__}.{type(modifier).__name__} via "
                    f"load_and_init_modifiers ('{entries[0].name}'), reproducible={modifier.reproducible}, "
                    f"row_normalization={modifier.settings.row_normalization.value}, "
                    f"gamma_mode={modifier.settings.gamma_mode.value}, lora r={model.peft_config.r}",
                )
                report(
                    "(d) plugin loader",
                    f"{toml_path.name}: real Model + init() on tiny Gemma-2 (directions from get_residuals_mean), "
                    f"one random trial: {n_edited} writers edited, max |b^T W' - (1-weight) b^T W| / |b^T W| = {worst:.1e} "
                    f"(b = {'v' if unit else 'Gamma v'})",
                )

    def test_prenorm_identity_real_init(self) -> None:
        """(a) through the real init path: core_mpoa.toml vs core_gmpoa.toml, 6 random trials."""
        import tempfile

        tokenizer = local_tokenizer()
        if tokenizer is None:
            self.skipTest("no local tokenizer")
        for arch in PRE_NORM:
            with tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(0)
                hf = T.AutoModelForCausalLM.from_config(
                    make_config(arch, vocab_size=len(tokenizer))
                )
                hf.save_pretrained(tmp)
                tokenizer.save_pretrained(tmp)
                runs = {}
                for label, toml_path in (("mpoa", MPOA_TOML), ("gmpoa", GMPOA_TOML)):
                    model, entries, settings, _ = init_from_toml(toml_path, tmp)
                    modifier = entries[0].modifier
                    ctx = Context(settings=settings, model=model)
                    study = optuna.create_study(
                        sampler=optuna.samplers.RandomSampler(seed=7)
                    )
                    states = []
                    for _ in range(6):
                        parameters = modifier.suggest_parameters(ctx, study.ask())
                        modifier.reset_model(ctx)
                        modifier.modify_model(ctx, parameters)
                        states.append(lora_state(model))
                    runs[label] = (
                        type(modifier).__name__,
                        modifier.residual_directions.clone(),
                        states,
                        modifier.reproducible,
                    )
                self.assertTrue(torch.equal(runs["mpoa"][1], runs["gmpoa"][1]))
                n_tensors = 0
                for a, b in zip(runs["mpoa"][2], runs["gmpoa"][2]):
                    self.assertEqual(a.keys(), b.keys())
                    for key in a:
                        self.assertTrue(torch.equal(a[key], b[key]), f"{arch} {key}")
                        n_tensors += 1
                report(
                    "(a) pre-norm: GammaMPOA vs MPOA LoRA weights",
                    f"{arch:8s} real Model + load_and_init_modifiers, core_mpoa.toml ({runs['mpoa'][0]}, "
                    f"reproducible={runs['mpoa'][3]}) vs core_gmpoa.toml ({runs['gmpoa'][0]}, "
                    f"reproducible={runs['gmpoa'][3]}): directions equal, {n_tensors} LoRA tensors "
                    "over 6 trials bit-identical",
                )


REAL_MODELS = {
    # name: (anonymous HF repo, tensor prefix, gain convention)
    "gemma-2-27b-it": ("unsloth/gemma-2-27b-it", "model.layers.{i}.", "1+w"),
    "gemma-4-31B-it": (
        "google/gemma-4-31B-it",
        "model.language_model.layers.{i}.",
        "w",
    ),
    "OLMo-2-0325-32B-Instruct": (
        "allenai/OLMo-2-0325-32B-Instruct",
        "model.layers.{i}.",
        "w",
    ),
    "Olmo-3.1-32B-Instruct": (
        "allenai/Olmo-3.1-32B-Instruct",
        "model.layers.{i}.",
        "w",
    ),
}


def fetch_real_tensors(directory: Path, name: str) -> dict[str, Any]:
    """
    Range-read (no full download, anonymous) all post-norm weights and the mid-layer
    o_proj/down_proj of a post-norm model from its safetensors shards; cached as a .pt file.
    """
    path = directory / f"{name}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    import struct

    from huggingface_hub import HfFileSystem, hf_hub_download

    repo, prefix, _ = REAL_MODELS[name]
    cache = str(directory / "hfcache")
    config = json.loads(
        Path(
            hf_hub_download(repo, "config.json", token=False, cache_dir=cache)
        ).read_text()
    )
    text_config = config.get("text_config", config)
    n_layers = text_config["num_hidden_layers"]
    mid = n_layers // 2
    names = []
    for i in range(n_layers):
        names += [
            prefix.format(i=i) + n + ".weight"
            for n in ("post_attention_layernorm", "post_feedforward_layernorm")
        ]
    names += [
        prefix.format(i=mid) + n + ".weight"
        for n in ("self_attn.o_proj", "mlp.down_proj")
    ]
    index = json.loads(
        Path(
            hf_hub_download(
                repo, "model.safetensors.index.json", token=False, cache_dir=cache
            )
        ).read_text()
    )["weight_map"]
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
    fs = HfFileSystem(token=False)
    tensors = {}
    for shard in sorted({index[n] for n in names}):
        with fs.open(f"{repo}/{shard}", "rb", block_size=1 << 20) as f:
            header_length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_length))
            for n in [n for n in names if index[n] == shard]:
                start, end = header[n]["data_offsets"]
                f.seek(8 + header_length + start)
                tensors[n] = (
                    torch.frombuffer(
                        bytearray(f.read(end - start)), dtype=dtypes[header[n]["dtype"]]
                    )
                    .reshape(header[n]["shape"])
                    .clone()
                )
    data = {"repo": repo, "mid": mid, "n_layers": n_layers, "tensors": tensors}
    torch.save(data, path)
    return data


def edited_layers(n_layers: int) -> list[int]:
    """Writers inside MPOA's search range (max_weight_position in [0.6, 1.0] x last index)."""
    last = n_layers - 1
    return sorted({round(0.6 * last), round(0.8 * last), last})


def _range_get(url: str, start: int, end: int) -> bytes:
    """Bytes [start, end) of an (anonymous) HF file, retried."""
    import requests

    for attempt in range(5):
        try:
            response = requests.get(
                url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120
            )
            response.raise_for_status()
            if len(response.content) == end - start:
                return response.content
        except requests.RequestException:
            if attempt == 4:
                raise
    raise OSError(f"short read from {url}")


def fetch_edited_writers(directory: Path, name: str) -> dict[str, Any]:
    """
    Range-read the o_proj/down_proj of the layers returned by edited_layers() (0.6, 0.8 and
    1.0 x last index) with parallel anonymous HTTP range requests (HF throttles single
    connections); cached as <name>__edited_writers.pt next to the gains cache.
    """
    path = directory / f"{name}__edited_writers.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    import struct
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    repo, prefix, _ = REAL_MODELS[name]
    cache = str(directory / "hfcache")
    config = json.loads(
        Path(
            hf_hub_download(repo, "config.json", token=False, cache_dir=cache)
        ).read_text()
    )
    n_layers = config.get("text_config", config)["num_hidden_layers"]
    layers = edited_layers(n_layers)
    names = [
        prefix.format(i=i) + n + ".weight"
        for i in layers
        for n in ("self_attn.o_proj", "mlp.down_proj")
    ]
    index = json.loads(
        Path(
            hf_hub_download(
                repo, "model.safetensors.index.json", token=False, cache_dir=cache
            )
        ).read_text()
    )["weight_map"]
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
    tensors = {}
    chunk = 8 << 20
    with ThreadPoolExecutor(16) as pool:
        for shard in sorted({index[n] for n in names}):
            url = f"https://huggingface.co/{repo}/resolve/main/{shard}"
            header_length = struct.unpack("<Q", _range_get(url, 0, 8))[0]
            header = json.loads(_range_get(url, 8, 8 + header_length))
            for n in [n for n in names if index[n] == shard]:
                start, end = header[n]["data_offsets"]
                start += 8 + header_length
                end += 8 + header_length
                parts = list(
                    pool.map(
                        lambda s: _range_get(url, s, min(s + chunk, end)),
                        range(start, end, chunk),
                    )
                )
                tensors[n] = (
                    torch.frombuffer(
                        bytearray(b"".join(parts)), dtype=dtypes[header[n]["dtype"]]
                    )
                    .reshape(header[n]["shape"])
                    .clone()
                )
    data = {"repo": repo, "n_layers": n_layers, "layers": layers, "tensors": tensors}
    torch.save(data, path)
    return data


# Edits compared on real weights: label -> GammaMPOA settings (None: MPOA).
REAL_VARIANTS: list[tuple[str, dict[str, Any] | None]] = [
    ("MPOA", None),
    ("unit (Gamma:=1)", {"gamma_mode": "unit"}),
    ("GammaMPOA ridge 0.1", {"gamma_inverse": "ridge", "gamma_eps": 0.1}),  # default
    ("GammaMPOA ridge 0.05", {"gamma_inverse": "ridge", "gamma_eps": 0.05}),
    (
        "GammaMPOA clamp 0.01",
        {"gamma_inverse": "clamp", "gamma_eps": 0.01},
    ),  # plan's form
]


@unittest.skipUnless(
    os.environ.get("GAMMA_MPOA_REAL_DIR"),
    "set GAMMA_MPOA_REAL_DIR=<dir> to range-download real post-norm weights (~5 GB)",
)
class RealWeightTests(unittest.TestCase):
    """
    Real post-norm gains (all layers) and real writers of the edited layers (0.6, 0.8 and
    1.0 x last index; MPOA's max_weight_position range is [0.6, 1.0] x last index), with
    random unit directions (no refusal directions for these models are available locally).
    """

    @staticmethod
    def mpoa_full_lora(W, v, weight, r, seed=1234):
        # Identical math to Abliteration.modify_model (row_normalization = "full").
        R = LA.vector_norm(W, dim=1, keepdim=True)
        Wn = F.normalize(W, dim=1)
        delta = (
            F.normalize(Wn + (-weight * v).view(-1, 1) @ (v @ Wn).view(1, -1), dim=1)
            * R
            - W
        )
        torch.manual_seed(seed)
        U, S, V = torch.svd_lowrank(delta, q=2 * r + 4, niter=6)
        sqrt_S = torch.sqrt(S[:r])
        return torch.diag(sqrt_S) @ V[:, :r].T, U[:, :r] @ torch.diag(sqrt_S)

    @staticmethod
    def gain_of_weight(weight: torch.Tensor, convention: str) -> torch.Tensor:
        return 1 + weight.float() if convention == "1+w" else weight.float()

    def test_real_gains(self) -> None:
        directory = Path(os.environ["GAMMA_MPOA_REAL_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        gains_section = (
            "real post-norm gains (all layers; range-read from HF safetensors)"
        )
        report(
            gains_section,
            f"{'model':26s} {'norm':26s} conv {'min':>8s} {'p1':>7s} {'median':>7s} {'p99':>7s} "
            f"{'max':>8s} {'CV|g|':>6s} {'neg%':>5s} {'|g|<0.1mean%':>12s} {'g==0%':>6s}",
        )
        off_section = (
            "off-target of the residual-space edit per unit of v-readout removed, real gains of "
            "every layer in the edited range (0.6-1.0 x depth), 4 random unit v per layer"
        )
        report(
            off_section,
            "GammaMPOA: ||Gamma a - v||. MPOA-equivalent: plain o-space projection, residual change "
            "-(v^T o) Gamma v, i.e. ||Gamma v / (v^T Gamma v) - v||. median / max (layer of max).",
        )
        off_variants = [
            ("ridge 0.1", "ridge", 0.1),
            ("ridge 0.05", "ridge", 0.05),
            ("clamp 0.01", "clamp", 0.01),
        ]
        for name, (_, prefix, convention) in REAL_MODELS.items():
            tensors = fetch_real_tensors(directory, name)["tensors"]
            for norm in ("post_attention_layernorm", "post_feedforward_layernorm"):
                per_layer = {
                    int(n.split("layers.")[1].split(".")[0]): self.gain_of_weight(
                        t, convention
                    )
                    for n, t in tensors.items()
                    if n.endswith(norm + ".weight")
                }
                g = torch.stack([per_layer[i] for i in sorted(per_layer)])
                abs_g = g.abs()
                q = torch.quantile(g.flatten(), torch.tensor([0.01, 0.5, 0.99]))
                report(
                    gains_section,
                    f"{name:26s} {norm:26s} {convention:4s} {g.min().item():8.3f} {q[0].item():7.3f} "
                    f"{q[1].item():7.3f} {q[2].item():7.3f} {g.max().item():8.2f} "
                    f"{(abs_g.std(1) / abs_g.mean(1)).mean().item():6.3f} "
                    f"{(g < 0).float().mean().item() * 100:5.2f} "
                    f"{(abs_g < 0.1 * abs_g.mean(1, keepdim=True)).float().mean().item() * 100:12.2f} "
                    f"{(g == 0).float().mean().item() * 100:6.2f}",
                )

                last = max(per_layer)
                edited = [i for i in sorted(per_layer) if i >= 0.6 * last]
                values: dict[str, list[tuple[float, int]]] = {}
                for i in edited:
                    gi = per_layer[i].double()
                    generator = torch.Generator().manual_seed(100 * i)
                    for _ in range(4):
                        v = F.normalize(
                            torch.randn(
                                gi.shape[0], generator=generator, dtype=torch.float64
                            ),
                            dim=0,
                        )
                        for label, inverse, eps in off_variants:
                            modifier = make_modifier(
                                GammaMPOA, gamma_inverse=inverse, gamma_eps=eps
                            )
                            pair = modifier.oblique_pair(v, gi)
                            assert pair is not None
                            values.setdefault(label, []).append(
                                (modifier.off_target_of(pair[0], gi, v), i)
                            )
                        values.setdefault("MPOA-equivalent", []).append(
                            (
                                LA.vector_norm(gi * v / (v @ (gi * v)) - v).item(),
                                i,
                            )
                        )
                cells = []
                for label, entries in values.items():
                    ordered = sorted(x for x, _ in entries)
                    worst = max(entries)
                    cells.append(
                        f"{label} {ordered[len(ordered) // 2]:.3f} / {worst[0]:.3f} (L{worst[1]})"
                    )
                report(
                    off_section,
                    f"{name:26s} {norm:26s} layers {edited[0]}-{edited[-1]}: "
                    + "; ".join(cells),
                )

    def test_real_edited_writers(self) -> None:
        directory = Path(os.environ["GAMMA_MPOA_REAL_DIR"])
        section = (
            "real writers of the edited layers (0.6, 0.8, 1.0 x depth) + their real gains, weight=1, "
            "deployed rank-3 LoRA, 2 random unit v per writer (means)"
        )
        report(
            section,
            "surv: ||b^T W'|| / ||b^T W|| for b = Gamma v and b = v. off: ||Gamma a - v||. dW: ||W'-W||/||W||. "
            "rn: |row norm'/row norm - 1| (max, p99, mean). Functional, 256 Gaussian inputs z, y = Gamma Wz / rms(Wz): "
            "y-leak = |v.y'|/|y'|; with P = I - v v^T (the ideal edit leaves P y unchanged): "
            "chg = |P y' - P y| / |P y| (total change of the write orthogonal to v), "
            "coll = sin angle(P y', P y) (its direction part), dnorm = |P y'|/|P y| - 1 (its norm part).",
        )
        report(
            section,
            f"{'model':14s} {'L':>2s} {'writer':9s} {'edit':21s} {'surv Gv':>8s} {'surv v':>7s} "
            f"{'off':>6s} {'dW':>6s} {'rn max':>7s} {'p99':>7s} {'mean':>7s} {'y-leak':>7s} {'chg':>7s} {'coll':>7s} {'dnorm':>8s}",
        )
        summary: dict[tuple[str, str, str], list[list[float]]] = {}
        for name, (_, prefix, convention) in REAL_MODELS.items():
            gains = fetch_real_tensors(directory, name)["tensors"]
            writers = fetch_edited_writers(directory, name)
            for layer_index in writers["layers"]:
                for writer, component, norm in (
                    ("o_proj", "self_attn.o_proj", "post_attention_layernorm"),
                    ("down_proj", "mlp.down_proj", "post_feedforward_layernorm"),
                ):
                    W = writers["tensors"][
                        prefix.format(i=layer_index) + component + ".weight"
                    ].float()
                    g = self.gain_of_weight(
                        gains[prefix.format(i=layer_index) + norm + ".weight"],
                        convention,
                    )
                    R = LA.vector_norm(W, dim=1)
                    generator = torch.Generator().manual_seed(layer_index)
                    Z = torch.randn(W.shape[1], 256, generator=generator)

                    def write(W_: torch.Tensor) -> torch.Tensor:
                        O = W_ @ Z
                        return (
                            g.view(-1, 1) * O / O.pow(2).mean(0, keepdim=True).sqrt()
                        ).T

                    Y = write(W)
                    rows: dict[str, list[list[float]]] = {}
                    for _ in range(2):
                        v = F.normalize(
                            torch.randn(W.shape[0], generator=generator), dim=0
                        )
                        b = g * v
                        Y_perp = Y - (Y @ v).view(-1, 1) * v
                        for label, settings in REAL_VARIANTS:
                            off = float("nan")
                            if settings is None:
                                A, B = self.mpoa_full_lora(W, v, 1.0, 3)
                            else:
                                modifier = make_modifier(GammaMPOA, **settings)
                                unit = settings.get("gamma_mode") == "unit"
                                g_edit = torch.ones_like(g) if unit else g
                                A, B = modifier.gamma_exact_lora(W, v, g_edit, 1.0, 3)
                                if not unit:
                                    a, _ = modifier.oblique_pair(v, g)
                                    off = modifier.off_target_of(a, g, v)
                            delta = B @ A
                            W_new = W + delta
                            err = ((LA.vector_norm(W_new, dim=1) - R) / R).abs()
                            Y_new = write(W_new)
                            Y_new_perp = Y_new - (Y_new @ v).view(-1, 1) * v
                            cos = F.cosine_similarity(Y_new_perp, Y_perp, dim=1).clamp(
                                -1, 1
                            )
                            rows.setdefault(label, []).append(
                                [
                                    (
                                        LA.vector_norm(b @ W_new)
                                        / LA.vector_norm(b @ W)
                                    ).item(),
                                    (
                                        LA.vector_norm(v @ W_new)
                                        / LA.vector_norm(v @ W)
                                    ).item(),
                                    off,
                                    (LA.matrix_norm(delta) / LA.matrix_norm(W)).item(),
                                    err.max().item(),
                                    err.quantile(0.99).item(),
                                    err.mean().item(),
                                    ((Y_new @ v).abs() / LA.vector_norm(Y_new, dim=1))
                                    .mean()
                                    .item(),
                                    torch.sqrt(1 - cos**2).mean().item(),
                                    (
                                        LA.vector_norm(Y_new_perp - Y_perp, dim=1)
                                        / LA.vector_norm(Y_perp, dim=1)
                                    )
                                    .mean()
                                    .item(),
                                    (
                                        LA.vector_norm(Y_new_perp, dim=1)
                                        / LA.vector_norm(Y_perp, dim=1)
                                        - 1
                                    )
                                    .mean()
                                    .item(),
                                ]
                            )
                    for label, values in rows.items():
                        m = torch.tensor(values).mean(0).tolist()
                        summary.setdefault((name, writer, label), []).append(
                            m + [layer_index]
                        )
                        report(
                            section,
                            f"{name[:14]:14s} {layer_index:2d} {writer:9s} {label:21s} {m[0]:8.1e} {m[1]:7.1e} "
                            f"{m[2]:6.3f} {m[3]:6.3f} {m[4]:7.1e} {m[5]:7.1e} {m[6]:7.1e} {m[7]:7.1e} {m[9]:7.1e} {m[8]:7.1e} {m[10]:+8.1e}",
                        )
                        settings = dict(REAL_VARIANTS)[label]
                        if settings is not None:
                            if settings.get("gamma_mode") == "unit":
                                self.assertLess(m[1], 1e-5)
                            else:
                                self.assertLess(m[0], 1e-5)

        worst_section = "real edited-layer writers: worst over the 3 layers (max row-norm error, off-target, write change, edit size)"
        report(
            worst_section,
            f"{'model':26s} {'writer':9s} {'edit':21s} {'rn max (layer)':>16s} {'off max':>8s} {'chg max':>8s} {'dW max':>7s} {'surv Gv max':>12s}",
        )
        for (name, writer, label), entries in summary.items():
            worst_rn = max(entries, key=lambda e: e[4])
            report(
                worst_section,
                f"{name:26s} {writer:9s} {label:21s} {worst_rn[4]:9.1e} (L{int(worst_rn[-1]):2d}) "
                f"{max(e[2] for e in entries):8.3f} {max(e[9] for e in entries):8.1e} "
                f"{max(e[3] for e in entries):7.3f} {max(e[0] for e in entries):12.1e}",
            )


class AnalysisTests(unittest.TestCase):
    """Claims in the module docstring about MPOA's magnitude preservation."""

    @staticmethod
    def mpoa_full(W, v, weight=1.0):
        R = LA.vector_norm(W, dim=1, keepdim=True)
        Wn = F.normalize(W, dim=1)
        return F.normalize(Wn - weight * torch.outer(v, v @ Wn), dim=1) * R

    def test_conjugation_invariance_and_variants(self) -> None:
        generator = torch.Generator().manual_seed(0)
        d_out, d_in = 512, 1024
        W = (
            torch.randn(d_out, d_in, generator=generator, dtype=torch.float64)
            / d_in**0.5
        )
        W = W * torch.exp(
            0.2 * torch.randn(d_out, 1, generator=generator, dtype=torch.float64)
        )
        v = F.normalize(
            torch.randn(d_out, generator=generator, dtype=torch.float64), dim=0
        )
        g = torch.exp(
            0.5 * torch.randn(d_out, generator=generator, dtype=torch.float64)
        )
        G = g.view(-1, 1)
        invariance = (
            (self.mpoa_full(G * W, v) / G - self.mpoa_full(W, v)).abs().max().item()
        )
        self.assertLess(invariance, 1e-12)
        section = "analysis: MPOA magnitude preservation vs oblique projection (float64, d=512x1024)"
        report(
            section,
            f"Gamma^-1 MPOA(Gamma W) - MPOA(W): max |diff| = {invariance:.1e} (conjugation-invariant)",
        )

        b = g * v
        a = (v / g) / ((g * v) @ (v / g))
        R = LA.vector_norm(W, dim=1, keepdim=True)
        Wn = F.normalize(W, dim=1)

        def frac(Wp, functional):
            return (
                LA.vector_norm(functional @ Wp) / LA.vector_norm(functional @ W)
            ).item()

        variants = {
            "MPOA (pre-norm write, b = v)": (self.mpoa_full(W, v), v),
            "MPOA (post-norm write, b = Gamma v)": (self.mpoa_full(W, v), b),
            "oblique in normalised space + renorm (substitution)": (
                F.normalize(Wn - torch.outer(a, b @ Wn), dim=1) * R,
                b,
            ),
            "oblique in raw space + renorm (no kernel confinement)": (
                F.normalize(W - torch.outer(a, b @ W), dim=1) * R,
                b,
            ),
        }
        for name, (Wp, functional) in variants.items():
            rn = ((LA.vector_norm(Wp, dim=1, keepdim=True) - R) / R).abs().max().item()
            report(
                section,
                f"{name:55s} surviving = {frac(Wp, functional):.3f}  row-norm err = {rn:.1e}",
            )
        for name, start in (
            (
                "substitution + kernel-confined restoration",
                F.normalize(Wn - torch.outer(a, b @ Wn), dim=1) * R,
            ),
            (
                "GammaMPOA: raw oblique + kernel-confined restoration",
                F.normalize(W - torch.outer(a, b @ W), dim=1) * R,
            ),
        ):
            final = start - torch.outer(a, b @ start)
            rn = (
                ((LA.vector_norm(final, dim=1, keepdim=True) - R) / R)
                .abs()
                .max()
                .item()
            )
            report(
                section,
                f"{name:55s} surviving = {frac(final, b):.1e}  row-norm err = {rn:.1e}",
            )

    def test_delta_reference(self) -> None:
        """
        Reference scale of delta = gamma_eps * scale(|gamma|) for the ridge: mean, median or
        RMS (shipped), on gain profiles with dead / dominant dimensions.
        """

        class Reference(GammaMPOA):
            def __init__(self, reference: str):
                super().__init__(
                    heretic_settings=cast(Any, SimpleNamespace(seed=1234)),
                    settings=GSettings(),
                )
                self.reference = reference

            def oblique_pair(self, v, g):
                scale = {
                    "mean": g.abs().mean(),
                    "median": g.abs().median(),
                    "rms": g.pow(2).mean().sqrt(),
                }[self.reference]
                delta = self.settings.gamma_eps * scale
                a = v * g / (g * g + delta * delta)
                ba = (g * v) @ a
                if (
                    not torch.isfinite(ba)
                    or ba <= 1e-12
                    or not torch.isfinite(a / ba).all()
                ):
                    return None
                return a / ba, g * v

        generator = torch.Generator().manual_seed(3)
        d = 64
        profiles = {}
        g = torch.empty(d).uniform_(0, 1e-3, generator=generator)
        g[5] = 51.0
        profiles["one gain 51, rest in [0, 1e-3]"] = g
        g = torch.randn(d, generator=generator) * 0.8 + 1
        g[: d // 2] = 0.0
        profiles["half the gains exactly 0"] = g
        profiles["log-normal (sigma 0.5)"] = torch.exp(
            0.5 * torch.randn(d, generator=generator)
        )
        section = "analysis: reference scale of delta (ridge, gamma_eps = 0.1), tiny writers 64x128, 3 random v"
        report(
            section,
            f"{'gain profile':32s} {'scale':6s} {'||dW||/||W|| max':>16s} {'row-norm err max':>16s} "
            f"{'off-target max':>14s} {'surviving b^T W max':>19s}",
        )
        for label, g in profiles.items():
            W = torch.randn(d, 128, generator=generator) / 128**0.5
            R = LA.vector_norm(W, dim=1)
            vs = [
                F.normalize(torch.randn(d, generator=generator), dim=0)
                for _ in range(3)
            ]
            module_rms = make_modifier(GammaMPOA)
            for reference in ("mean", "median", "rms"):
                modifier = Reference(reference)
                worst = [0.0, 0.0, 0.0, 0.0]
                for v in vs:
                    A, B = modifier.gamma_exact_lora(W, v, g, 1.0, 3)
                    if reference == "rms":
                        # The shipped modifier uses the RMS reference.
                        A2, B2 = module_rms.gamma_exact_lora(W, v, g, 1.0, 3)
                        self.assertTrue(torch.equal(A, A2) and torch.equal(B, B2))
                    W_new = W + B @ A
                    pair = modifier.oblique_pair(v, g)
                    b = g * v
                    worst = [
                        max(
                            worst[0], (LA.matrix_norm(B @ A) / LA.matrix_norm(W)).item()
                        ),
                        max(
                            worst[1],
                            ((LA.vector_norm(W_new, dim=1) - R) / R).abs().max().item(),
                        ),
                        max(
                            worst[2],
                            float("nan")
                            if pair is None
                            else LA.vector_norm(g * pair[0] - v).item(),
                        ),
                        max(
                            worst[3],
                            (LA.vector_norm(b @ W_new) / LA.vector_norm(b @ W)).item(),
                        ),
                    ]
                report(
                    section,
                    f"{label:32s} {reference:6s} {worst[0]:16.3f} {worst[1]:16.2e} {worst[2]:14.3f} {worst[3]:19.1e}",
                )
                if reference == "rms":
                    self.assertLess(worst[0], 0.5)
                    self.assertLess(worst[3], 1e-5)

    def test_real_qwen_prenorm_leak(self) -> None:
        """MPOA's own (row-norm) leak on real pre-norm weights, if available locally."""
        snapshots = glob.glob(
            "/data/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/"
        )
        direction_file = PAPER / "directions" / "qwen2.5-7b__jspace.pt"
        if not snapshots or not direction_file.exists():
            self.skipTest("Qwen2.5-7B-Instruct weights not available")
        from safetensors import safe_open

        snap = snapshots[0]
        index = json.loads(Path(snap, "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        directions = torch.load(direction_file, map_location="cpu", weights_only=False)[
            "directions"
        ].double()
        generator = torch.Generator().manual_seed(0)
        section = "analysis: MPOA leak on real pre-norm weights (Qwen2.5-7B-Instruct, weight=1)"
        report(
            section,
            "layer  writer            row-norm CV  surviving v^T W (J-space v)  (random v)",
        )
        for layer in (8, 14, 18, 22):
            for comp in ("self_attn.o_proj", "mlp.down_proj"):
                name = f"model.layers.{layer}.{comp}.weight"
                with safe_open(snap + index[name], "pt") as f:
                    W = f.get_tensor(name).double()
                R = LA.vector_norm(W, dim=1)
                v = F.normalize(directions[layer + 1], dim=0)
                vr = F.normalize(
                    torch.randn(W.shape[0], generator=generator, dtype=torch.float64),
                    dim=0,
                )

                def leak(u):
                    return (
                        LA.vector_norm(u @ self.mpoa_full(W, u)) / LA.vector_norm(u @ W)
                    ).item()

                report(
                    section,
                    f"{layer:5d}  {comp:16s}  {(R.std() / R.mean()).item():11.3f}  {leak(v):27.3f}  {leak(vr):10.3f}",
                )


def main() -> None:
    report_path = None
    if "--report" in sys.argv:
        i = sys.argv.index("--report")
        report_path = sys.argv[i + 1]
        del sys.argv[i : i + 2]

    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    # The modifier's own console output (init summaries) is not part of the report.
    with contextlib.redirect_stdout(io.StringIO()):
        result = runner.run(suite)

    import subprocess

    import heretic.modifiers.gamma_mpoa as module

    repo = Path(cast(str, module.__file__)).parents[3]
    head = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%H %s"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    header = (
        f"GammaMPOA tests: {repo} @ {head}{' (+uncommitted changes)' if dirty else ''}\n"
        f"torch {torch.__version__}, transformers {T.__version__}, CPU only, "
        f"real-weight tests {'ON' if os.environ.get('GAMMA_MPOA_REAL_DIR') else 'OFF'}\n"
    )
    out = [header, stream.getvalue()]
    for section, lines in REPORT.items():
        out.append(f"\n== {section}")
        out.extend(lines)
    text = "\n".join(out) + "\n"
    print(text)
    if report_path:
        Path(report_path).write_text(text)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
