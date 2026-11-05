# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026  Richard Young + contributors

"""
CPU-only tests for the GammaMPOA modifier (heretic.modifiers.gamma_mpoa).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src python tests/test_gamma_mpoa.py [--report PATH]

Tiny random-weight models are built from transformers configs (hidden size 64, 3 layers).
Post-norm gains are randomised (log-normal, ~5% negative) so that gamma is non-uniform.

(a) pre-norm models: GammaMPOA LoRA == MPOA LoRA, max abs diff 0
(b) post-norm models: |v^T y| / ||y|| of the residual writes (forward hooks on the post-norms)
    for unedited / MPOA / GammaMPOA at weight 1
(c) merged weights equal W' computed directly (independent float64 reference)
(d) the plugin loads through Heretic's plugin loader with configs/core_gmpoa.toml and runs a trial
plus checks of the gain conventions and of the analysis claims in the module docstring.
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


def gain_of(model: Model, layer_index: int, component: str) -> torch.Tensor | None:
    layer = model.get_layers()[layer_index]
    chain = post_norm_chain(layer, component)
    if chain is None:
        return None
    gain = None
    for name in chain:
        norm_gain = rmsnorm_gain(getattr(layer, name)).cpu()
        gain = norm_gain if gain is None else gain * norm_gain
    return cast(torch.Tensor, gain).double()


def weight_leak(model: Model, directions: torch.Tensor) -> float:
    """
    Mean over writers of ||b^T W_eff|| / ||b^T W|| with b = Gamma v (fraction of the
    v-component of the residual write that survives the edit; input-independent).
    """
    fractions = []
    for layer_index, component, module in writer_modules(model):
        v = directions[layer_index + 1].double()
        g = gain_of(model, layer_index, component)
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
    For Gemma-4 MoE layers the dense-MLP write is measured at post_feedforward_layernorm_1
    with v replaced by Gamma_0 v (Gamma_0 = post_feedforward_layernorm gain).
    """
    hooks = []
    captured: dict[tuple[int, str], torch.Tensor] = {}
    probes: dict[tuple[int, str], torch.Tensor] = {}
    layers = model.get_layers()
    for layer_index, component, module in writer_modules(model):
        layer = layers[layer_index]
        chain = post_norm_chain(layer, component)
        v = directions[layer_index + 1].float()
        if chain is None:
            target = module
            probe = v
        elif len(chain) == 1:
            target = getattr(layer, chain[0])
            probe = v
        else:
            target = getattr(layer, chain[0])
            outer = torch.ones_like(v)
            for name in chain[1:]:
                outer = outer * rmsnorm_gain(getattr(layer, name)).cpu()
            probe = outer * v
        key = (layer_index, component)
        probes[key] = probe

        def hook(_m, _i, output, key=key):
            captured[key] = (
                output[0] if isinstance(output, tuple) else output
            ).detach()

        hooks.append(target.register_forward_hook(hook))
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
    out["all"] = sum(ratios.values()) / len(ratios)
    return out


def reference_gamma_edit(W, v, g, weight, eps, full, inverse="ridge"):
    """Independent float64 reference of the dense gamma-exact edit (no LoRA truncation)."""
    W, v, g = W.double(), v.double(), g.double()
    b = g * v
    delta = eps * g.abs().mean()
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
        ctx = prepare(model, [mpoa, gmpoa], directions)
        parameters = full_strength(model, 1.0)

        results = {}
        model._zero_lora_weights()
        results["unedited"] = (
            write_leak(model, directions),
            weight_leak(model, directions),
        )
        apply(mpoa, ctx, parameters)
        results["MPOA"] = (
            write_leak(model, directions),
            weight_leak(model, directions),
        )
        apply(gmpoa, ctx, parameters)
        results["GammaMPOA"] = (
            write_leak(model, directions),
            weight_leak(model, directions),
        )
        return results

    def test_leak(self) -> None:
        header = (
            f"{'model':10s} {'gamma':7s} {'edit':10s} {'|v.y|/|y| attn':>15s} "
            f"{'mlp':>9s} {'all':>9s} {'surviving b^T W':>16s}"
        )
        report(
            "(b) post-norm residual-write leak, weight=1, row_normalization=full",
            header,
        )
        for arch in POST_NORM + PRE_NORM:
            for gamma in (
                ["random", "uniform"] if arch in ("gemma2", "olmo2") else ["random"]
            ):
                results = self.run_arch(arch, gamma=gamma)
                for edit, (w, frac) in results.items():
                    report(
                        "(b) post-norm residual-write leak, weight=1, row_normalization=full",
                        f"{arch:10s} {gamma if arch in POST_NORM else 'n/a':7s} {edit:10s} "
                        f"{w['attn.o_proj']:15.2e} {w['mlp.down_proj']:9.2e} {w['all']:9.2e} {frac:16.2e}",
                    )
                if arch in POST_NORM:
                    self.assertLess(results["GammaMPOA"][0]["all"], 1e-5, arch)
                    self.assertLess(results["GammaMPOA"][1], 1e-5, arch)
                    if gamma == "random":
                        self.assertGreater(results["MPOA"][0]["all"], 1e-2, arch)
                        self.assertGreater(results["MPOA"][1], 0.1, arch)

    def test_leak_none_mode(self) -> None:
        for arch in ("gemma2", "olmo3"):
            results = self.run_arch(arch, row_norm="none")
            for edit, (w, frac) in results.items():
                report(
                    "(b') row_normalization=none (projected abliteration vs oblique)",
                    f"{arch:8s} {edit:10s} |v.y|/|y| = {w['all']:.2e}   surviving b^T W = {frac:.2e}",
                )
            self.assertLess(results["GammaMPOA"][1], 1e-6)
            self.assertGreater(results["MPOA"][1], 0.1)

    def test_exact_prenorm_control(self) -> None:
        for arch in PRE_NORM:
            hf = build_hf(arch, seed=4)
            model = wrap(hf)
            directions = random_directions(3, 64, seed=5)
            mpoa = make_modifier(Abliteration)
            exact = make_modifier(GammaMPOA, exact_prenorm=True)
            ctx = prepare(model, [mpoa, exact], directions)
            parameters = full_strength(model)
            apply(mpoa, ctx, parameters)
            leak_mpoa = weight_leak(model, directions)
            apply(exact, ctx, parameters)
            leak_exact = weight_leak(model, directions)
            report(
                "exact_prenorm=true control (pre-norm models, weight=1)",
                f"{arch:8s} surviving v^T W: MPOA {leak_mpoa:.2e}   exact {leak_exact:.2e}",
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

    def test_load_and_run(self) -> None:
        import tempfile

        import heretic.plugin as plugin_module
        from heretic.config import InitOnlySettings
        from heretic.modifier import load_and_init_modifiers
        from heretic.utils import Prompt

        tokenizer_dirs = glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/"
            )
        )
        if not tokenizer_dirs:
            self.skipTest("no local tokenizer")
        tokenizer = T.AutoTokenizer.from_pretrained(tokenizer_dirs[0])

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

            toml = tomllib.loads(GMPOA_TOML.read_text())
            settings = InitOnlySettings.model_validate(
                toml
                | {
                    "model": tmp,
                    "dtypes": ["float32"],
                    "device_map": "cpu",
                    "batch_size": 4,
                    "seed": 1234,
                }
            )

            good = [
                Prompt(
                    system="You are a helpful assistant.",
                    user=f"Write a poem about {w}.",
                )
                for w in ("cats", "rain", "tea", "maps")
            ]
            bad = [
                Prompt(
                    system="You are a helpful assistant.",
                    user=f"Explain how to steal {w}.",
                )
                for w in ("cars", "data", "money", "art")
            ]
            original = plugin_module.load_prompts
            # No dataset download: distinct synthetic good/bad prompts.
            plugin_module.load_prompts = lambda _settings, spec: (
                bad if "harmful" in str(spec) else good
            )
            try:
                stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    model = Model(settings)
                    entries = load_and_init_modifiers(settings, model)
                finally:
                    sys.stdout = stdout
            finally:
                plugin_module.load_prompts = original

            self.assertEqual(len(entries), 1)
            modifier = entries[0].modifier
            self.assertIsInstance(modifier, GammaMPOA)
            self.assertFalse(modifier.reproducible)
            self.assertEqual(modifier.settings.row_normalization, RowNormalization.FULL)
            self.assertEqual(model.peft_config.r, 3)
            self.assertTrue(modifier.use_gamma_path)

            ctx = Context(settings=settings, model=model)
            study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
            parameters = modifier.suggest_parameters(ctx, study.ask())
            modifier.reset_model(ctx)
            modifier.modify_model(ctx, parameters)

            worst = 0.0
            n_edited = 0
            for layer_index, component, module in writer_modules(model):
                A = module.lora_A["default"].weight
                if A.abs().max() == 0:
                    continue
                n_edited += 1
                distribution = parameters.weight_distributions[component]
                distance = abs(layer_index - distribution.max_weight_position)
                weight = distribution.max_weight + (
                    distance / distribution.min_weight_distance
                ) * (distribution.min_weight - distribution.max_weight)
                if parameters.direction_index is None:
                    v = modifier.residual_directions[layer_index + 1]
                else:
                    frac, index = math.modf(parameters.direction_index + 1)
                    v = F.normalize(
                        modifier.residual_directions[int(index)].lerp(
                            modifier.residual_directions[int(index) + 1], frac
                        ),
                        dim=0,
                    )
                b = (
                    cast(torch.Tensor, gain_of(model, layer_index, component))
                    * v.double()
                )
                W = module.base_layer.weight.detach().double()
                residual = b @ effective_weight(module) - (1 - weight) * (b @ W)
                worst = max(
                    worst, (LA.vector_norm(residual) / LA.vector_norm(b @ W)).item()
                )
            self.assertGreater(n_edited, 0)
            self.assertLess(worst, 1e-5)

            report(
                "(d) plugin loader",
                f"loaded {type(modifier).__module__}.{type(modifier).__name__} via load_and_init_modifiers "
                f"('{entries[0].name}'), reproducible={modifier.reproducible}, "
                f"row_normalization={modifier.settings.row_normalization.value}, lora r={model.peft_config.r}",
            )
            report(
                "(d) plugin loader",
                f"real Model + init() on tiny Gemma-2 (directions from get_residuals_mean), one random trial: "
                f"{n_edited} writers edited, max |b^T W' - (1-weight) b^T W| / |b^T W| = {worst:.1e}",
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


@unittest.skipUnless(
    os.environ.get("GAMMA_MPOA_REAL_DIR"),
    "set GAMMA_MPOA_REAL_DIR=<dir> to range-download real post-norm weights (~0.8 GB)",
)
class RealWeightTests(unittest.TestCase):
    """Leak and norm preservation with real post-norm gains and real mid-layer writers."""

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

    def test_real_gains_and_writers(self) -> None:
        directory = Path(os.environ["GAMMA_MPOA_REAL_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        gains_section = (
            "real post-norm gains (all layers; range-read from HF safetensors)"
        )
        report(
            gains_section,
            f"{'model':26s} {'norm':26s} conv {'min':>8s} {'p1':>7s} {'median':>7s} {'p99':>7s} "
            f"{'max':>8s} {'CV|g|':>6s} {'neg%':>5s} {'|g|<0.1mean%':>12s}",
        )
        writer_section = "real mid-layer writers + real gains, 4 random unit v, weight=1, deployed rank-3 LoRA"
        report(
            writer_section,
            f"{'model':26s} {'writer':9s} {'edit':22s} {'surviving b^T W':>15s} "
            f"{'row-norm err max':>16s} {'p99':>8s} {'mean':>8s} {'cos(Ga,v)':>9s}",
        )
        for name, (_, prefix, convention) in REAL_MODELS.items():
            data = fetch_real_tensors(directory, name)
            tensors = data["tensors"]

            def gain(weight: torch.Tensor) -> torch.Tensor:
                return 1 + weight.float() if convention == "1+w" else weight.float()

            for norm in ("post_attention_layernorm", "post_feedforward_layernorm"):
                g = torch.stack(
                    [
                        gain(t)
                        for n, t in tensors.items()
                        if n.endswith(norm + ".weight")
                    ]
                )
                abs_g = g.abs()
                q = torch.quantile(g.flatten(), torch.tensor([0.01, 0.5, 0.99]))
                report(
                    gains_section,
                    f"{name:26s} {norm:26s} {convention:4s} {g.min().item():8.3f} {q[0].item():7.3f} "
                    f"{q[1].item():7.3f} {q[2].item():7.3f} {g.max().item():8.2f} "
                    f"{(abs_g.std(1) / abs_g.mean(1)).mean().item():6.3f} "
                    f"{(g < 0).float().mean().item() * 100:5.2f} "
                    f"{(abs_g < 0.1 * abs_g.mean(1, keepdim=True)).float().mean().item() * 100:12.2f}",
                )

            mid = data["mid"]
            for writer, norm in (
                ("o_proj", "post_attention_layernorm"),
                ("down_proj", "post_feedforward_layernorm"),
            ):
                component = (
                    "self_attn.o_proj" if writer == "o_proj" else "mlp.down_proj"
                )
                W = tensors[prefix.format(i=mid) + component + ".weight"].float()
                g = gain(tensors[prefix.format(i=mid) + norm + ".weight"])
                R = LA.vector_norm(W, dim=1)
                generator = torch.Generator().manual_seed(0)
                directions = [
                    F.normalize(torch.randn(W.shape[0], generator=generator), dim=0)
                    for _ in range(4)
                ]
                edits = [("MPOA", None)] + [
                    (f"GammaMPOA {inverse} eps={eps:g}", (inverse, eps))
                    for inverse, eps in (
                        ("ridge", 0.1),
                        ("ridge", 0.05),
                        ("clamp", 0.01),
                    )
                ]
                for label, variant in edits:
                    rows = []
                    for v in directions:
                        b = g * v
                        if variant is None:
                            A, B = self.mpoa_full_lora(W, v, 1.0, 3)
                            cos = float("nan")
                        else:
                            modifier = make_modifier(
                                GammaMPOA,
                                gamma_inverse=variant[0],
                                gamma_eps=variant[1],
                            )
                            A, B = modifier.gamma_exact_lora(W, v, g, 1.0, 3)
                            a, _ = modifier.oblique_pair(v, g)
                            cos = (1 / LA.vector_norm(g * a)).item()
                        W_new = W + B @ A
                        err = ((LA.vector_norm(W_new, dim=1) - R) / R).abs()
                        rows.append(
                            [
                                (
                                    LA.vector_norm(b @ W_new) / LA.vector_norm(b @ W)
                                ).item(),
                                err.max().item(),
                                err.quantile(0.99).item(),
                                err.mean().item(),
                                cos,
                            ]
                        )
                    m = torch.tensor(rows).mean(0).tolist()
                    report(
                        writer_section,
                        f"{name:26s} {writer:9s} {label:22s} {m[0]:15.2e} {m[1]:16.1e} {m[2]:8.1e} {m[3]:8.1e} {m[4]:9.3f}",
                    )
                    if variant is not None:
                        self.assertLess(m[0], 1e-5)


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
