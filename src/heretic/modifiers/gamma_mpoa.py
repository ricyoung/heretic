# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026  Richard Young + contributors

"""Gamma-exact MPOA for post-norm ("sandwich-norm") architectures.

Problem
-------
In Gemma-2/3/4 and OLMo-2/3 the output of every writer o = W z (W = attn o_proj or
mlp down_proj, i.e. exactly the matrices MPOA edits) passes through an RMSNorm with a
per-dimension gain before it is added to the residual stream:

    y = Gamma * o / rms(o)        (Gamma = diag(gamma), rms(o) is a per-token scalar)

so the residual write along the refusal direction v is v^T y = (Gamma v)^T o / rms(o).
Orthogonalising W against v (what Abliteration/MPOA do) zeroes v^T o, not
(Gamma v)^T o, so the write keeps a component along v whenever gamma is non-uniform.

Edit
----
For writers followed by such a post-norm, the output-side projection is replaced by the
oblique ("gamma-exact") projection

    W' = W - lambda * a b^T W,    b = Gamma v,   b^T a = 1

so that b^T W' = (1 - lambda) b^T W, i.e. v^T y' = 0 at lambda = 1 (up to the per-token rms
scalar, which cannot create a v-component). Exactness only needs b = Gamma v (the true gains)
and b^T a = 1; the o-space removal direction a is free. In residual coordinates the edit is

    Gamma o' = (I - lambda (Gamma a) v^T) Gamma o,     Gamma a = v + w,  w orthogonal to v,

an *oblique* projection of the residual write. Only for a = Gamma^-1 v (exact inverse, possible
only if no gain is zero) is Gamma a = v and the edit the orthogonal projection of the write.
Otherwise the change of the write, -lambda (v^T Gamma o)(v + w), also has an off-target part
-lambda (v^T Gamma o) w: per unit of v-readout removed, ||w|| = ||Gamma a - v|| is written along
w (concretely: the v-part on low-gain dimensions is left in place and an equal v-readout is
removed on high-gain dimensions instead). Gains that are exactly 0 make some off-target
unavoidable. For comparison, plain MPOA changes the write by about -lambda (v^T o) Gamma v, an
off-target of ||Gamma v / (v^T Gamma v) - v|| per unit removed, while leaving 6-81% of the
v-readout in place. init() logs ||Gamma a - v|| per writer (self.off_target) and
modify_model() records it per trial (self.edit_stats).

Choice of a
-----------
Real post-norm gains have many near-zero entries (Gemma-4-31B post_attention_layernorm: 25% of
gains below 10% of the mean; Gemma-2-27B: 0.5% exactly 0), where the exact Gamma^-1 explodes:
the edit then spends large o-space changes on dimensions the writer can barely write to, which
inflates rms(o) and defeats the low-rank row-norm restoration. Gamma^-1 is therefore regularised
with delta = gamma_eps * rms(gamma) of the writer's gain:

    "ridge" (default):  a ~ (Gamma^2 + delta^2)^-1 Gamma v = argmin ||Gamma a - v||^2 + delta^2 ||a||^2
    "clamp":            a ~ v / (sign(gamma) * max(|gamma|, delta))   (the plan's |gamma| >= eps clamp)

Both are normalised to b^T a = 1, so neither affects exactness; they trade the off-target
||Gamma a - v|| against the size of the o-space edit (row-norm preservation, rms(o) change).
The RMS reference (instead of mean |gamma| or the median) keeps delta from sitting far below a
few dominant gains, which made the ridge put its removal mass on near-dead dimensions (one gain
51, the rest <= 1e-3: ||dW||/||W|| 6.4 with mean |gamma|, 0.2 with the RMS; with the median,
9000, and no edit at all when half the gains are 0). The ridge edit is bounded,
||a b^T W|| <= sqrt(1 + (max|gamma| / delta)^2) ||W||_2; the clamp's is not.

Measured on the real writers and gains of the edited layers (0.6, 0.8, 1.0 x depth; random v;
notes/gamma_mpoa_tests.txt), ridge with gamma_eps = 0.1 gives off-target medians 0.14
(Gemma-2-27B), 0.25 (Gemma-4-31B), 0.03-0.06 (OLMo-2/3), max 1.02 on Gemma-2's last
post-attention norm (exact-zero gains), max deployed row-norm errors <= 4% (Gemma) and 0.6%
(OLMo-3; OLMo-2's last o_proj: 40% on near-dead rows with ~0 gain, p99 1.8%), and a change of
the write orthogonal to v below MPOA's on 23 of 24 writers. The plan's clamp with
gamma_eps = 0.01 lowers the off-target (medians 0.01-0.17) but is unusable on these gains:
Gemma-2's last o_proj gets a max row-norm error of 370% (mean 48%) and a 6.6% shrink of its
write, Gemma-4 52% and 1.8%. The default (ridge, 0.1) therefore deviates from the plan's clamp;
it needs the PI's sign-off. gamma_inverse = "clamp" reproduces the plan's form exactly.

Edit-size guard: modify_model() warns once per writer when the deployed edit has
||dW||/||W|| > GUARD_RELATIVE_EDIT or a max row-norm error > GUARD_MAX_ROW_NORM_ERROR, and when
a writer is left unedited because v has no weight on its live dimensions.

Interaction with MPOA's magnitude preservation
----------------------------------------------
Heretic's MPOA ("full" row normalization, abliteration.py) normalises the *output rows* of W
(one row per residual dimension), projects the row-normalised matrix, renormalises the rows and
restores the original row norms R. Two consequences (both verified numerically in
tests/test_gamma_mpoa.py):

1. The edit is invariant under conjugation by any positive diagonal output scaling:
   Gamma^-1 MPOA(Gamma W) = MPOA(W). Applying MPOA "in residual coordinates" is therefore a
   no-op, and a post-norm writer is treated exactly like a pre-norm writer with row norms
   |gamma| * R.
2. Projecting the row-normalised matrix and then restoring row norms is an oblique projection
   in the original coordinates (removal direction ~ R v, killed functional ~ R^-1 v). MPOA thus
   leaves a v-component in the write even on pre-norm models, roughly the (v^2-weighted)
   coefficient of variation of the writer's row norms (6-20% of the v-readout survives on
   Qwen2.5-7B o_proj/down_proj at weight 1); on post-norm models the spread of |gamma| * R sets
   the leak (Gemma-2-27B 30-44%, Gemma-4-31B 57-63%, OLMo-2-32B 27-38%, OLMo-3.1-32B 8-9%).

Substituting the oblique pair into MPOA's normalised-space projection therefore cannot be exact.
GammaMPOA instead (for row_normalization = "full"):

  a) applies the (regularised) oblique projection to W in raw coordinates,
  b) restores the original row norms exactly as MPOA does (normalise rows, multiply by R),
  c) confines that row-norm restoration to the kernel of b^T (K = (I - a b^T) * delta), so it
     cannot reintroduce a v-component, and
  d) exports the same rank-r LoRA as MPOA: one exact rank-1 component (-lambda a, b^T W) plus a
     rank-(r-1) randomized-SVD approximation of K (same svd_lowrank call and reseeding as MPOA),
     whose left factor is re-projected onto ker(b^T).

The deployed (LoRA or merged) weight satisfies b^T W' = (1 - lambda) b^T W to floating-point
precision; row norms are preserved up to the rank-(r-1) truncation of K (MPOA: rank-r truncation
of its whole delta). With row_normalization = "none" the edit is the rank-1 oblique projection.
"pre" only changes the projection geometry (no magnitude restoration), and exactness fixes the
geometry, so for gamma writers "pre" is identical to "none".

Matched control (gamma_mode = "unit")
-------------------------------------
On a post-norm model GammaMPOA differs from MPOA by (i) the gain correction (with its off-target
w and its weaker row-norm preservation) and (ii) the removal of MPOA's own row-norm leak (kernel
confinement). gamma_mode = "unit" applies the identical exact, kernel-confined edit with
Gamma := 1 (a = b = v) to every writer, post-norm or not: GammaMPOA vs GammaMPOA(unit) isolates
(i), MPOA vs GammaMPOA(unit) isolates (ii). On pre-norm models "unit" is the "exact MPOA" edit.

Pre-norm writers
----------------
If the model has no post-norm writers (Qwen, Llama, Mistral, ...) and gamma_mode = "gain",
modify_model() delegates to Abliteration.modify_model(): the edit is bit-identical to MPOA.
Mixed stacks (some writers post-norm, some not) are rejected, and so are decoder layers that are
not in POST_NORM_CHAINS but have sandwich-norm modules (post_feedforward_layernorm,
post_mlp_layernorm, post_self_attn_layernorm, ...; e.g. GLM-4, EXAONE-4, Gemma-3n), instead of
silently treating them as pre-norm.

MoE layers
----------
Heretic edits only the modules returned by Model.get_layer_modules(). For Gemma-4 MoE layers that
is the dense MLP, whose write is corrected through post_feedforward_layernorm_1 and the outer
post_feedforward_layernorm; the expert path (post_feedforward_layernorm_2) is edited by neither
MPOA nor GammaMPOA and keeps its v-component, so the MLP block's total write is not v-free.
(The paper's Gemma-4-31B is dense.)

Use with:
    modifiers = [{ plugin = "heretic.modifiers.gamma_mpoa.GammaMPOA" }]
    [modifier.GammaMPOA]          # optional
    gamma_mode = "unit"           # matched Gamma := 1 control
"""

import math
from enum import Enum
from typing import cast

import bitsandbytes.functional as BNB_F
import torch
import torch.linalg as LA
import torch.nn.functional as F
from peft.tuners.lora.layer import Linear
from pydantic import Field, PositiveFloat
from torch import Tensor
from torch.nn import Module

from heretic.modifier import Context
from heretic.modifiers.abliteration import Abliteration, Parameters, RowNormalization
from heretic.modifiers.abliteration import Settings as AbliterationSettings
from heretic.utils import print

# Decoder layer class -> component -> post-norm attributes applied (in this order) between the
# writer's output and the residual add. Verified against transformers 5.18.0:
#   gemma2/modeling_gemma2.py:326,332  gemma3/modeling_gemma3.py:420,426
#   gemma4/modeling_gemma4.py:1407,1428 (+1415 for the MoE block)
#   olmo2/modeling_olmo2.py:315,321    olmo3/modeling_olmo3.py:259,265
_SANDWICH = {
    "attn.o_proj": ("post_attention_layernorm",),
    "mlp.down_proj": ("post_feedforward_layernorm",),
}
POST_NORM_CHAINS: dict[str, dict[str, tuple[str, ...]]] = {
    "Gemma2DecoderLayer": _SANDWICH,
    "Gemma3DecoderLayer": _SANDWICH,
    "Gemma4TextDecoderLayer": _SANDWICH,
    "Olmo2DecoderLayer": _SANDWICH,
    "Olmo3DecoderLayer": _SANDWICH,
}

# Direct children of a decoder layer whose name starts with "post_" and contains "norm" but that
# do not sit between a writer and the residual add. "post_attention_layernorm" is the pre-MLP
# norm in Llama-style layers, so it cannot identify a sandwich norm on its own.
_NOT_SANDWICH_NORMS = {
    "post_attention_layernorm",
    "post_layernorm",
    "post_norm",
    "post_layer_norm",
    "post_per_layer_input_norm",
    "post_projection_norm",
}

# Edit-size guard: warn (once per writer) when the deployed edit of a writer exceeds these.
GUARD_MAX_ROW_NORM_ERROR = 0.10
GUARD_RELATIVE_EDIT = 0.5

# oblique_pair(): b^T a (before normalisation) is the gain-weighted fraction of v's mass on
# dimensions the writer can write to, in [0, 1] and invariant to the scale of gamma. Below this
# the v-readout of the write is negligible and the edit is skipped.
MIN_LIVE_FRACTION = 1e-12


def post_norm_chain(layer: Module, component: str) -> tuple[str, ...] | None:
    chains = POST_NORM_CHAINS.get(type(layer).__name__)
    if chains is None:
        return None
    chain = chains.get(component)
    if chain is None:
        return None
    if (
        type(layer).__name__ == "Gemma4TextDecoderLayer"
        and component == "mlp.down_proj"
        and getattr(layer, "enable_moe_block", False)
    ):
        # Gemma-4 MoE layers: dense MLP -> post_feedforward_layernorm_1, summed with the
        # (normalised) expert output, then post_feedforward_layernorm (modeling_gemma4.py:1415,1428).
        # The outer rms is still a per-token scalar, so the gains compose. The experts
        # themselves are not returned by Model.get_layer_modules() and are not edited.
        chain = ("post_feedforward_layernorm_1", "post_feedforward_layernorm")
    return chain


def gain_scale(g: Tensor) -> Tensor:
    """
    Reference scale of a gain vector for the Gamma^-1 regularisation: its RMS, sqrt(mean gamma^2).
    Unlike mean |gamma| (or the median) it cannot sit far below a few dominant gains, which is
    what makes the ridge put its removal mass on near-dead dimensions (see EdgeCaseTests and
    AnalysisTests.test_delta_reference).
    """
    return g.pow(2).mean().sqrt()


def sandwich_norm_names(layer: Module) -> list[str]:
    """Names of direct children of `layer` that look like sandwich (post-writer) norms."""
    return [
        name
        for name, _ in layer.named_children()
        if name.startswith("post_")
        and "norm" in name
        and name not in _NOT_SANDWICH_NORMS
    ]


def rmsnorm_gain(norm: Module) -> Tensor:
    """
    The per-dimension gain an RMSNorm applies after dividing by rms(x), in float32,
    computed exactly as the module's forward computes it.
    """
    name = type(norm).__name__
    weight = cast(Tensor, getattr(norm, "weight", None))
    if name in ("Gemma2RMSNorm", "Gemma3RMSNorm"):
        # output * (1.0 + self.weight.float())  (modeling_gemma2.py:62, modeling_gemma3.py:149)
        gain = 1.0 + weight.float()
    elif name == "Gemma4RMSNorm":
        # normed_output * self.weight.float() if with_scale  (modeling_gemma4.py:213-214)
        if getattr(norm, "with_scale", True):
            gain = weight.float()
        else:
            raise ValueError("Gemma4RMSNorm without scale is not a post-norm with gain")
    elif name in ("Olmo2RMSNorm", "Olmo3RMSNorm"):
        # (self.weight * hidden_states)  (modeling_olmo2.py:65, modeling_olmo3.py:59)
        gain = weight.float()
    else:
        raise ValueError(f"Unsupported post-norm class {name}")

    # Self-check against the module's own forward pass, so that a convention change in a
    # future transformers version cannot silently produce the wrong gains. For x = 1024 * 1,
    # rms(x) = 2^10 exactly and eps is below float32 resolution, so forward(x) = gain.
    if weight.device.type != "meta":
        with torch.no_grad():
            x = torch.full(
                (1, gain.shape[0]), 1024.0, dtype=torch.float32, device=weight.device
            )
            probe = norm(x).float().view(-1)
        tolerance = 1e-4 * gain.abs().max().clamp_min(1e-6)
        if (probe - gain).abs().max() > tolerance:
            raise ValueError(
                f"{name}: gain convention does not match the module's forward pass"
            )

    return gain


class GammaInverse(str, Enum):
    RIDGE = "ridge"
    CLAMP = "clamp"


class GammaMode(str, Enum):
    GAIN = "gain"
    UNIT = "unit"


class Settings(AbliterationSettings):
    gamma_mode: GammaMode = Field(
        default=GammaMode.GAIN,
        description=(
            '"gain" (default): gamma-exact edit with the true post-norm gains; pre-norm models '
            'get the bit-identical MPOA edit. "unit": the same exact, kernel-confined edit with '
            "Gamma := 1 for every writer (matched control that isolates the gain correction; "
            'on pre-norm models the "exact MPOA" edit).'
        ),
    )

    gamma_inverse: GammaInverse = Field(
        default=GammaInverse.RIDGE,
        description=(
            "Regularisation of Gamma^-1 in the removal direction a. "
            '"ridge": a ~ Gamma v / (Gamma^2 + delta^2) (default); '
            '"clamp": a ~ v / (sign(gamma) * max(|gamma|, delta)).'
        ),
    )

    gamma_eps: PositiveFloat = Field(
        default=0.1,
        description=(
            "delta of the Gamma^-1 regularisation, as a fraction of the RMS of the writer's "
            "post-norm gain, sqrt(mean gamma^2). Does not affect exactness (v^T y' = 0 at weight 1); "
            "trades the off-target write ||Gamma a - v|| (smaller delta: smaller) against the "
            "size of the o-space edit and row-norm preservation (smaller delta: worse)."
        ),
    )


class GammaMPOA(Abliteration):
    settings: Settings

    @property
    def reproducible(self) -> bool:
        # Fork-only plugin: cannot be reproduced with upstream Heretic.
        return False

    @property
    def modifier_name(self) -> str:
        unit = (
            " (Gamma := 1 control)"
            if self.settings.gamma_mode == GammaMode.UNIT
            else ""
        )
        if (
            self.settings.orthogonalize_direction
            and self.settings.row_normalization == RowNormalization.FULL
        ):
            return f"Gamma-Exact Magnitude-Preserving Orthogonal Ablation (GammaMPOA){unit}"
        return f"Gamma-Exact {super().modifier_name}{unit}"

    def init(self, ctx: Context) -> None:
        # Same direction estimation, LoRA rank and adapter setup as MPOA.
        super().init(ctx)
        self.detect_post_norm_gains(ctx)

    def detect_post_norm_gains(self, ctx: Context) -> None:
        model = ctx.get_model()
        layers = model.get_layers()

        # (layer_index, component) -> float32 gain on CPU; None for writers without a post-norm.
        self.post_norm_gains: dict[tuple[int, str], Tensor | None] = {}
        # Per-writer diagnostics, filled here (off-target with the layer's own direction) and
        # updated by modify_model() (worst values over all trials so far).
        self.off_target: dict[tuple[int, str], float] = {}
        self.edit_stats: dict[tuple[int, str], dict[str, float]] = {}
        self._warned: set[tuple[int, str, str]] = set()
        unknown_sandwich: dict[str, list[str]] = {}

        for layer_index in range(len(layers)):
            layer = layers[layer_index]
            for component in model.get_layer_modules(layer_index):
                chain = post_norm_chain(layer, component)
                if chain is None:
                    names = sandwich_norm_names(layer)
                    if names:
                        unknown_sandwich[type(layer).__name__] = names
                    self.post_norm_gains[(layer_index, component)] = None
                    continue
                gain = None
                for norm_name in chain:
                    norm_gain = rmsnorm_gain(getattr(layer, norm_name)).cpu()
                    gain = norm_gain if gain is None else gain * norm_gain
                self.post_norm_gains[(layer_index, component)] = gain

        if unknown_sandwich:
            raise ValueError(
                "GammaMPOA: decoder layer class(es) "
                + ", ".join(
                    f"{c} ({', '.join(n)})" for c, n in unknown_sandwich.items()
                )
                + " have sandwich-norm modules but are not in POST_NORM_CHAINS. Add the "
                "writer -> post-norm chain (and the norm's gain convention to rmsnorm_gain) "
                "instead of treating the model as pre-norm."
            )

        n_post = sum(gain is not None for gain in self.post_norm_gains.values())
        n_total = len(self.post_norm_gains)

        print()
        if 0 < n_post < n_total:
            raise ValueError(
                f"GammaMPOA: mixed pre/post-norm writers ({n_post} of {n_total}) are not supported"
            )

        unit = self.settings.gamma_mode == GammaMode.UNIT
        self.use_gamma_path = n_post > 0 or unit

        if unit:
            print(
                f"gamma_mode = unit: exact, kernel-confined edit with Gamma := 1 for all "
                f"{n_total} writers ({n_post} of them post-norm)."
            )
            return

        if n_post == 0:
            print("No post-norm writers found: edit is identical to MPOA.")
            return

        if self.settings.row_normalization == RowNormalization.PRE:
            print(
                '[yellow]Note: row_normalization = "pre" is identical to "none" for '
                "gamma-exact writers (see module docstring).[/]"
            )

        print(
            f"Post-norm gains for [bold]{n_post}[/] writers "
            f"(Gamma^-1: {self.settings.gamma_inverse.value}, "
            f"delta = {self.settings.gamma_eps:g} * rms(gamma)):"
        )
        for component in model.get_modifiable_components():
            entries = [
                (layer_index, gain)
                for (layer_index, c), gain in self.post_norm_gains.items()
                if c == component and gain is not None
            ]
            if not entries:
                continue
            stacked = torch.stack([gain for _, gain in entries])
            abs_gains = stacked.abs()
            cv = (abs_gains.std(dim=1) / abs_gains.mean(dim=1)).mean().item()
            below = (
                (
                    abs_gains
                    < self.settings.gamma_eps
                    * abs_gains.pow(2).mean(dim=1, keepdim=True).sqrt()
                )
                .float()
                .mean()
                .item()
            )
            print(
                f"* [bold]{component}[/]: gamma in [{stacked.min().item():.3g}, "
                f"{stacked.max().item():.3g}], CV(|gamma|) {cv:.3f}, "
                f"{below:.1%} of |gamma| below delta"
            )
            # Off-target ||Gamma a - v|| per unit of v-readout removed, with each layer's own
            # direction (trials with a global direction are logged by modify_model()).
            per_layer = []
            for layer_index, gain in entries:
                v = self.residual_directions[layer_index + 1].cpu().float()
                pair = self.oblique_pair(v, gain)
                if pair is None:
                    continue
                off = self.off_target_of(pair[0], gain, v)
                self.off_target[(layer_index, component)] = off
                per_layer.append((layer_index, off))
            if per_layer:
                values = sorted(off for _, off in per_layer)
                worst = max(per_layer, key=lambda x: x[1])
                print(
                    f"  off-target ||Gamma a - v|| (per-layer directions): median "
                    f"{values[len(values) // 2]:.3f}, max {worst[1]:.3f} (layer {worst[0]})"
                )
                print(
                    "  by layer: "
                    + " ".join(
                        f"{layer_index}:{off:.2f}" for layer_index, off in per_layer
                    )
                )

    @staticmethod
    def off_target_of(a: Tensor, g: Tensor, v: Tensor) -> float:
        """||Gamma a - v||: off-target part of the residual-space edit per unit removed."""
        return LA.vector_norm(g * a - v).item()

    def warn_once(self, key: tuple[int, str], kind: str, message: str) -> None:
        if (key[0], key[1], kind) in self._warned:
            return
        self._warned.add((key[0], key[1], kind))
        print(f"[yellow]GammaMPOA warning (layer {key[0]} {key[1]}): {message}[/]")

    def modify_model(self, ctx: Context, parameters: Parameters) -> None:
        if not self.use_gamma_path:
            # Pre-norm model: exactly the MPOA edit (same code path, bit-identical).
            super().modify_model(ctx, parameters)
            return

        model = ctx.get_model()
        unit = self.settings.gamma_mode == GammaMode.UNIT

        # Direction selection and the per-component layer weight kernel are identical to
        # Abliteration.modify_model().
        if parameters.direction_index is None:
            residual_direction = None
        else:
            weight, index = math.modf(parameters.direction_index + 1)
            residual_direction = F.normalize(
                self.residual_directions[int(index)].lerp(
                    self.residual_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        for layer_index in range(len(model.get_layers())):
            for component, modules in model.get_layer_modules(layer_index).items():
                weight_distribution = parameters.weight_distributions[component]

                distance = abs(layer_index - weight_distribution.max_weight_position)

                if distance > weight_distribution.min_weight_distance:
                    continue

                weight = weight_distribution.max_weight + (
                    distance / weight_distribution.min_weight_distance
                ) * (weight_distribution.min_weight - weight_distribution.max_weight)

                if weight == 0:
                    continue

                if residual_direction is None:
                    layer_residual_direction = self.residual_directions[layer_index + 1]
                else:
                    layer_residual_direction = residual_direction

                key = (layer_index, component)
                gain = None if unit else self.post_norm_gains[key]

                for module in modules:
                    module = cast(Linear, module)

                    v = layer_residual_direction.to(module.weight.device)

                    base_weight = cast(Tensor, module.base_layer.weight)
                    quant_state = getattr(base_weight, "quant_state", None)

                    if quant_state is None:
                        W = base_weight.to(torch.float32)
                    else:
                        W = BNB_F.dequantize_4bit(
                            base_weight.data,
                            quant_state,
                        ).to(torch.float32)

                    W = W.view(W.shape[0], -1)

                    g = (
                        torch.ones_like(v)
                        if gain is None
                        else gain.to(device=v.device, dtype=torch.float32)
                    )

                    lora_A, lora_B = self.gamma_exact_lora(
                        W, v, g, weight, model.peft_config.r
                    )
                    self.check_edit(key, W, v, g, lora_A, lora_B)

                    weight_A = cast(Tensor, module.lora_A["default"].weight)
                    weight_B = cast(Tensor, module.lora_B["default"].weight)
                    weight_A.data = lora_A.to(weight_A.dtype)
                    weight_B.data = lora_B.to(weight_B.dtype)

    def check_edit(
        self,
        key: tuple[int, str],
        W: Tensor,
        v: Tensor,
        g: Tensor,
        lora_A: Tensor,
        lora_B: Tensor,
    ) -> None:
        """Record the deployed edit's size, row-norm error and off-target; warn on outliers."""
        stats = self.edit_stats.setdefault(
            key,
            {"relative_edit": 0.0, "max_row_norm_error": 0.0, "off_target": 0.0},
        )
        delta = lora_B @ lora_A
        if not bool(delta.any()):
            self.warn_once(
                key,
                "skip",
                "v has (numerically) no weight on dimensions with non-zero gain; "
                "writer left unedited",
            )
            return
        R = LA.vector_norm(W, dim=1)
        relative_edit = (LA.matrix_norm(delta) / LA.matrix_norm(W)).item()
        row_error = (
            ((LA.vector_norm(W + delta, dim=1) - R) / R.clamp_min(1e-30))
            .abs()
            .max()
            .item()
        )
        pair = self.oblique_pair(v, g)
        off = 0.0 if pair is None else self.off_target_of(pair[0], g, v)
        stats["relative_edit"] = max(stats["relative_edit"], relative_edit)
        stats["max_row_norm_error"] = max(stats["max_row_norm_error"], row_error)
        stats["off_target"] = max(stats["off_target"], off)
        if row_error > GUARD_MAX_ROW_NORM_ERROR or relative_edit > GUARD_RELATIVE_EDIT:
            errors = ((LA.vector_norm(W + delta, dim=1) - R) / R.clamp_min(1e-30)).abs()
            worst_row = int(errors.argmax().item())
            self.warn_once(
                key,
                "size",
                f"large edit: ||dW||/||W|| = {relative_edit:.3f}, max row-norm error "
                f"{row_error:.1%} (p99 {errors.quantile(0.99).item():.1%}; worst row: norm "
                f"{(R[worst_row] / R.median()).item():.2g} x median, gain "
                f"{g[worst_row].item():.3g}), off-target ||Gamma a - v|| = {off:.3f}",
            )

    def oblique_pair(self, v: Tensor, g: Tensor) -> tuple[Tensor, Tensor] | None:
        """
        Returns (a, b) with b = Gamma v, a proportional to the regularised Gamma^-1 v and
        b^T a = 1, or None if v has (numerically) no weight on dimensions with non-zero gain.

        Before normalisation b^T a = sum_i v_i^2 * gamma_i^2 / (gamma_i^2 + delta^2) (ridge) or
        sum_i v_i^2 * |gamma_i| / max(|gamma_i|, delta) (clamp): every term is >= 0, so there is
        no cancellation, and the value is a scale-invariant fraction in [0, 1]. The normalised
        edit a b^T W stays bounded however small that fraction is (for the ridge
        ||a b^T W|| <= sqrt(1 + (max|gamma| / delta)^2) * ||W||_2), so the edit is skipped only
        when the fraction is below MIN_LIVE_FRACTION. The clamp has no such bound.
        """
        b = g * v
        delta = self.settings.gamma_eps * gain_scale(g)
        if self.settings.gamma_inverse == GammaInverse.RIDGE:
            a = v * g / (g * g + delta * delta)
        else:
            sign = torch.where(g < 0, -1.0, 1.0).to(g.dtype)
            a = v / torch.where(g.abs() < delta, sign * delta, g)
        ba = b @ a
        if not torch.isfinite(ba) or ba <= MIN_LIVE_FRACTION:
            return None
        a = a / ba
        if not bool(torch.isfinite(a).all()):
            return None
        return a, b

    def gamma_exact_lora(
        self,
        W: Tensor,
        v: Tensor,
        g: Tensor,
        weight: float,
        r: int,
    ) -> tuple[Tensor, Tensor]:
        """
        LoRA factors (lora_A: (r', d_in), lora_B: (d_out, r')) of the gamma-exact edit of
        W (d_out, d_in, float32) for residual direction v and post-norm gain g, such that
        b^T (W + lora_B @ lora_A) = (1 - weight) * b^T W with b = g * v.
        """
        pair = self.oblique_pair(v, g)
        if pair is None:
            # The write has no component along v to remove: identity adapter.
            return (
                torch.zeros(r, W.shape[1], device=W.device),
                torch.zeros(W.shape[0], r, device=W.device),
            )
        a, b = pair

        # Exact rank-1 oblique projection: delta W = -weight * a (b^T W).
        lora_A = (b @ W).view(1, -1)
        lora_B = (-weight * a).view(-1, 1)

        # "none"/"pre" use rank 1 (set by Abliteration.init); "full" with rank 1 has no room
        # for the magnitude restoration, so it reduces to the exact projection.
        if self.settings.row_normalization != RowNormalization.FULL or r == 1:
            return lora_A, lora_B

        # MPOA's magnitude preservation: restore the original row norms of the projected matrix.
        W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
        W_full = F.normalize(W + lora_B @ lora_A, p=2, dim=1) * W_row_norms
        delta = W_full - W

        # Confine the row-norm restoration to ker(b^T): K = (I - a b^T) delta.
        # (I - a b^T) annihilates the rank-1 projection term, so K is the restoration only.
        K = delta - a.view(-1, 1) @ (b @ delta).view(1, -1)

        rank = r - 1
        # Same randomized SVD and reseeding as Abliteration.modify_model().
        torch.manual_seed(self.heretic_settings.seed)
        torch.cuda.manual_seed_all(self.heretic_settings.seed)  # ty:ignore[invalid-argument-type]
        U, S, V = torch.svd_lowrank(K, q=min(2 * rank + 4, *K.shape), niter=6)
        U = U[:, :rank]
        S = S[:rank]
        Vh = V[:, :rank].T
        # Remove the float-level b-component the SVD factors may carry.
        U = U - a.view(-1, 1) @ (b @ U).view(1, -1)

        sqrt_S = torch.sqrt(S)
        lora_B = torch.cat([lora_B, U @ torch.diag(sqrt_S)], dim=1)
        lora_A = torch.cat([lora_A, torch.diag(sqrt_S) @ Vh], dim=0)

        return lora_A, lora_B
