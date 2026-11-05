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

    W' = W - lambda * a b^T W,    b = Gamma v,   a = Gamma^-1 v / (b^T Gamma^-1 v)

Then Gamma o' = (I - lambda v v^T) Gamma o: before the per-token rms rescaling, the residual
write is exactly the orthogonal projection of the original write, and b^T W' = (1 - lambda) b^T W,
i.e. v^T y' = 0 at lambda = 1.

Exactness only needs b = Gamma v (true gains) and b^T a = 1; the removal direction a is free.
Real post-norm gains have many near-zero entries (Gemma-4-31B post_attention_layernorm: 23% of
gains below 5% of the mean), where Gamma^-1 explodes: the oblique edit then spends large o-space
changes on dimensions the writer can barely write to, which inflates rms(o) and breaks the row-norm
restoration (max row-norm error 44% with a 1%-of-mean clamp on a real Gemma-4 writer). Gamma^-1 is
therefore regularised (gamma_inverse = "ridge", default):

    a  proportional to  (Gamma^2 + delta^2)^-1 Gamma v  =  argmin_a ||Gamma a - v||^2 + delta^2 ||a||^2,
    delta = gamma_eps * mean |gamma|

which tends to the exact oblique pair as delta -> 0 and to the orthogonal projection against
Gamma v as delta -> inf. It is sign-safe and smooth; dimensions with |gamma_i| << delta are left
almost untouched. gamma_inverse = "clamp" uses sign(gamma) * max(|gamma|, delta) instead.
Neither choice affects exactness (b^T a = 1 is enforced by normalisation).

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
   the leak (mid-layer writers, random v: Gemma-2-27B 33-44%, Gemma-4-31B 58-63%,
   OLMo-2-32B 27-38%, OLMo-3.1-32B 8-9%).

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
of its whole delta). On the same real writers (ridge, gamma_eps = 0.1, rank 3) the mean row-norm
error is 1e-4-1e-3 and the max 0.2-4% (worst: Gemma-4, whose near-zero gains force the largest
o-space edits), versus mean ~2e-4 / max ~0.2% for MPOA. Because the restoration is confined to
ker(b^T), the exact edit also removes MPOA's own row-norm leak: on a post-norm model GammaMPOA
differs from MPOA by both the gamma correction and that leak (use exact_prenorm on pre-norm models
for the matched control). With row_normalization = "none" the edit is the rank-1 oblique projection.
"pre" only changes the projection geometry (no magnitude restoration), and exactness fixes the
geometry, so for gamma writers "pre" is identical to "none".

Pre-norm writers
----------------
If the model has no post-norm writers (Qwen, Llama, Mistral, ...), modify_model() delegates to
Abliteration.modify_model(): the edit is bit-identical to MPOA. Set exact_prenorm = true to apply
the exact edit with Gamma = 1 to such writers instead (a matched "exact MPOA" control that removes
MPOA's own row-norm leak). Mixed stacks (some writers post-norm, some not) are rejected.

Use with:
    modifiers = [{ plugin = "heretic.modifiers.gamma_mpoa.GammaMPOA" }]
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
        # The outer rms is still a per-token scalar, so the gains compose.
        chain = ("post_feedforward_layernorm_1", "post_feedforward_layernorm")
    return chain


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


class Settings(AbliterationSettings):
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
            "delta of the Gamma^-1 regularisation, as a fraction of the mean |gamma| of the "
            "writer's post-norm gain. Only shapes the removal direction; exactness "
            "(v^T y' = 0 at weight 1) does not depend on it."
        ),
    )

    exact_prenorm: bool = Field(
        default=False,
        description=(
            "Also apply the exact edit (with Gamma = 1) to models whose writers have no post-norm. "
            "Off (default): such models get the bit-identical MPOA edit. "
            "On: matched 'exact MPOA' control without MPOA's row-norm leak."
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
        if (
            self.settings.orthogonalize_direction
            and self.settings.row_normalization == RowNormalization.FULL
        ):
            return "Gamma-Exact Magnitude-Preserving Orthogonal Ablation (GammaMPOA)"
        return f"Gamma-Exact {super().modifier_name}"

    def init(self, ctx: Context) -> None:
        # Same direction estimation, LoRA rank and adapter setup as MPOA.
        super().init(ctx)
        self.detect_post_norm_gains(ctx)

    def detect_post_norm_gains(self, ctx: Context) -> None:
        model = ctx.get_model()
        layers = model.get_layers()

        # (layer_index, component) -> float32 gain on CPU; None for writers without a post-norm.
        self.post_norm_gains: dict[tuple[int, str], Tensor | None] = {}
        unknown_sandwich = set()

        for layer_index in range(len(layers)):
            layer = layers[layer_index]
            for component in model.get_layer_modules(layer_index):
                chain = post_norm_chain(layer, component)
                if chain is None:
                    if hasattr(layer, "post_feedforward_layernorm"):
                        unknown_sandwich.add(type(layer).__name__)
                    self.post_norm_gains[(layer_index, component)] = None
                    continue
                gain = None
                for norm_name in chain:
                    norm_gain = rmsnorm_gain(getattr(layer, norm_name)).cpu()
                    gain = norm_gain if gain is None else gain * norm_gain
                self.post_norm_gains[(layer_index, component)] = gain

        n_post = sum(gain is not None for gain in self.post_norm_gains.values())
        n_total = len(self.post_norm_gains)

        print()
        if unknown_sandwich:
            print(
                f"[yellow]Warning: layer class(es) {sorted(unknown_sandwich)} have a "
                "post_feedforward_layernorm but are not in POST_NORM_CHAINS; "
                "their writers are treated as pre-norm.[/]"
            )
        if 0 < n_post < n_total:
            raise ValueError(
                f"GammaMPOA: mixed pre/post-norm writers ({n_post} of {n_total}) are not supported"
            )

        self.use_gamma_path = n_post > 0 or self.settings.exact_prenorm

        if n_post == 0:
            if self.settings.exact_prenorm:
                print(
                    "No post-norm writers found: exact edit with Gamma = 1 (exact_prenorm)."
                )
            else:
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
            f"delta = {self.settings.gamma_eps:g} * mean |gamma|):"
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
                    < self.settings.gamma_eps * abs_gains.mean(dim=1, keepdim=True)
                )
                .float()
                .mean()
                .item()
            )
            # cos(Gamma a, v) = 1 / ||Gamma a||: how much of the residual-space edit is along v
            # (1 = pure oblique projection), with each layer's own direction.
            cosines = []
            for layer_index, gain in entries:
                pair = self.oblique_pair(
                    self.residual_directions[layer_index + 1].cpu(), gain
                )
                if pair is not None:
                    cosines.append(1 / LA.vector_norm(gain * pair[0]).item())
            cos_text = (
                f", cos(Gamma a, v) median {sorted(cosines)[len(cosines) // 2]:.3f} "
                f"min {min(cosines):.3f}"
                if cosines
                else ""
            )
            print(
                f"* [bold]{component}[/]: gamma in [{stacked.min().item():.3g}, "
                f"{stacked.max().item():.3g}], CV(|gamma|) {cv:.3f}, "
                f"{below:.1%} of |gamma| below delta{cos_text}"
            )

    def modify_model(self, ctx: Context, parameters: Parameters) -> None:
        if not self.use_gamma_path:
            # Pre-norm model: exactly the MPOA edit (same code path, bit-identical).
            super().modify_model(ctx, parameters)
            return

        model = ctx.get_model()

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

                gain = self.post_norm_gains[(layer_index, component)]

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

                    weight_A = cast(Tensor, module.lora_A["default"].weight)
                    weight_B = cast(Tensor, module.lora_B["default"].weight)
                    weight_A.data = lora_A.to(weight_A.dtype)
                    weight_B.data = lora_B.to(weight_B.dtype)

    def oblique_pair(self, v: Tensor, g: Tensor) -> tuple[Tensor, Tensor] | None:
        """
        Returns (a, b) with b = Gamma v, a proportional to the regularised Gamma^-1 v and
        b^T a = 1, or None if v has (numerically) no weight on dimensions with non-zero gain.
        """
        b = g * v
        delta = self.settings.gamma_eps * g.abs().mean()
        if self.settings.gamma_inverse == GammaInverse.RIDGE:
            a = v * g / (g * g + delta * delta)
        else:
            sign = torch.where(g < 0, -1.0, 1.0).to(g.dtype)
            a = v / torch.where(g.abs() < delta, sign * delta, g)
        ba = b @ a
        if not torch.isfinite(ba) or ba <= 1e-6:
            return None
        return a / ba, b

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
