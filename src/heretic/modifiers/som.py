# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Multi-directional refusal suppression with Self-Organizing Maps (SOM)
# (Piras et al. 2025, "SOM Directions are Better than One", https://arxiv.org/abs/2511.08379).
# Plugin port of the proof of concept in https://github.com/p-e-w/heretic/pull/196.
#
# Overview:
# 1. Collect per-prompt residuals for the bad prompts and the residual mean
#    for the good prompts (both via heretic's own residual collection).
# 2. For every layer, fit a small SOM (default 4x4) to the bad residuals.
#    The SOM is implemented in torch and trained for all layers in parallel.
# 3. Rank the neurons by how many bad residuals they win (data-driven top-k),
#    and turn the prototypes of the top k neurons into directions
#    `normalize(prototype - good_mean)` (optionally projected, as in Abliteration).
# 4. Ablate the k directions jointly with a rank-k LoRA, using the same
#    per-component layer weight kernel as Abliteration, plus one decay
#    parameter that controls how strongly the lower-ranked directions are ablated.

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, cast

import bitsandbytes.functional as BNB_F
import torch
import torch.linalg as LA
import torch.nn.functional as F
from optuna import Trial
from peft.tuners.lora.layer import Linear
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from torch import Tensor

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.modifier import Context, Modifier, Serializable
from heretic.modifiers.abliteration import RowNormalization, WeightDistribution
from heretic.utils import batchify, format_dataset_specification, print


@dataclass
class Parameters(Serializable):
    # Index of the layer whose SOM directions are used for all layers,
    # or None if each layer uses its own SOM directions.
    direction_index: int | None
    # Relative strength of the i-th ranked direction is direction_decay**i.
    direction_decay: float
    weight_distributions: dict[str, WeightDistribution]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_presentation_dict(self) -> dict[str, str]:
        parameters = {}

        parameters["direction_index"] = (
            "per layer" if (self.direction_index is None) else f"{self.direction_index}"
        )
        parameters["direction_decay"] = f"{self.direction_decay:.2f}"

        for component, weight_distribution in self.weight_distributions.items():
            for name, value in asdict(weight_distribution).items():
                parameters[f"{component}.{name}"] = f"{value:.2f}"

        return parameters

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        return Parameters(
            direction_index=data["direction_index"],
            direction_decay=data["direction_decay"],
            weight_distributions={
                component: WeightDistribution(**weight_distribution)
                for component, weight_distribution in data[
                    "weight_distributions"
                ].items()
            },
        )


class Topology(str, Enum):
    RECTANGULAR = "rectangular"
    HEXAGONAL = "hexagonal"


class Settings(BaseModel):
    good_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to produce desirable responses.",
    )

    bad_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
        ),
        description=(
            "Dataset of prompts that tend to produce undesirable responses. "
            "The SOM is fitted to the per-prompt residuals of these prompts."
        ),
    )

    som_x: PositiveInt = Field(
        default=4,
        description="Number of SOM neurons along the x-axis of the map.",
    )

    som_y: PositiveInt = Field(
        default=4,
        description="Number of SOM neurons along the y-axis of the map.",
    )

    som_topology: Topology = Field(
        default=Topology.HEXAGONAL,
        description=(
            'Grid topology of the SOM. Options: "rectangular", "hexagonal" '
            "(the default, matching the MiniSom-based proof of concept in PR #196)."
        ),
    )

    n_directions: PositiveInt = Field(
        default=4,
        description=(
            "Number of directions (k) to ablate per layer. These are derived from the k SOM "
            "neurons that win the most bad residuals. Must not exceed som_x * som_y."
        ),
    )

    som_iterations: PositiveInt = Field(
        default=10000,
        description=(
            "Number of online SOM training iterations. Each iteration presents one randomly "
            "drawn bad residual (independently drawn for each layer)."
        ),
    )

    som_learning_rate: PositiveFloat = Field(
        default=0.01,
        description=(
            "Initial SOM learning rate. Decays as lr / (1 + t / (T / 2)) "
            '(MiniSom\'s "asymptotic decay").'
        ),
    )

    som_sigma: PositiveFloat = Field(
        default=0.5,
        description=(
            "Initial radius of the Gaussian SOM neighborhood function, in grid units. "
            "Decays like the learning rate."
        ),
    )

    som_seed: int | None = Field(
        default=None,
        description=(
            "Seed for the SOM initialization and sampling order. "
            "If unset, the global heretic seed is used."
        ),
    )

    orthogonalize_direction: bool = Field(
        default=True,
        description=(
            "Whether to adjust each SOM direction so that only the component that is "
            "orthogonal to the good mean direction is ablated (as in Abliteration)."
        ),
    )

    orthonormalize_directions: bool = Field(
        default=True,
        description=(
            "Whether to Gram-Schmidt orthonormalize the k directions (in rank order) before "
            "ablating them, so that at full strength the whole k-dimensional subspace is "
            "projected out exactly. If false, the k rank-1 ablations are simply summed "
            "(as in the PR #196 proof of concept), which over-ablates shared components "
            "of non-orthogonal directions."
        ),
    )

    row_normalization: RowNormalization = Field(
        default=RowNormalization.FULL,
        description=(
            "How to apply row normalization of the weights. Options: "
            '"none" (no normalization), '
            '"pre" (compute LoRA adapter relative to row-normalized weights), '
            '"full" (like "pre", but renormalizes to preserve original row magnitudes).'
        ),
    )

    full_normalization_lora_rank: PositiveInt = Field(
        default=6,
        description=(
            'The rank of the LoRA adapter to use when "full" row normalization is used. '
            "The exact update before renormalization has rank n_directions, and renormalization "
            "adds non-linear effects, so this should be at least n_directions "
            "(the default of 6 is n_directions + 2 for the default of 4 directions). "
            'Without "full" row normalization, the LoRA rank is exactly n_directions.'
        ),
    )

    winsorization_quantile: float = Field(
        default=1.0,
        description=(
            "The symmetric winsorization to apply to the per-prompt, per-layer residual vectors, "
            "expressed as the quantile to clamp to (between 0 and 1). Disabled by default. "
            "Since the SOM uses Euclidean distances, this can stop a few massive activation "
            "components from dominating the map."
        ),
    )

    @model_validator(mode="after")
    def validate_n_directions(self) -> "Settings":
        if self.n_directions > self.som_x * self.som_y:
            raise ValueError(
                f"n_directions ({self.n_directions}) must not exceed the number of "
                f"SOM neurons ({self.som_x} * {self.som_y})"
            )
        return self


def get_grid_coordinates(som_x: int, som_y: int, topology: Topology) -> Tensor:
    """
    Returns the (som_x * som_y, 2) grid coordinates of the SOM neurons,
    in row-major order (neuron index = i * som_y + j).
    The hexagonal layout follows MiniSom (odd rows shifted by half a unit,
    rows spaced by sqrt(3)/2), so that all six neighbors are at distance 1.
    """
    i, j = torch.meshgrid(
        torch.arange(som_x, dtype=torch.float32),
        torch.arange(som_y, dtype=torch.float32),
        indexing="ij",
    )

    if topology == Topology.HEXAGONAL:
        # MiniSom: xx[::-2] -= 0.5 on the (y, x)-shaped meshgrid, yy *= sqrt(3)/2.
        i = i - 0.5 * (j % 2)
        j = j * (math.sqrt(3) / 2)

    return torch.stack([i.flatten(), j.flatten()], dim=1)


def fit_som(
    data: Tensor,
    som_x: int,
    som_y: int,
    iterations: int,
    learning_rate: float,
    sigma: float,
    topology: Topology,
    generator: torch.Generator,
) -> Tensor:
    """
    Fits independent online SOMs to a batch of datasets in parallel.

    Args:
        data: (n_maps, n_samples, dim) tensor, one dataset per map.
        generator: CPU generator used for initialization and sampling,
            which makes the result independent of the compute device's RNG.

    Returns:
        (n_maps, som_x * som_y, dim) tensor of neuron prototypes.
    """
    n_maps, n_samples, _ = data.shape
    n_neurons = som_x * som_y
    device = data.device

    # Initialize the prototypes with randomly drawn samples (MiniSom's random_weights_init).
    init_indices = torch.randint(
        n_samples, (n_maps, n_neurons), generator=generator
    ).to(device)
    weights = torch.gather(
        data,
        1,
        init_indices.unsqueeze(2).expand(-1, -1, data.shape[2]),
    ).clone()

    coordinates = get_grid_coordinates(som_x, som_y, topology).to(device)
    # (n_neurons, n_neurons) squared grid distances between all pairs of neurons.
    grid_distances = torch.cdist(coordinates, coordinates) ** 2

    sample_indices = torch.randint(
        n_samples, (iterations, n_maps), generator=generator
    ).to(device)
    map_indices = torch.arange(n_maps, device=device)

    half_iterations = iterations / 2

    for t in range(iterations):
        # (n_maps, dim)
        x = data[map_indices, sample_indices[t]]
        # (n_maps, n_neurons, dim)
        differences = x.unsqueeze(1) - weights

        # Best matching unit (ties resolve to the lowest index, deterministically).
        bmu = (differences**2).sum(dim=2).argmin(dim=1)

        decay = 1 + t / half_iterations
        eta = learning_rate / decay
        sig = sigma / decay

        # (n_maps, n_neurons) Gaussian neighborhood around the BMU on the grid.
        neighborhood = torch.exp(-grid_distances[bmu] / (2 * sig * sig))

        weights += (eta * neighborhood).unsqueeze(2) * differences

    return weights


def count_wins(data: Tensor, weights: Tensor) -> Tensor:
    """
    Counts how many samples each neuron wins (is the best matching unit for).

    Args:
        data: (n_maps, n_samples, dim)
        weights: (n_maps, n_neurons, dim)

    Returns:
        (n_maps, n_neurons) integer tensor of win counts.
    """
    bmu = torch.cdist(data, weights).argmin(dim=2)
    counts = torch.zeros(weights.shape[:2], dtype=torch.long, device=weights.device)
    counts.scatter_add_(1, bmu, torch.ones_like(bmu))
    return counts


def select_top_k_neurons(counts: Tensor, k: int) -> Tensor:
    """
    Returns the (n_maps, k) indices of the k neurons with the most wins, in descending
    order of wins. Ties are broken by neuron index (stable sort), for determinism.

    This replaces `get_top_k_neuron_weights` from the PR #196 proof of concept, which
    counted the winners over the SOM's own prototypes instead of over the data.
    Since every prototype is its own best matching unit, all counts were 1 there,
    and "top-k" degenerated to the first k neurons of the grid.
    """
    order = torch.sort(counts, dim=1, descending=True, stable=True).indices
    return order[:, :k]


def orthonormalize(directions: Tensor, eps: float = 1e-4) -> Tensor:
    """
    Gram-Schmidt orthonormalization of the (n_maps, k, dim) directions along k,
    in order (so the first direction is preserved exactly). A direction that is
    (numerically) contained in the span of the preceding ones is replaced by zero,
    which disables it instead of ablating an arbitrary noise direction.
    """
    basis = []

    for i in range(directions.shape[1]):
        v = directions[:, i, :]
        # Two passes ("twice is enough") for numerical stability.
        for _ in range(2):
            for q in basis:
                v = v - (v * q).sum(dim=1, keepdim=True) * q
        norm = LA.vector_norm(v, dim=1, keepdim=True)
        v = torch.where(norm > eps, v / norm.clamp_min(eps), torch.zeros_like(v))
        basis.append(v)

    return torch.stack(basis, dim=1)


class SOM(Modifier[Parameters]):
    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def modifier_name(self) -> str:
        return "Self-Organizing Map Multi-Directional Ablation (SOM)"

    def init(self, ctx: Context) -> None:
        model = ctx.get_model()

        print()
        print(
            f"Loading good prompts from [bold]{format_dataset_specification(self.settings.good_prompts)}[/]..."
        )
        good_prompts = ctx.load_prompts(self.settings.good_prompts)
        print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

        print()
        print(
            f"Loading bad prompts from [bold]{format_dataset_specification(self.settings.bad_prompts)}[/]..."
        )
        bad_prompts = ctx.load_prompts(self.settings.bad_prompts)
        print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

        print()
        print("Calculating per-layer SOM directions...")

        print("* Obtaining residual mean for good prompts...")
        good_means = model.get_residuals_mean(
            good_prompts,
            winsorization_quantile=self.settings.winsorization_quantile,
        )

        print("* Obtaining residuals for bad prompts...")
        # Move each batch to the CPU right away to keep peak VRAM usage low.
        # Shape: (prompt, layer, component).
        bad_residuals = torch.cat(
            [
                model.get_residuals(
                    batch,
                    winsorization_quantile=self.settings.winsorization_quantile,
                ).cpu()
                for batch in batchify(bad_prompts, model.settings.batch_size)
            ],
            dim=0,
        )

        self.residual_directions, self.win_fractions = self.compute_directions(
            good_means,
            bad_residuals,
            device=next(model.model.parameters()).device,
        )

        k = self.settings.n_directions
        print(
            f"* Top-{k} neurons win on average [bold]{self.win_fractions.sum(dim=1).mean().item():.0%}[/] "
            f"of bad residuals per layer (top-1: [bold]{self.win_fractions[:, 0].mean().item():.0%}[/])"
        )

        if self.settings.row_normalization != RowNormalization.FULL:
            # The update -Q diag(s) Q^T W has rank exactly k.
            self.lora_rank = k
        else:
            # Row magnitude preservation introduces nonlinear effects.
            self.lora_rank = self.settings.full_normalization_lora_rank
            if self.lora_rank < k:
                print(
                    f"* [yellow]Warning: full_normalization_lora_rank ({self.lora_rank}) is smaller "
                    f"than n_directions ({k}), so the ablation will be truncated.[/]"
                )

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.
        model.apply_lora(self.lora_rank)

    def compute_directions(
        self,
        good_means: Tensor,
        bad_residuals: Tensor,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            good_means: (layer, component) residual means for the good prompts.
            bad_residuals: (prompt, layer, component) residuals for the bad prompts.

        Returns:
            directions: (layer, k, component) unit directions (or zero for disabled
                directions), in descending order of SOM neuron wins. If
                orthonormalize_directions is set, they are orthonormal per layer.
            win_fractions: (layer, k) fraction of bad residuals won by each selected neuron.
        """
        settings = self.settings
        k = settings.n_directions

        seed = settings.som_seed
        if seed is None:
            seed = self.heretic_settings.seed or 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        # (layer, prompt, component)
        data = bad_residuals.transpose(0, 1).to(torch.float32)
        n_layers = data.shape[0]

        top_prototypes = []
        win_fractions = []

        # Fit the SOMs for several layers at once. Chunking bounds the memory
        # used on the compute device. Since the generator is consumed in a fixed
        # order, results don't depend on the device.
        chunk_size = 8
        for start in range(0, n_layers, chunk_size):
            chunk = data[start : start + chunk_size].to(device)

            weights = fit_som(
                chunk,
                som_x=settings.som_x,
                som_y=settings.som_y,
                iterations=settings.som_iterations,
                learning_rate=settings.som_learning_rate,
                sigma=settings.som_sigma,
                topology=settings.som_topology,
                generator=generator,
            )

            counts = count_wins(chunk, weights)
            top = select_top_k_neurons(counts, k)

            top_prototypes.append(
                torch.gather(
                    weights,
                    1,
                    top.unsqueeze(2).expand(-1, -1, weights.shape[2]),
                ).cpu()
            )
            win_fractions.append(
                (torch.gather(counts, 1, top).to(torch.float32) / chunk.shape[1]).cpu()
            )

        # (layer, k, component)
        prototypes = torch.cat(top_prototypes, dim=0)
        good_means = good_means.cpu().to(torch.float32)

        # Each SOM prototype plays the role of the "bad mean" in diff-in-means.
        directions = F.normalize(prototypes - good_means.unsqueeze(1), p=2, dim=2)

        if settings.orthogonalize_direction:
            # Implements https://huggingface.co/blog/grimjim/projected-abliteration
            # for each direction separately.
            good_directions = F.normalize(good_means, p=2, dim=1).unsqueeze(1)
            projections = (directions * good_directions).sum(dim=2, keepdim=True)
            directions = F.normalize(
                directions - projections * good_directions, p=2, dim=2
            )

        if settings.orthonormalize_directions:
            # Gram-Schmidt preserves orthogonality to the good direction,
            # since it only forms linear combinations of the directions.
            directions = orthonormalize(directions)

        return directions, torch.cat(win_fractions, dim=0)

    def suggest_parameters(self, ctx: Context, trial: Trial) -> Parameters:
        model = ctx.get_model()

        # Search space design: Abliteration's space (direction scope + index,
        # and a 4-parameter layer kernel per component) plus a single
        # direction_decay parameter, i.e. 11 parameters for a typical dense model.
        # The PR #196 proof of concept used per-direction max/min weights per
        # component (2 * k extra parameters per component, 21 for k = 4), which
        # converges poorly with TPE in a few hundred trials. With the geometric
        # decay, direction i (in descending order of SOM wins) is ablated with
        # strength kernel(layer) * decay**i. This nests the interesting cases:
        # decay = 0 is single-direction ablation along the dominant SOM cluster,
        # decay = 1 projects out the full k-dimensional subspace.

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        last_layer_index = len(model.get_layers()) - 1

        # Unlike Abliteration, the directions are not interpolated between layers:
        # the SOMs of different layers are fitted independently, so the i-th
        # direction of one layer has no correspondence with the i-th direction
        # of the next layer. A global scope therefore picks a single layer.
        #
        # As in Abliteration, this is always sampled because multivariate TPE
        # doesn't work with conditional parameters.
        direction_index = trial.suggest_int(
            "direction_index",
            round(0.4 * last_layer_index),
            round(0.9 * last_layer_index),
        )

        if direction_scope == "per layer":
            direction_index = None

        # The negative lower bound is clamped to 0, putting a positive probability
        # mass on exact single-direction ablation (same trick as for the MLP
        # max_weight in Abliteration).
        direction_decay = max(
            0.0,
            trial.suggest_float("direction_decay", -0.2, 1.0),
        )

        weight_distributions = {}

        for component in model.get_modifiable_components():
            # Same ranges as Abliteration (see the comments there).
            max_weight_lower_bound = -0.25 if component == "mlp.down_proj" else 0.8
            max_weight = max(
                0.0,
                trial.suggest_float(
                    f"{component}.max_weight",
                    max_weight_lower_bound,
                    1.5,
                ),
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                max(0.6 * last_layer_index, 1.0),
            )

            weight_distributions[component] = WeightDistribution(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        return Parameters(
            direction_index=direction_index,
            direction_decay=direction_decay,
            weight_distributions=weight_distributions,
        )

    def get_relative_strengths(self, parameters: Parameters) -> Tensor:
        # 0.0**0 == 1.0, so the top-ranked direction always has strength 1.
        return torch.tensor(
            [parameters.direction_decay**i for i in range(self.settings.n_directions)],
            dtype=torch.float32,
        )

    def compute_lora(
        self,
        W: Tensor,
        directions: Tensor,
        strengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Computes the LoRA factors (lora_A, lora_B) for a single weight matrix.

        Args:
            W: (d_out, d_in) FP32 weight matrix.
            directions: (k, d_out) directions in the output (residual) space.
            strengths: (k,) absolute ablation strengths of the directions.

        Without row normalization, the update is
            delta W = -sum_i s_i v_i v_i^T W = (-V^T diag(s)) (V W),
        i.e. k stacked rank-1 abliteration updates. If the directions are orthonormal
        and all s_i = 1, this is the orthogonal projection of W's outputs onto the
        complement of span(v_1, ..., v_k).
        """
        V = directions.to(W.device)
        s = strengths.to(W.device)

        if self.settings.row_normalization == RowNormalization.FULL:
            # Keep a reference to the original weight matrix so we can subtract it later.
            W_org = W

        if self.settings.row_normalization != RowNormalization.NONE:
            W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
            W = F.normalize(W, p=2, dim=1)

        # (k, d_in)
        lora_A = V @ W
        # (d_out, k)
        lora_B = -V.T * s.unsqueeze(0)

        if self.settings.row_normalization == RowNormalization.PRE:
            # Make the LoRA adapter apply to the original weight matrix.
            lora_B = W_row_norms * lora_B
        elif self.settings.row_normalization == RowNormalization.FULL:
            # Same as Abliteration: renormalize the rows, then approximate the
            # resulting delta with a low-rank SVD.
            W = W + lora_B @ lora_A
            W = F.normalize(W, p=2, dim=1)
            W = W * W_row_norms
            W = W - W_org
            r = self.lora_rank

            # svd_lowrank is randomized. Reseed immediately before the call so
            # restoring a trial is independent of RNG history.
            torch.manual_seed(self.heretic_settings.seed or 0)
            torch.cuda.manual_seed_all(self.heretic_settings.seed or 0)
            U, S, Vh = torch.svd_lowrank(W, q=min(2 * r + 4, min(W.shape)), niter=6)

            # Note: svd_lowrank actually returns V, so transpose it to get Vh.
            U = U[:, :r]
            S = S[:r]
            Vh = Vh[:, :r].T
            sqrt_S = torch.sqrt(S)
            lora_B = U @ torch.diag(sqrt_S)
            lora_A = torch.diag(sqrt_S) @ Vh

        return lora_A, lora_B

    def modify_model(self, ctx: Context, parameters: Parameters) -> None:
        model = ctx.get_model()

        relative_strengths = self.get_relative_strengths(parameters)

        for layer_index in range(len(model.get_layers())):
            for component, modules in model.get_layer_modules(layer_index).items():
                weight_distribution = parameters.weight_distributions[component]

                distance = abs(layer_index - weight_distribution.max_weight_position)

                # Don't ablate layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > weight_distribution.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance.
                weight = weight_distribution.max_weight + (
                    distance / weight_distribution.min_weight_distance
                ) * (weight_distribution.min_weight - weight_distribution.max_weight)

                # reset_model() has already left the adapter at identity.
                if weight == 0:
                    continue

                # The index must be shifted by 1 because the first element
                # of residual_directions is the direction for the embeddings.
                if parameters.direction_index is None:
                    directions = self.residual_directions[layer_index + 1]
                else:
                    directions = self.residual_directions[
                        parameters.direction_index + 1
                    ]

                strengths = weight * relative_strengths

                for module in modules:
                    # See the FIXME comments in Abliteration.modify_model
                    # regarding the validity of these casts.
                    module = cast(Linear, module)

                    base_weight = cast(Tensor, module.base_layer.weight)
                    quant_state = getattr(base_weight, "quant_state", None)

                    if quant_state is None:
                        W = base_weight.to(torch.float32)
                    else:
                        # 4-bit quantization.
                        W = BNB_F.dequantize_4bit(
                            base_weight.data,
                            quant_state,
                        ).to(torch.float32)

                    # Flatten weight matrix to (out_features, in_features).
                    W = W.view(W.shape[0], -1)

                    lora_A, lora_B = self.compute_lora(W, directions, strengths)

                    weight_A = cast(Tensor, module.lora_A["default"].weight)
                    weight_B = cast(Tensor, module.lora_B["default"].weight)
                    weight_A.data = lora_A.to(weight_A.dtype)
                    weight_B.data = lora_B.to(weight_B.dtype)

    def reset_model(self, ctx: Context) -> None:
        model = ctx.get_model()
        fast_path = model.reset_model()
        if not fast_path:
            model.apply_lora(self.lora_rank)
