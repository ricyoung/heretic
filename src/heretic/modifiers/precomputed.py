# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026  Richard Young + contributors

"""Directional ablation with externally computed per-layer directions.

Uses exactly the same weight edit, parameter search and row normalization as the
built-in Abliteration modifier; only the *source* of the refusal directions differs.
This isolates "where does the direction come from" from "how is it removed", e.g. for
Jacobian-lens pullback directions (J-space), SVD/cone directions, or new methods.

Direction file (torch.save): {"directions": Tensor[n_layers + 1, d_model], "meta": {...}}
Row 0 = embeddings, row l + 1 = output of block l (the same layout as
Abliteration.residual_directions).

Use with:
    modifiers = [{ plugin = "heretic.modifiers.precomputed.PrecomputedDirections" }]
    [modifier.PrecomputedDirections]
    directions_path = "/path/to/directions.pt"
"""

import torch
import torch.nn.functional as F
from pydantic import Field

from heretic.modifier import Context
from heretic.modifiers.abliteration import Abliteration, RowNormalization
from heretic.modifiers.abliteration import Settings as AbliterationSettings
from heretic.utils import print


class Settings(AbliterationSettings):
    directions_path: str = Field(
        description="Path to a torch file with a [n_layers + 1, d_model] 'directions' tensor.",
    )

    method_label: str = Field(
        default="Precomputed-Direction Ablation",
        description="Name shown in logs and model cards (e.g. 'J-space pullback ablation').",
    )

    # J-space style directions are defined without a harmless contrast set,
    # so projecting out the harmless mean is off by default.
    orthogonalize_direction: bool = Field(
        default=False,
        description=(
            "Adjust the directions so that only the component orthogonal to the "
            "harmless-prompt residual mean is removed (needs good_prompts)."
        ),
    )


class PrecomputedDirections(Abliteration):
    settings: Settings

    @property
    def modifier_name(self) -> str:
        return self.settings.method_label

    def init(self, ctx: Context) -> None:
        model = ctx.get_model()
        n_layers = len(model.get_layers())

        print()
        print(f"Loading precomputed directions from [bold]{self.settings.directions_path}[/]...")
        data = torch.load(self.settings.directions_path, map_location="cpu", weights_only=False)
        directions = data["directions"] if isinstance(data, dict) else data
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        directions = directions.to(torch.float32)

        if directions.shape[0] != n_layers + 1:
            raise ValueError(
                f"direction file has {directions.shape[0]} rows, expected n_layers + 1 = {n_layers + 1}"
            )
        print(f"* shape {tuple(directions.shape)}, method: {meta.get('method', 'unknown')}")

        directions = F.normalize(directions, p=2, dim=1)

        if self.settings.orthogonalize_direction:
            print("* Obtaining residual mean for good prompts (for orthogonalization)...")
            good_prompts = ctx.load_prompts(self.settings.good_prompts)
            good_means = model.get_residuals_mean(
                good_prompts,
                winsorization_quantile=self.settings.winsorization_quantile,
            )
            good_directions = F.normalize(good_means.to(torch.float32).cpu(), p=2, dim=1)
            projection = torch.sum(directions * good_directions, dim=1)
            directions = F.normalize(
                directions - projection.unsqueeze(1) * good_directions, p=2, dim=1
            )

        # modify_model() moves each direction to the weight's device.
        self.residual_directions = directions.cpu()

        if self.settings.row_normalization != RowNormalization.FULL:
            self.lora_rank = 1
        else:
            self.lora_rank = self.settings.full_normalization_lora_rank

        model.apply_lora(self.lora_rank)
