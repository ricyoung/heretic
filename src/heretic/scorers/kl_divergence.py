# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor

from heretic.config import DatasetSpecification
from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import print


class Settings(BaseModel):
    prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        ),
        description="Prompt dataset used to measure KL divergence from original model.",
    )

    tokens: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of response tokens to measure KL divergence over. "
            "With 1, only the first token is compared. With more, the original model's "
            "greedy continuation is generated once, and both models are scored on those "
            "same tokens (teacher forcing), averaging KL divergence over all positions "
            "up to the end of the response."
        ),
    )


class KLDivergence(Scorer):
    """
    KL divergence between current model and baseline.

    Measures how much the model's behavior has drifted from baseline.
    Lower is better (less damage).
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "KL divergence"

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading KL divergence evaluation prompts from [bold]{self.settings.prompts.dataset}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

        if self.settings.tokens == 1:
            print("* Obtaining baseline first-token probability distributions...")
            baseline_logits = ctx.get_logits(self.prompts)
        else:
            print(
                f"* Generating baseline continuations of [bold]{self.settings.tokens}[/] tokens..."
            )
            self._continuations, self._mask = ctx.get_continuations(
                self.prompts, self.settings.tokens
            )
            print("* Obtaining baseline probability distributions...")
            baseline_logits = ctx.get_continuation_logits(
                self.prompts, self._continuations
            )
            self._mask = self._mask.to(baseline_logits.device)

        self._baseline_logprobs = F.log_softmax(baseline_logits, dim=-1)

    def _get_kl_divergence(self, logits: Tensor) -> float:
        logprobs = F.log_softmax(logits, dim=-1)

        if self.settings.tokens == 1:
            return F.kl_div(
                logprobs,
                self._baseline_logprobs,
                reduction="batchmean",
                log_target=True,
            ).item()

        # Per-position KL divergence, shape (prompt, token), averaged over the
        # positions that belong to the response (up to and including EOS).
        kl_divergence = F.kl_div(
            logprobs,
            self._baseline_logprobs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        return kl_divergence[self._mask].mean().item()

    def get_score(self, ctx: Context) -> Score:
        if self.settings.tokens == 1:
            logits = ctx.get_logits(self.prompts)
        else:
            logits = ctx.get_continuation_logits(self.prompts, self._continuations)

        kl_divergence = self._get_kl_divergence(logits)

        return Score(
            value=kl_divergence,
            rich_display=f"[bold]{kl_divergence:.4f}[/]",
            md_display=f"{kl_divergence:.4f}",
        )

    def get_baseline_score(self, ctx: Context) -> Score:
        return Score(
            value=0,
            rich_display="[bold]0[/] [italic](by definition)[/]",
            md_display="0 *(by definition)*",
        )
