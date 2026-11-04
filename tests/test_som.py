# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import unittest
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.linalg as LA
import torch.nn.functional as F
from pydantic import ValidationError

from heretic.modifiers.abliteration import RowNormalization, WeightDistribution
from heretic.modifiers.som import (
    SOM,
    Parameters,
    Settings,
    Topology,
    count_wins,
    fit_som,
    get_grid_coordinates,
    orthonormalize,
    select_top_k_neurons,
)


def make_som(**settings: Any) -> SOM:
    heretic_settings = cast(Any, SimpleNamespace(seed=1234))
    return SOM(heretic_settings=heretic_settings, settings=Settings(**settings))


def make_clusters(sizes: list[int], dim: int, seed: int = 0) -> tuple[Any, Any]:
    """Well-separated Gaussian clusters; returns (data, centers)."""
    generator = torch.Generator().manual_seed(seed)
    centers = 20 * torch.randn(len(sizes), dim, generator=generator)
    data = torch.cat(
        [
            center + 0.1 * torch.randn(size, dim, generator=generator)
            for center, size in zip(centers, sizes)
        ]
    )
    # Shuffle so that cluster membership is not encoded in the sample order.
    data = data[torch.randperm(data.shape[0], generator=generator)]
    return data, centers


def fit(data, seed=0, som_x=3, som_y=3, iterations=3000, topology=Topology.HEXAGONAL):
    return fit_som(
        data,
        som_x=som_x,
        som_y=som_y,
        iterations=iterations,
        learning_rate=0.5,
        sigma=1.0,
        topology=topology,
        generator=torch.Generator().manual_seed(seed),
    )


class GridTests(unittest.TestCase):
    def test_hexagonal_neighbors_at_unit_distance(self) -> None:
        coordinates = get_grid_coordinates(4, 4, Topology.HEXAGONAL)
        distances = torch.cdist(coordinates, coordinates)
        # Interior neurons of a hexagonal grid have exactly six neighbors at distance 1.
        interior = [5, 6, 9, 10]
        for neuron in interior:
            n_neighbors = ((distances[neuron] - 1).abs() < 1e-5).sum().item()
            self.assertEqual(n_neighbors, 6)

    def test_rectangular_grid(self) -> None:
        coordinates = get_grid_coordinates(2, 3, Topology.RECTANGULAR)
        self.assertEqual(coordinates.tolist()[4], [1.0, 1.0])


class FitTests(unittest.TestCase):
    def test_deterministic_given_seed(self) -> None:
        data, _ = make_clusters([40, 30, 20], dim=16)
        data = data.unsqueeze(0)
        a = fit(data, seed=7)
        b = fit(data, seed=7)
        c = fit(data, seed=8)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))

    def test_prototypes_cover_clusters(self) -> None:
        data, centers = make_clusters([50, 30, 20], dim=16)
        weights = fit(data.unsqueeze(0))[0]
        # Every cluster center must have a prototype close to it.
        distances = torch.cdist(centers, weights).min(dim=1).values
        self.assertTrue(
            (distances < 1.0).all(), msg=f"center distances: {distances.tolist()}"
        )

    def test_parallel_maps_are_independent_datasets(self) -> None:
        a, _ = make_clusters([30, 30], dim=8, seed=1)
        b, _ = make_clusters([30, 30], dim=8, seed=2)
        weights = fit(torch.stack([a, b]))
        # Each map's prototypes must lie near its own data, not the other map's.
        self.assertLess(
            torch.cdist(weights[0], a).min(dim=1).values.max(),
            torch.cdist(weights[0], b).min(dim=1).values.min(),
        )


class TopKTests(unittest.TestCase):
    def test_count_wins_matches_brute_force(self) -> None:
        generator = torch.Generator().manual_seed(0)
        data = torch.randn(2, 50, 8, generator=generator)
        weights = torch.randn(2, 6, 8, generator=generator)
        counts = count_wins(data, weights)
        self.assertEqual(counts.sum(dim=1).tolist(), [50, 50])
        for m in range(2):
            for neuron in range(6):
                expected = sum(
                    int(
                        torch.argmin(((weights[m] - x) ** 2).sum(dim=1)).item()
                        == neuron
                    )
                    for x in data[m]
                )
                self.assertEqual(counts[m, neuron].item(), expected)

    def test_top_k_ranked_by_data_wins(self) -> None:
        # Cluster sizes deliberately not in grid order.
        sizes = [10, 60, 30]
        data, centers = make_clusters(sizes, dim=16, seed=3)
        data = data.unsqueeze(0)
        weights = fit(data, som_x=2, som_y=2)
        counts = count_wins(data, weights)
        top = select_top_k_neurons(counts, 3)[0]
        top_prototypes = weights[0, top]
        # The i-th ranked prototype must be closest to the i-th largest cluster.
        nearest_cluster = torch.cdist(top_prototypes, centers).argmin(dim=1)
        self.assertEqual(nearest_cluster.tolist()[:2], [1, 2])
        # Counts must be non-increasing along the ranking.
        ranked_counts = counts[0, top]
        self.assertTrue((ranked_counts[:-1] >= ranked_counts[1:]).all())

    def test_regression_poc_bug(self) -> None:
        # The PR #196 PoC counted winners over the prototypes themselves.
        # Every (distinct) prototype wins only itself, so the counts carry no
        # information about the data and "top-k" is just the first k neurons.
        weights = torch.randn(1, 9, 16, generator=torch.Generator().manual_seed(0))
        poc_counts = count_wins(weights, weights)
        self.assertTrue((poc_counts == 1).all())
        self.assertEqual(select_top_k_neurons(poc_counts, 3)[0].tolist(), [0, 1, 2])
        # Counting over the data instead ranks a neuron that sits on the
        # largest cluster first, wherever it is on the grid.
        data, centers = make_clusters([10, 60, 30], dim=16, seed=3)
        data = data.unsqueeze(0)
        weights = fit(data, som_x=2, som_y=2)
        top1 = select_top_k_neurons(count_wins(data, weights), 1)[0, 0]
        top1_cluster = torch.cdist(weights[0, top1 : top1 + 1], centers).argmin()
        self.assertEqual(top1_cluster.item(), 1)

    def test_ties_broken_by_index(self) -> None:
        counts = torch.tensor([[3, 5, 5, 0, 3]])
        self.assertEqual(select_top_k_neurons(counts, 4)[0].tolist(), [1, 2, 0, 4])


class DirectionTests(unittest.TestCase):
    def test_orthonormalize(self) -> None:
        generator = torch.Generator().manual_seed(0)
        directions = F.normalize(torch.randn(3, 4, 32, generator=generator), dim=2)
        # Make the last direction a linear combination of the first two.
        directions[:, 3] = F.normalize(directions[:, 0] + directions[:, 1], dim=1)
        basis = orthonormalize(directions)
        self.assertTrue(torch.allclose(basis[:, 0], directions[:, 0], atol=1e-6))
        gram = basis[:, :3] @ basis[:, :3].transpose(1, 2)
        self.assertTrue(torch.allclose(gram, torch.eye(3).expand(3, 3, 3), atol=1e-5))
        # Dependent direction is disabled instead of becoming noise.
        self.assertTrue((basis[:, 3] == 0).all())

    def test_compute_directions(self) -> None:
        n_layers, dim, k = 3, 24, 4
        generator = torch.Generator().manual_seed(0)
        good_means = torch.randn(n_layers, dim, generator=generator)
        bad_residuals = torch.stack(
            [make_clusters([40, 25, 15, 10], dim, seed=s)[0] for s in range(n_layers)],
            dim=1,
        )
        som = make_som(n_directions=k, som_iterations=2000, som_learning_rate=0.5)
        directions, win_fractions = som.compute_directions(good_means, bad_residuals)
        self.assertEqual(tuple(directions.shape), (n_layers, k, dim))
        self.assertEqual(tuple(win_fractions.shape), (n_layers, k))
        self.assertTrue((win_fractions[:, :-1] >= win_fractions[:, 1:]).all())
        self.assertTrue((win_fractions.sum(dim=1) <= 1 + 1e-6).all())
        # Orthogonal to the good mean direction and orthonormal among each other.
        good_directions = F.normalize(good_means, dim=1).unsqueeze(2)
        self.assertLess((directions @ good_directions).abs().max().item(), 1e-5)
        gram = directions @ directions.transpose(1, 2)
        self.assertTrue(torch.allclose(gram, torch.eye(k).expand_as(gram), atol=1e-5))
        # Reproducible with the same seed.
        again, _ = som.compute_directions(good_means, bad_residuals)
        self.assertTrue(torch.equal(directions, again))

    def test_compute_directions_without_orthonormalization(self) -> None:
        good_means = torch.zeros(1, 8)
        bad_residuals = torch.randn(
            30, 1, 8, generator=torch.Generator().manual_seed(0)
        )
        som = make_som(
            n_directions=2,
            som_iterations=200,
            orthogonalize_direction=False,
            orthonormalize_directions=False,
        )
        directions, _ = som.compute_directions(good_means, bad_residuals)
        # With a zero good mean, each direction is the normalized prototype.
        self.assertTrue(
            torch.allclose(LA.vector_norm(directions, dim=2), torch.ones(1, 2))
        )


class LoRATests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.W = torch.randn(48, 40, generator=generator, dtype=torch.float32)
        raw = F.normalize(torch.randn(1, 3, 48, generator=generator), dim=2)
        self.V = orthonormalize(raw)[0]

    def lora(self, row_normalization, strengths, rank=None):
        som = make_som(n_directions=3, row_normalization=row_normalization)
        som.lora_rank = rank or 3
        A, B = som.compute_lora(self.W, self.V, strengths)
        return A, B

    def test_none_full_strength_projects_out_subspace(self) -> None:
        A, B = self.lora(RowNormalization.NONE, torch.ones(3))
        self.assertEqual(tuple(A.shape), (3, 40))
        self.assertEqual(tuple(B.shape), (48, 3))
        W_new = self.W + B @ A
        self.assertLess((self.V @ W_new).abs().max().item(), 1e-5)
        P = torch.eye(48) - self.V.T @ self.V
        self.assertTrue(torch.allclose(W_new, P @ self.W, atol=1e-5))

    def test_none_equals_stacked_rank_one_updates(self) -> None:
        strengths = torch.tensor([1.2, 0.5, 0.25])
        A, B = self.lora(RowNormalization.NONE, strengths)
        expected = sum(
            -s * torch.outer(v, v @ self.W) for v, s in zip(self.V, strengths)
        )
        self.assertTrue(torch.allclose(B @ A, cast(Any, expected), atol=1e-5))

    def test_zero_decay_is_single_direction_abliteration(self) -> None:
        som = make_som(n_directions=3)
        parameters = Parameters(
            direction_index=None, direction_decay=0.0, weight_distributions={}
        )
        self.assertEqual(som.get_relative_strengths(parameters).tolist(), [1, 0, 0])
        parameters.direction_decay = 0.5
        self.assertEqual(
            som.get_relative_strengths(parameters).tolist(), [1, 0.5, 0.25]
        )

    def test_pre(self) -> None:
        A, B = self.lora(RowNormalization.PRE, torch.ones(3))
        norms = LA.vector_norm(self.W, dim=1, keepdim=True)
        W_n = self.W / norms
        P = torch.eye(48) - self.V.T @ self.V
        self.assertTrue(torch.allclose(B @ A, norms * (P @ W_n - W_n), atol=1e-5))

    def test_full_preserves_row_norms(self) -> None:
        A, B = self.lora(RowNormalization.FULL, torch.ones(3), rank=6)
        self.assertEqual(tuple(A.shape), (6, 40))
        norms = LA.vector_norm(self.W, dim=1, keepdim=True)
        W_n = self.W / norms
        P = torch.eye(48) - self.V.T @ self.V
        reference = F.normalize(P @ W_n, dim=1) * norms
        W_new = self.W + B @ A
        rel = LA.matrix_norm(W_new - reference) / LA.matrix_norm(reference)
        self.assertLess(rel.item(), 0.05)
        self.assertTrue(
            torch.allclose(LA.vector_norm(W_new, dim=1), norms.squeeze(1), rtol=0.1)
        )


class SerializationTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        parameters = Parameters(
            direction_index=12,
            direction_decay=0.3,
            weight_distributions={
                "attn.o_proj": WeightDistribution(1.0, 20.0, 0.5, 8.0),
            },
        )
        data = json.loads(json.dumps(parameters.to_dict()))
        self.assertEqual(Parameters.from_dict(data), parameters)
        presentation = parameters.to_presentation_dict()
        self.assertEqual(presentation["direction_index"], "12")
        self.assertEqual(presentation["attn.o_proj.max_weight"], "1.00")

    def test_settings_validation(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(som_x=2, som_y=2, n_directions=5)
        Settings(som_x=2, som_y=2, n_directions=4)

    def test_plugin_contract(self) -> None:
        SOM.validate_contract()
        self.assertIs(SOM.get_settings_model(), Settings)
        self.assertIs(make_som().parameters_class, Parameters)
        self.assertTrue(make_som().reproducible)


if __name__ == "__main__":
    unittest.main()
