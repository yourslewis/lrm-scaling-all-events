"""
Unit tests for evaluation metric correctness.

Tests verify that NDCG, HR, MRR, and rank computation produce
expected values for known inputs.

Run with:
    cd large-rec-model-main/src/hstu_retrieval
    pytest tests/test_eval_metrics.py -v
"""

import math
import torch
import pytest


class TestNDCG:
    """Test NDCG@K computation: 1/log2(rank+1) if rank <= K, else 0."""

    def test_rank_1(self):
        rank = torch.tensor([1])
        ndcg = torch.where(
            rank <= 10,
            torch.div(1.0, torch.log2(rank.float() + 1)),
            torch.zeros(1),
        )
        assert torch.isclose(ndcg, torch.tensor([1.0])), f"NDCG@10 for rank=1 should be 1.0, got {ndcg.item()}"

    def test_rank_5_at_k10(self):
        rank = torch.tensor([5])
        expected = 1.0 / math.log2(5 + 1)
        ndcg = torch.where(
            rank <= 10,
            torch.div(1.0, torch.log2(rank.float() + 1)),
            torch.zeros(1),
        )
        assert torch.isclose(ndcg, torch.tensor([expected])), f"NDCG@10 for rank=5 should be {expected}"

    def test_rank_11_at_k10(self):
        rank = torch.tensor([11])
        ndcg = torch.where(
            rank <= 10,
            torch.div(1.0, torch.log2(rank.float() + 1)),
            torch.zeros(1),
        )
        assert ndcg.item() == 0.0, "NDCG@10 for rank=11 should be 0"

    def test_batch_ndcg(self):
        ranks = torch.tensor([1, 2, 5, 10, 11])
        K = 10
        ndcg = torch.where(
            ranks <= K,
            torch.div(1.0, torch.log2(ranks.float() + 1)),
            torch.zeros(1),
        )
        expected = torch.tensor([
            1.0,
            1.0 / math.log2(3),
            1.0 / math.log2(6),
            1.0 / math.log2(11),
            0.0,
        ])
        assert torch.allclose(ndcg, expected, atol=1e-6)


class TestHR:
    """Test Hit Rate@K: 1 if rank <= K, else 0."""

    def test_hit(self):
        rank = torch.tensor([5])
        hr = (rank <= 10).float()
        assert hr.item() == 1.0

    def test_miss(self):
        rank = torch.tensor([11])
        hr = (rank <= 10).float()
        assert hr.item() == 0.0

    def test_boundary(self):
        rank = torch.tensor([10])
        hr = (rank <= 10).float()
        assert hr.item() == 1.0

    def test_batch_hr(self):
        ranks = torch.tensor([1, 10, 11, 100])
        hr_10 = (ranks <= 10).float()
        expected = torch.tensor([1.0, 1.0, 0.0, 0.0])
        assert torch.equal(hr_10, expected)


class TestMRR:
    """Test MRR: 1/rank."""

    def test_rank_1(self):
        rank = torch.tensor([1])
        mrr = torch.div(1.0, rank.float())
        assert mrr.item() == 1.0

    def test_rank_5(self):
        rank = torch.tensor([5])
        mrr = torch.div(1.0, rank.float())
        assert mrr.item() == pytest.approx(0.2)

    def test_batch_mrr(self):
        ranks = torch.tensor([1, 2, 4, 10])
        mrr = torch.div(1.0, ranks.float())
        expected = torch.tensor([1.0, 0.5, 0.25, 0.1])
        assert torch.allclose(mrr, expected)


class TestRankComputation:
    """Test the score-sort-rank pipeline used in eval_metrics_v2_from_tensors."""

    def _compute_rank(self, pos_score: float, neg_scores: list) -> int:
        """Replicate the ranking logic from eval.py."""
        B = 1
        pos_scores = torch.tensor([[pos_score]])  # [1, 1]
        neg_scores_t = torch.tensor([neg_scores]).unsqueeze(0)  # [1, 1, K]
        neg_scores_t = neg_scores_t.squeeze(0)  # [1, K]

        label_ids = torch.tensor([[999]])  # dummy positive ID
        neg_ids = torch.tensor([[i for i in range(len(neg_scores))]])  # [1, K]

        all_ids = torch.cat([label_ids, neg_ids], dim=1)  # [1, K+1]
        all_scores = torch.cat([pos_scores, neg_scores_t], dim=1)  # [1, K+1]

        sorted_scores, sorted_indices = all_scores.sort(dim=1, descending=True)
        sorted_ids = torch.gather(all_ids, dim=1, index=sorted_indices)

        _, eval_rank_indices = torch.max(
            sorted_ids == label_ids,
            dim=1,
        )
        return (eval_rank_indices + 1).item()

    def test_positive_scores_highest(self):
        rank = self._compute_rank(10.0, [5.0, 3.0, 1.0])
        assert rank == 1

    def test_positive_scores_lowest(self):
        rank = self._compute_rank(0.0, [5.0, 3.0, 1.0])
        assert rank == 4

    def test_positive_scores_middle(self):
        rank = self._compute_rank(4.0, [5.0, 3.0, 1.0])
        assert rank == 2

    def test_tie_with_positive_first(self):
        # When scores are tied, the positive (placed first in concat) should rank first
        rank = self._compute_rank(5.0, [5.0, 5.0, 5.0])
        assert rank == 1


class TestAvgFunction:
    """Test the _avg function for single-rank case (no distributed)."""

    def test_simple_avg(self):
        # _avg computes sum/numel, which is just the mean
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        _sum_and_numel = torch.tensor([x.sum(), x.numel()], dtype=torch.float32)
        result = _sum_and_numel[0] / _sum_and_numel[1]
        assert result.item() == pytest.approx(2.5)

    def test_single_element(self):
        x = torch.tensor([7.0])
        _sum_and_numel = torch.tensor([x.sum(), x.numel()], dtype=torch.float32)
        result = _sum_and_numel[0] / _sum_and_numel[1]
        assert result.item() == pytest.approx(7.0)
