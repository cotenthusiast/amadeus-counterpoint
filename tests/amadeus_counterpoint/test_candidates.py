import torch

from amadeus_counterpoint.models.candidates import select_candidates, topk_target_coverage

INF = float("inf")

# Six candidates, five legal (finite) plus one illegal (-inf), used by every
# test below that doesn't need a different ranking/legality pattern.
SAMPLE_LOGITS = torch.tensor([[5.0, 1.0, 9.0, 3.0, -INF, 7.0]])


def test_topk_selects_correct_values_and_indices():
    logits = SAMPLE_LOGITS

    indices, base_logits, valid, local_target = select_candidates(logits, k=3)

    assert local_target is None
    assert torch.equal(indices, torch.tensor([[2, 5, 0]]))
    assert torch.equal(base_logits, torch.tensor([[9.0, 7.0, 5.0]]))
    assert torch.equal(valid, torch.tensor([[True, True, True]]))


def test_target_already_in_topk_is_not_duplicated():
    logits = SAMPLE_LOGITS
    target = torch.tensor([2])  # value 9.0, already the top-1 candidate

    indices, base_logits, valid, local_target = select_candidates(
        logits, k=3, target_index=target
    )

    assert indices.shape == (1, 4)
    assert indices[0, :3].tolist() == [2, 5, 0]
    assert not valid[0, 3].item()
    assert local_target.item() == 0  # position of index 2 within the top-3


def test_target_absent_is_appended_with_real_gathered_logit():
    logits = SAMPLE_LOGITS
    target = torch.tensor([3])  # value 3.0, not in top-3 ([9, 7, 5])

    indices, base_logits, valid, local_target = select_candidates(
        logits, k=3, target_index=target
    )

    assert indices[0, :3].tolist() == [2, 5, 0]
    assert indices[0, 3].item() == 3
    assert base_logits[0, 3].item() == 3.0
    assert valid[0, 3].item()
    assert local_target.item() == 3


def test_batched_target_present_and_absent_mixed():
    logits = torch.tensor(
        [
            [5.0, 1.0, 9.0, 3.0, -INF, 7.0],
            [5.0, 1.0, 9.0, 3.0, -INF, 7.0],
        ]
    )
    target = torch.tensor([2, 3])  # row 0: present (top-1); row 1: absent

    indices, base_logits, valid, local_target = select_candidates(
        logits, k=3, target_index=target
    )

    assert not valid[0, 3].item()
    assert local_target[0].item() == 0

    assert valid[1, 3].item()
    assert indices[1, 3].item() == 3
    assert local_target[1].item() == 3


def test_fewer_than_k_finite_candidates_marks_overflow_invalid():
    logits = torch.tensor([[5.0, -INF, -INF, -INF]])

    indices, base_logits, valid, _ = select_candidates(logits, k=3)

    assert valid.tolist() == [[True, False, False]]
    assert base_logits[0, 0].item() == 5.0
    assert indices[0, 0].item() == 0


def test_fewer_than_k_finite_entries_with_target_among_them():
    # Only two finite (legal) entries exist; k=3, so one slot overflows to
    # -inf. Note a target can never be genuinely "absent" when the number of
    # finite entries is <= k -- topk always captures every finite entry
    # first -- so this exercises the overflow-invalid path together with the
    # target-already-present (no-op append) path simultaneously.
    logits = torch.tensor([[5.0, -INF, -INF, 2.0]])
    target = torch.tensor([3])  # the second finite entry (value 2.0)

    indices, base_logits, valid, local_target = select_candidates(
        logits, k=3, target_index=target
    )

    assert valid[0, 0].item()  # index 0, value 5.0
    assert valid[0, 1].item()  # index 3, value 2.0 -- both finite entries fit in top-3
    assert not valid[0, 2].item()  # -inf overflow slot
    assert not valid[0, 3].item()  # appended slot unused: target already present

    assert indices[0, local_target.item()].item() == 3


def test_topk_target_coverage_true_when_present():
    logits = SAMPLE_LOGITS
    coverage = topk_target_coverage(logits, k=3, target_index=torch.tensor([2]))
    assert coverage.tolist() == [True]


def test_topk_target_coverage_false_when_absent():
    logits = SAMPLE_LOGITS
    coverage = topk_target_coverage(logits, k=3, target_index=torch.tensor([3]))
    assert coverage.tolist() == [False]


def test_topk_target_coverage_matches_select_candidates_append_decision():
    logits = SAMPLE_LOGITS
    target = torch.tensor([3])

    coverage = topk_target_coverage(logits, k=3, target_index=target)
    _, _, _, local_target = select_candidates(logits, k=3, target_index=target)

    assert not coverage.item()
    assert local_target.item() == 3  # appended (last) slot, consistent with coverage=False


def test_dtype_shape_and_device_sanity():
    logits = torch.randn(4, 20)
    target = torch.randint(0, 20, (4,))

    indices, base_logits, valid, local_target = select_candidates(logits, k=5, target_index=target)

    assert indices.shape == (4, 6)
    assert base_logits.shape == (4, 6)
    assert valid.shape == (4, 6)
    assert local_target.shape == (4,)

    assert indices.dtype == torch.int64
    assert base_logits.dtype == logits.dtype
    assert valid.dtype == torch.bool
    assert local_target.dtype == torch.int64
    assert indices.device == logits.device
