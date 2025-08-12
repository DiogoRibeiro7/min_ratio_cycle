"""Tests for validation helpers."""

import pytest

from min_ratio_cycle.exceptions import (
    CycleValidationError,
    GraphStructureError,
    NumericalInstabilityError,
    ValidationError,
)
from utils.validation import (
    ValidationHelper,
    generate_validation_report,
    post_solve_validate,
    pre_solve_validate,
    validate_cycle,
    validate_graph,
)


def test_validate_graph_ok():
    edges = [(0, 1, 2.0, 1.0), (1, 0, 1.0, 1.0)]
    validate_graph(2, edges)  # should not raise


def test_validate_graph_bad_vertex():
    edges = [(0, 2, 1, 1)]  # vertex 2 out of range for n=2
    with pytest.raises(ValidationError):
        validate_graph(2, edges)


def test_validate_graph_non_positive_time():
    edges = [(0, 1, 1, 0)]
    with pytest.raises(ValidationError, match="non-positive transit time"):
        validate_graph(2, edges)

    edges = [(0, 1, 1, -1)]
    with pytest.raises(ValidationError, match="non-positive transit time"):
        validate_graph(2, edges)


def test_validate_graph_non_numeric_weights():
    edges = [(0, 1, "a", 1)]
    with pytest.raises(ValidationError, match="non-numeric weights"):
        validate_graph(2, edges)


def test_validate_cycle_ok():
    edges = {(0, 1): (2.0, 1.0), (1, 0): (1.0, 1.0)}
    cost, time, ratio = validate_cycle([0, 1, 0], edges)
    assert cost == pytest.approx(3.0)
    assert time == pytest.approx(2.0)
    assert ratio == pytest.approx(1.5)


def test_validate_cycle_not_closed():
    edges = {(0, 1): (1.0, 1.0)}
    with pytest.raises(CycleValidationError):
        validate_cycle([0, 1], edges)


def test_validate_cycle_missing_edge():
    edges = {(0, 1): (1.0, 1.0)}
    with pytest.raises(CycleValidationError, match="Missing edge"):
        validate_cycle([0, 1, 2, 0], edges)


def test_helper_class():
    edges = {(0, 1): (2.0, 1.0), (1, 0): (1.0, 1.0)}
    ValidationHelper.validate_graph(2, [(0, 1, 2.0, 1.0), (1, 0, 1.0, 1.0)])
    cost, time, ratio = ValidationHelper.validate_cycle([0, 1, 0], edges)
    assert ratio == pytest.approx(1.5)


def test_generate_graph_report():
    report = generate_validation_report(2, [(0, 2, 1, 1)])
    assert not report.is_valid
    assert any("outside" in issue.message for issue in report.issues)


def test_validation_helper_generate_report():
    report = ValidationHelper.generate_graph_report(2, [(0, 2, 1, 1)])
    assert not report.is_valid


def test_pre_solve_detects_disconnected():
    edges = [(0, 1, 1.0, 1.0)]
    with pytest.raises(GraphStructureError, match="disconnected"):
        pre_solve_validate(3, edges)


def test_pre_solve_detects_dag():
    edges = [(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0)]
    with pytest.raises(GraphStructureError, match="acyclic"):
        pre_solve_validate(3, edges)


def test_pre_solve_condition_number():
    edges = [(0, 1, 1e9, 1.0), (1, 0, 1.0, 1.0)]
    with pytest.raises(NumericalInstabilityError):
        pre_solve_validate(2, edges, cond_threshold=1e6, weight_limit=1e12)


def test_pre_solve_weight_limit():
    edges = [(0, 1, 1e13, 1.0), (1, 0, 1.0, 1.0)]
    with pytest.raises(ValidationError, match="exceeds"):
        pre_solve_validate(2, edges, weight_limit=1e12)


def test_post_solve_validate_ok():
    edge_lookup = {(0, 1): (2.0, 1.0), (1, 0): (1.0, 1.0)}
    cost, time, ratio = post_solve_validate([0, 1, 0], 1.5, edge_lookup)
    assert cost == pytest.approx(3.0)
    assert time == pytest.approx(2.0)
    assert ratio == pytest.approx(1.5)


def test_post_solve_validate_mismatch():
    edge_lookup = {(0, 1): (2.0, 1.0), (1, 0): (1.0, 1.0)}
    with pytest.raises(CycleValidationError, match="ratio"):
        post_solve_validate([0, 1, 0], 2.0, edge_lookup)
