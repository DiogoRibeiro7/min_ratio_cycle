from min_ratio_cycle.exceptions import (
    GraphStructureError,
    NumericalInstabilityError,
    ResourceExhaustionError,
)


def test_exception_details():
    err = NumericalInstabilityError(
        "unstable",
        suggested_fix="increase precision",
        recovery_hint="use exact mode",
    )
    assert "increase precision" in str(err)
    assert err.details["recovery_hint"] == "use exact mode"

    gerr = GraphStructureError("bad graph", suggested_fix="connect components")
    assert gerr.details["suggested_fix"] == "connect components"

    rerr = ResourceExhaustionError("oom", resource="memory", limit=1, usage=2)
    assert rerr.details["usage"] == 2
