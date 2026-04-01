from yourtts.engines.standard import StandardEngine
from yourtts.engines.vieneu_turbo import VieneuTurboEngine
from yourtts.factory import create_engine


def test_create_engine_standard() -> None:
    engine = create_engine(mode="standard")
    assert isinstance(engine, StandardEngine)


def test_create_engine_invalid_mode() -> None:
    try:
        create_engine(mode="unknown")
    except ValueError as exc:
        assert "Unsupported engine mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported mode")


def test_create_engine_turbo_mode() -> None:
    engine = create_engine(mode="turbo")
    assert isinstance(engine, VieneuTurboEngine)
