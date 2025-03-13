from src import shape_gen
import pytest

def test_load():
    try:
        shape_gen.generate_dataset()
    except Exception as e:
        pytest.fail(f"shape_gen.generate_dataset() raised an exception: {e}")
