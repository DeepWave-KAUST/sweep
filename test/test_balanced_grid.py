"""balanced_grid picks the (py, px) that the 8x V100 measurements showed
fastest for strong scaling (squarest tile, fatter contiguous x; py<=2 unless
y-thin is explicitly allowed). Pure arithmetic — no GPU / no distributed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sweep.parallel import balanced_grid  # noqa: E402


def test_safe_default_beats_x_cut():
    # (py, px); the measured strong-scaling winners within the safe py<=2 region
    assert balanced_grid(8, (256, 256, 1024)) == (2, 4)   # tile (256,128,256)
    assert balanced_grid(8, (384, 384, 384)) == (2, 4)    # tile (384,192,96)
    assert balanced_grid(8, (512, 512, 512)) == (2, 4)    # tile (512,256,128)
    assert balanced_grid(8, (256, 512, 1024)) == (2, 4)   # tile (256,256,256)


def test_two_d_is_x_cut_only():
    assert balanced_grid(8, (256, 1024)) == (1, 8)


def test_allow_y_thin_unlocks_cubic_optimum():
    # 384^3: with py>=3 allowed, py4 (ny=96) ties min-edge but has fatter x ->
    # picked (matches the px2py4 0.688 ms best in the generalization sweep).
    assert balanced_grid(8, (384, 384, 384), allow_y_thin=True) == (4, 2)


def test_falls_back_when_indivisible():
    # Ny/Nx not divisible by any balanced py<=2 factor -> x-cut fallback path
    py, px = balanced_grid(8, (256, 250, 1000))
    assert py * px == 8


def test_world_one():
    assert balanced_grid(1, (256, 256, 256)) == (1, 1)


if __name__ == "__main__":
    for fn in [test_safe_default_beats_x_cut, test_two_d_is_x_cut_only,
               test_allow_y_thin_unlocks_cubic_optimum,
               test_falls_back_when_indivisible, test_world_one]:
        fn()
        print(f"OK {fn.__name__}")
    print("ALL PASS")
