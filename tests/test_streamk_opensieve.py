from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "generate_streamk_opensieve.py"
GENERATED = ROOT / "csrc" / "quantization" / "marlin" / "generated"


def test_opensieve_generator_is_deterministic():
    subprocess.check_call([sys.executable, str(GENERATOR)], cwd=ROOT)
    lut = (GENERATED / "marlin_streamk_lut.inc").read_text(encoding="utf-8")
    bloom = (GENERATED / "marlin_streamk_bloom.inc").read_text(encoding="utf-8")
    assert "kStreamKLutSize = 120" in lut
    assert "kStreamKBloomPolicyCount = 7" in bloom

    subprocess.check_call([sys.executable, str(GENERATOR)], cwd=ROOT)
    assert lut == (GENERATED / "marlin_streamk_lut.inc").read_text(encoding="utf-8")
    assert bloom == (GENERATED / "marlin_streamk_bloom.inc").read_text(encoding="utf-8")


def test_opensieve_bucket_index_matches_python():
    sys.path.insert(0, str(ROOT / "tools"))
    import generate_streamk_opensieve as gen

    cases = [
        (1, 8, 132),
        (64, 32, 264),
        (512, 128, 528),
        (2048, 256, 132),
    ]
    for mn, k, grid in cases:
        py_idx = gen.lut_index(mn, k, grid)
        assert 0 <= py_idx < gen.lut_size()
        py_key = gen.bucket_key(mn, k, grid)
        mn_b = gen.bucket_index(max(mn, 1), gen.MN_BOUNDS)
        k_b = gen.bucket_index(max(k, 1), gen.K_BOUNDS)
        g_b = gen.bucket_index(max(grid, 1), gen.GRID_BOUNDS)
        assert py_key == (mn_b << 16) | (k_b << 8) | g_b
