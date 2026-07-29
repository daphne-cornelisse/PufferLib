import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Generated from KempnerInstitute/wef ElectricScene using the deterministic
# two-fish/one-food fixture implemented by
# `./electric_fish --physics-fixture`.
UPSTREAM_FIXTURE = {
    "eod": [2.156647270767495e-05, 2.2287084167914553e-05],
    "intrinsic": [1.303699593123667e-11, 2.044228319704432e-11],
    "induced": [-5.345843529811098e-09, -6.1168561334662774e-09],
    "induced_fish": [-1.9715927047534895e-22, -2.4709593293090667e-22],
    "induced_food": [-1.0289523437915805e-22, -6.632397743951107e-23],
    "morm_cd0": 0.19028434017121887,
    "amp_baseline0": 1.9838499805530827e-07,
}


def test_electric_fish_scene_matches_upstream_fixture():
    executable = ROOT / "electric_fish"
    if not executable.is_file():
        subprocess.run(
            [str(ROOT / "build.sh"), "electric_fish", "--fast"],
            cwd=ROOT,
            check=True,
        )

    output = subprocess.check_output(
        [str(executable), "--physics-fixture"],
        cwd=ROOT,
        text=True,
    )
    actual = json.loads(output)

    for key, expected in UPSTREAM_FIXTURE.items():
        got = actual[key]
        if isinstance(expected, list):
            assert len(got) == len(expected)
            for got_value, expected_value in zip(got, expected):
                assert math.isclose(
                    got_value, expected_value, rel_tol=2e-13, abs_tol=1e-30
                ), key
        else:
            assert math.isclose(
                got, expected, rel_tol=2e-13, abs_tol=1e-30
            ), key
