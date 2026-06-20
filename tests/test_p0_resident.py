"""P0 resident-trunk integration -- end-to-end gate suite.

P0 (the GPU-resident single-tape trunk, now the default backend) was validated
by a set of canonical scripts that each exit 0/1 on PASS/FAIL with a built-in
numpy / finite-difference reference. This suite RUNS those exact gates and
asserts each passes, so a future change that silently breaks the resident path
is caught instead of shipping.

Coverage (P0 steps 1-4):
  1  single-tape full-trunk forward+backward gradcheck vs finite-diff   (tied head)
  2  tied embedding/head gradient merge -- untied-control gradcheck
  2b resident GPU forward on the single tape -- logits parity + gradcheck
  3  persistent resident weights + resident AdamW vs numpy AdamW
  -  resident AdamW unit (adamw-update.glsl vs numpy, persistent W/m/v)
  4  L=18 / v3.3-shape capacity (records+backprops+AdamW, no OOM)   [slow]

Run:
    pytest  tests/test_p0_resident.py           # fast gates
    P0_SLOW=1 pytest tests/test_p0_resident.py  # + the heavy capacity gate
"""
import os
import sys
import subprocess

import pathlib
GRILLY = str(pathlib.Path(__file__).resolve().parents[1])
RT = GRILLY + r"\experimental\resident_train"
TRUNK_LM = RT + r"\train_trunk_lm.py"
ADAMW = RT + r"\test_resident_adamw.py"
PY = sys.executable
SLOW = bool(os.environ.get("P0_SLOW"))
SLOW_TESTS = {"test_capacity_v3_3_shape_step4"}


def _run_gate(script, args=(), cwd=None, timeout=300, label="PASS"):
    """Run a gate script and assert it exits 0 AND prints '<label>' (the gate's
    own PASS line). Returns combined output. Raises AssertionError with the tail
    of the output on failure."""
    cmd = [PY, "-u", script, *args]
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise AssertionError("%s %s TIMED OUT after %ss" % (os.path.basename(script), " ".join(args), timeout)) from e
    out = (p.stdout or "") + (p.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-15:])
    assert p.returncode == 0, ("%s %s exited %d\n--- tail ---\n%s"
                               % (os.path.basename(script), " ".join(args), p.returncode, tail))
    assert label in out, ("%s %s exited 0 but '%s' not found\n--- tail ---\n%s"
                          % (os.path.basename(script), " ".join(args), label, tail))
    return out


def _skip_unless_slow():
    if SLOW:
        return False
    try:
        import pytest
        pytest.skip("heavy gate; set P0_SLOW=1 to run")
    except ImportError:
        pass
    return True   # stdlib runner: caller returns early


# ---------------------------------------------------------------- step 1
def test_gradcheck_single_tape_tied():
    """Step 1: the whole trunk on ONE tape, backward() once, matches finite-diff."""
    _run_gate(TRUNK_LM, ["gradcheck"], cwd=GRILLY, label="GRADCHECK: PASS")


# ---------------------------------------------------------------- step 2
def test_tied_head_merge_untied_control():
    """Step 2: gradcheck still passes with an UNTIED head -- proves the tied-E
    merge (head-weight grad + embedding scatter) is correct, not masking error."""
    _run_gate(TRUNK_LM, ["gradcheck", "--untied"], cwd=GRILLY, label="GRADCHECK: PASS")


# ---------------------------------------------------------------- step 2b
def test_resident_forward_parity_and_gradcheck():
    """Step 2b: the forward runs fully on-GPU (no activation leaves VRAM); its
    logits match numpy and the resident-seeded backward still gradchecks."""
    out = _run_gate(TRUNK_LM, ["gradcheck", "--resident"], cwd=GRILLY, label="GRADCHECK: PASS")
    assert "parity" in out.lower(), "resident-forward parity line missing"


# ---------------------------------------------------------------- resident AdamW unit
def test_resident_adamw_matches_numpy():
    """adamw-update.glsl vs numpy AdamW over 25 steps with persistent W/m/v."""
    _run_gate(ADAMW, [], cwd=GRILLY, timeout=120, label="RESIDENT-ADAMW: PASS")


# ---------------------------------------------------------------- step 3
def test_persistent_weights_resident_adamw_step3():
    """Step 3: persistent resident weights + resident AdamW reproduce the numpy
    AdamW loss curve from identical init (only the optimizer impl differs)."""
    _run_gate(TRUNK_LM, ["--resident-opt"], cwd=GRILLY, label="STEP-3: PASS")


# ---------------------------------------------------------------- step 4 (slow)
def test_capacity_v3_3_shape_step4():
    """Step 4: the full v3.3 trunk (V=65k, d=1024, L=18) records + backprops +
    runs resident AdamW with no arena/grad-table overflow or VRAM OOM."""
    if _skip_unless_slow():
        return
    _run_gate(TRUNK_LM, ["--resident-opt", "--big"], cwd=GRILLY, timeout=1200, label="STEP-4: PASS")


# ---------------------------------------------------------------- stdlib runner
if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed, skipped = [], []
    print("=== P0 resident-trunk gate suite (%s) ===" % ("FULL incl. slow" if SLOW else "fast; P0_SLOW=1 for all"))
    for name, fn in fns:
        if name in SLOW_TESTS and not SLOW:
            print("SKIP  %s  (set P0_SLOW=1)" % name); skipped.append(name); continue
        sys.stdout.write("RUN   %s ... " % name); sys.stdout.flush()
        try:
            fn(); print("PASS")
        except Exception as e:
            failed.append(name)
            print("FAIL\n      %s" % str(e).replace("\n", "\n      "))
    ran = len(fns) - len(skipped)
    print("\n%d/%d passed%s" % (ran - len(failed), ran, (", %d skipped" % len(skipped)) if skipped else ""))
    sys.exit(1 if failed else 0)
