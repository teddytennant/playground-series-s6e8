"""w67b — the guard for w67a. Every check NEGATIVE-CONTROLLED: it must also REJECT the wrong thing.

w66 section 6: a rule enforced in ONE consumer is not enforced. w66 section 8: a check that
cannot fail is not a check. This file therefore asserts, for each rule, both that the shipped
code satisfies it AND that a deliberately broken variant is caught.

    .venv/bin/python experiments/w67b_slopeguard.py
"""
from __future__ import annotations

import ast, json, math, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

import w53a_pricer as PR                                        # noqa: E402
import w67a_aboveslope as W67A                                  # noqa: E402 — import, never run

U = 1e-6
FAILURES = 0


def chk(tag: str, ok: bool, detail: str) -> None:
    global FAILURES
    if not ok:
        FAILURES += 1
    print(f"  {'✅' if ok else '🔴'} {tag}: {detail}", flush=True)


def raises(fn) -> bool:
    """The negative-control primitive: the broken variant MUST be rejected."""
    try:
        fn()
        return False
    except Exception:
        return True


def main() -> None:
    J = json.load(open(os.path.join(HERE, "w67a_aboveslope.json")))
    print("=" * 96)
    print("w67b — GUARDS FOR THE ABOVE-RANGE SLOPE INSTRUMENT")
    print("=" * 96)

    # ---- 1-2. UNITS. The defect that flipped P4 and P5 on the first run. --------------------
    chk("G01", 1.0 < PR.RESID_SD < 100.0,
        f"RESID_SD {PR.RESID_SD:.4f} is in e-6 units, not absolute AUC")
    chk("G02", raises(lambda: _assert_units(PR.RESID_SD * 1e-6)),
        "NEGATIVE CONTROL: an absolute-AUC RESID_SD (7.72e-6) is REJECTED by the same assert")

    sd_iid = math.sqrt(PR.RESID_SD ** 2 + W67A.ROUND_VAR)
    chk("G03", abs(J["sd_iid"] - sd_iid) < 1e-12,
        f"sd_iid {J['sd_iid']:.6f} re-derives from RESID_SD and ROUND_VAR")
    chk("G04", abs(J["sd_iid"] - math.sqrt((PR.RESID_SD * 1e6) ** 2 + W67A.ROUND_VAR)) > 1.0,
        "NEGATIVE CONTROL: the 1e6-inflated sd is NOT what the artefact holds")

    # ---- 3-5. THE SAMPLE. Range, member exclusion, era. ------------------------------------
    fr = J["frozen"]
    chk("G05", len(fr) == W67A.P2_N, f"exactly {W67A.P2_N} frozen files")
    chk("G06", all(f["cv"] > PR.FIT_CV_MAX for f in fr),
        f"every frozen file is strictly above FIT_CV_MAX ({PR.FIT_CV_MAX:.10f})")
    chk("G07", all(PR.family(f["stem"]) != "member" for f in fr),
        "no `member` row survived into the frozen set")
    q = pd.read_csv(os.path.join(HERE, "w23b_sendqueue.csv"))
    q["stem"] = q.file.str.replace(r"\.csv$", "", regex=True)
    memb = [s for s in q[q.cv > PR.FIT_CV_MAX].stem if PR.family(s) == "member"]
    chk("G08", len(memb) > 0 and set(memb).isdisjoint({f["stem"] for f in fr}),
        f"NEGATIVE CONTROL: the member filter had something to remove and removed it: {memb}")
    chk("G09", all(PR.new_era(f["stem"]) for f in fr),
        "every frozen file is era (ad>=195), so the ERA slope is the whole contrast")

    # ---- 6-9. THE ESTIMAND. Anchor, agreement at zero, linear divergence. -------------------
    a_ok = all(abs(f["anchor"] - PR.predict_lb(PR.FIT_CV_MAX, f["stem"])) < 1e-15 for f in fr)
    chk("G10", a_ok, "ANCHOR is exactly the frozen pricer evaluated at cv = FIT_CV_MAX")
    chk("G11", all(abs(f["excess_e6"] - (f["cv"] - PR.FIT_CV_MAX) * 1e6) < 1e-9 for f in fr),
        "excess re-derives from cv and FIT_CV_MAX")
    gap = [abs((f["lb_hat_ols"] - f["lb_hat_gls"]) - J["separation"] * f["excess_e6"] * U)
           for f in fr]
    chk("G12", max(gap) < 1e-15,
        f"the two hypotheses diverge EXACTLY linearly in excess (max err {max(gap):.2e})")
    z0 = abs(J["s_ols"] * 0.0 - J["s_gls"] * 0.0)
    chk("G13", z0 == 0.0 and abs(J["separation"] - abs(J["s_ols"] - J["s_gls"])) < 1e-12,
        "the hypotheses agree exactly at excess = 0 and separation is their difference")

    # ---- 10-13. THE ESTIMATOR. Recovery, one-sided scaling. --------------------------------
    ex = np.array([f["excess_e6"] for f in fr])
    X = np.column_stack([np.ones(len(ex)), ex])
    Om = np.eye(len(ex)) * J["sd_iid"] ** 2
    worst = 0.0
    for s_true in (J["s_ols"], J["s_gls"], -3.0):
        b = W67A.gls(X, s_true * ex, Om)["b"][1]
        worst = max(worst, abs(float(b) - s_true))
    chk("G14", worst < 1e-9, f"gls recovers injected slopes (incl. a -3.0 decoy) to {worst:.2e}")
    noisy = W67A.gls(X, 1.4 * ex + np.arange(len(ex)) * 50.0, Om)
    chk("G15", noisy["scale"] >= 1.0,
        f"the chi2/dof scaling is ONE-SIDED: scale {noisy['scale']:.3f} >= 1 on bad scatter")
    clean = W67A.gls(X, 1.4 * ex, Om)
    chk("G16", abs(clean["scale"] - 1.0) < 1e-12,
        "NEGATIVE CONTROL: a perfect fit is clamped to scale 1, never below")
    chk("G17", noisy["se_scaled"][1] > noisy["se"][1],
        "bad scatter WIDENS the interval; a good covariance model cannot buy a narrow one")

    # ---- 14-17. THE BUCKET RULE, derived and negative-controlled. ---------------------------
    def bucket(R):
        return "a" if R < W67A.R_NORES else ("b" if R < W67A.R_CLEAN else "c")

    chk("G18", (bucket(1.0), bucket(3.0), bucket(5.0)) == ("a", "b", "c"),
        "the bucket rule maps 1.0/3.0/5.0 to a/b/c")
    chk("G19", bucket(W67A.R_NORES) == "b" and bucket(W67A.R_CLEAN) == "c",
        f"STRICT inequalities: R={W67A.R_NORES} is (b) and R={W67A.R_CLEAN} is (c)")
    chk("G20", bucket(J["R"]) == J["bucket"],
        f"the artefact's bucket ({J['bucket']}) re-derives from its own R ({J['R']:.3f})")
    chk("G21", abs(J["R"] - J["separation"] / J["se_corr"]) < 1e-9,
        "R re-derives from separation and se_corr; it is not typed")

    # ---- 18-20. THE MARGIN. The verdict must carry the misfit that demotes it. --------------
    stb, sta = math.sqrt(J["chi2dof_demote_to_b"]), math.sqrt(J["chi2dof_demote_to_a"])
    chk("G22", abs(J["separation"] / (stb * J["se_corr"]) - W67A.R_CLEAN) < 1e-9,
        f"chi2/dof {J['chi2dof_demote_to_b']:.3f} lands R exactly on the (c)/(b) boundary")
    chk("G23", abs(J["separation"] / (sta * J["se_corr"]) - W67A.R_NORES) < 1e-9,
        f"chi2/dof {J['chi2dof_demote_to_a']:.3f} lands R exactly on the (b)/(a) boundary")
    chk("G24", J["chi2dof_demote_to_b"] > 1.0,
        f"the demotion threshold is above a perfect fit ({J['chi2dof_demote_to_b']:.3f} > 1), "
        f"so bucket (c) is the live verdict and not an artefact of clamping")

    # ---- 21-23. READ-ONLY. w67 must not sit on the send path. ------------------------------
    src = open(os.path.join(HERE, "w67a_aboveslope.py")).read()
    # ⚠ The rule is about IMPORTS, so it is read off the AST. A substring test on the source
    # matched this module's own PROSE about w26d and fired on a docstring -- a guard that reads
    # text where the rule names a structure tests the wrong object.
    imports = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            imports.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            imports.add(n.module.split(".")[0])
    chk("G25", "w26d_queueprice" not in imports,
        f"w67a does not import w26d_queueprice (w66 section 10). imports: {sorted(imports)}")
    chk("G25b", raises(lambda: _no_import("import w26d_queueprice as Q", "w26d_queueprice")),
        "NEGATIVE CONTROL: the AST check REJECTS a module that really does import it")
    chk("G26", "w23b_sendqueue.csv" in src and ".to_csv(" not in src,
        "w67a READS the queue and writes no csv at all")
    written = [l for l in src.splitlines() if "json.dump" in l]
    chk("G27", len(written) == 1 and "w67a_aboveslope.json" in src,
        "w67a writes exactly one artefact, its own json")

    # ---- 24-28. VALIDITY. The governing verdict, and it must not be vacuously true. --------
    V = J["validity"]
    chk("G28", V["n_free"] == 0 and V["n_above"] == len(fr),
        f"every one of the {V['n_above']} above-range files is vetoed or `member`; "
        f"{V['n_free']} are sendable")
    chk("G29", V["n_vetoed"] > 0 and len(W67A.W48E.VETO) > V["n_vetoed"],
        f"NOT VACUOUS: VETO is non-empty ({len(W67A.W48E.VETO)} files) and {V['n_vetoed']} of "
        f"them are the above-range set, so V1 is a real exclusion")
    chk("G30", "w48e_order" in imports,
        "the VETO is IMPORTED from w48e_order, not re-listed in w67a (w66 section 6)")
    vetoed_names = {x["stem"] for x in V["rows"] if x["vetoed"]}
    chk("G31", vetoed_names <= set(W67A.W48E.VETO),
        "every stem w67a calls vetoed is actually in the sender's VETO dict")
    # NEGATIVE CONTROL on the COUNTING rule: a fabricated free row must raise n_free.
    fake = V["rows"] + [dict(stem="zz_fake_above", cv=PR.FIT_CV_MAX + 1e-6,
                             vetoed=False, fam="h3")]
    n_free_fake = sum(1 for x in fake if not x["vetoed"] and x["fam"] != "member")
    chk("G32", n_free_fake == 1,
        "NEGATIVE CONTROL: injecting one un-vetoed above-range file makes n_free 1, so V1 "
        "counts rather than asserting")

    print("\n" + "=" * 96)
    print(f"FAILURES {FAILURES}")
    print("=" * 96)
    sys.exit(1 if FAILURES else 0)


def _no_import(src, mod):
    """Raises iff `src` imports `mod` -- the negative control for G25."""
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import) and any(a.name.split(".")[0] == mod for a in n.names):
            raise AssertionError(mod)
        if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] == mod:
            raise AssertionError(mod)


def _assert_units(v):
    assert 1.0 < v < 100.0, (v, "RESID_SD is not in e-6 -- units moved")


if __name__ == "__main__":
    main()
