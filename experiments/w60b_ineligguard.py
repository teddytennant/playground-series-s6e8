"""w60 (2026-08-22) — the WANTED-ineligibility registry must be COMPLETE, not merely present.

WHY THIS EXISTS, AND WHY w56b DID NOT COVER IT
-----------------------------------------------
`w56b_wantedguard.py` asks three good questions — is the guard there, does live WANTED pass it,
does it FIRE — and it passed every day since 2026-08-22 while `WANTED_INELIGIBLE` was MISSING
`w40_ad211`, the arm w40d_prereg.txt was written about and the source of the very sentence the
dict exists to enforce. w56b could not have caught it: its case list is hand-typed, and it was
typed from the two arms w56 was looking at. A guard whose cases come from the same place as the
bug cannot see the bug.

⚠ THE LESSON IS NARROWER THAN w54/w55/w56's "a rule in a paragraph is not a rule", and nastier:
**when you move a rule into an enforcing structure, enumerate its cases FROM THE RULE, not from
the instances in front of you.** A partly-populated enforcer reads exactly like a complete one.

SO THIS GUARD DERIVES THE CASES FROM THE PREREGISTRATIONS, IN BOTH DIRECTIONS
-----------------------------------------------------------------------------
  FORWARD   every prereg on disk that DECLARES an arm WANTED-ineligible must have a live key.
            The declaring files are enumerated in MAPPING below, with the clause each one is
            required to still contain — so deleting the clause is a failure, not a pass.
  SWEEP     every `*prereg*.txt` in experiments/ is scanned for the declaration pattern, and a
            declaring file that MAPPING does not know about is a FAILURE. This is the half that
            catches the NEXT omission: a future arm registered as ineligible and never keyed.
  REVERSE   every key in WANTED_INELIGIBLE must be backed by a declaring prereg on disk, so a
            key cannot be invented (or silently retired) without the registration moving too.
  EXERCISE  `assert_wanted_eligible` and `w26g_send.above_tier_reason` are both made to FIRE on
            a real STILL-BARRED file by name, and to ADMIT a real retired one. w54's lesson: a
            guard nobody has seen fire is decorative — and w61's: nor is one nobody has seen
            stop firing on the arm it was retired for.

EXTENDED w61 (2026-08-22) — RETIREMENT IS A THIRD STATE, NOT AN ABSENCE
-----------------------------------------------------------------------
w61 ran the four-base matched-control test w60_prereg registered and w60 deliberately left
unrun, and it PASSED, so `w40_ad211` moved out of `WANTED_INELIGIBLE`. Under the guard as w60
wrote it that is indistinguishable from someone deleting the key to make a run pass: FORWARD
would fail (w40d still declares the arm) and REVERSE would never look. So a declared arm may
now be in EITHER dict — but a RETIRED key has to carry more than a live one does: the retiring
artefact must exist on disk, its own verdict must say RETIRE, and the numbers in the dict value
must be the numbers in the artefact. ⛔ The one thing still forbidden is what it always was: a
declared arm in NEITHER dict.

Exit 0 = registry complete and both enforcement paths bite. Exit 1 = a registered rule is not
being enforced somewhere. Run on any run that touches `check_selection.py`, `w26g_send.py`, or
adds a prereg with an ineligibility clause.

    .venv/bin/python experiments/w60b_ineligguard.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TARGET = os.path.join(HERE, "check_selection.py")

# Every preregistration that DECLARES an arm ineligible for CV-based selection -> the
# WANTED_INELIGIBLE key(s) that must exist because of it, and a substring of the clause that
# must still be in the file. The clause is quoted so that softening the prereg fails here too:
# retiring an entry is allowed (the dict says how), quietly deleting the rule is not.
MAPPING = {
    "w40d_prereg.txt": (
        {"w40_ad211"},
        "ARM 211 IS NOT ELIGIBLE FOR check_selection.WANTED, WHATEVER ITS CV.",
    ),
    # w42b's clause is written about the ext_members16 POOL, so it covers BOTH arms built on
    # it — ARM 217 and ARM 216 (which is ARM 217 minus one member; w50_prereg §5 says removing a
    # member does not discharge the clause).
    "w42b_prereg.txt": (
        {"w42_ad217", "w50_ad216"},
        "NO ARM BUILT ON ext_members16 IS ELIGIBLE FOR check_selection.WANTED ON CV ALONE.",
    ),
    # w51 is the ARM 216 es-read and quotes w42b's clause as the thing it is trying to
    # discharge. Found by the SWEEP below, not by hand — which is the sweep's whole point.
    "w51_prereg.txt": (
        {"w42_ad217", "w50_ad216"},
        "NO ARM BUILT ON ext_members16 IS ELIGIBLE FOR check_selection.WANTED ON CV ALONE.",
    ),
    "w50_prereg.txt": (
        {"w50_ad216"},
        "NOT WANTED-eligible, whatever d_five turns out to be.",
    ),
    # w60 restates w40d's clause verbatim in the section that ADDS the key. It maps to the same
    # arm; it is listed so the SWEEP below does not read the restatement as an unkeyed rule.
    "w60_prereg.txt": (
        {"w40_ad211"},
        "ARM 211 IS NOT ELIGIBLE FOR check_selection.WANTED, WHATEVER ITS CV.",
    ),
}

# A DECLARATION, for the sweep. BOTH halves are required on the same line:
#   * the standalone caps word ELIGIBLE — `WANTED_INELIGIBLE` does not match it (no space
#     before ELIGIBLE), so prose that merely *mentions* the dict is not read as a new rule;
#   * the literal `check_selection.WANTED`, so the line has to be about THE SELECTION SET.
# ⚠ The first cut used only the caps word and flagged w59_prereg's "P5/P8 measure ELIGIBLE
# files with cross-fitted CVs" as a registration. A sweep that cries wolf gets its MAPPING
# padded to silence it, which is how a completeness check turns back into a hand-typed list.
DECLARE = re.compile(r"(?<= )ELIGIBLE(?= )")
SCOPE = "check_selection.WANTED"

# w61: what a RETIRED key must carry before this guard will accept it in place of a live one.
# key -> (artefact json, the substring its "verdict" must equal, numbers that must appear
# verbatim in the dict value). The numbers are re-read from the ARTEFACT, not typed here, so
# the value cannot drift away from the measurement that retired the bar. ⚠ Typing them here
# would reproduce exactly the w56 failure this file exists for: a check whose expected values
# come from the same keystrokes as the thing checked.
RETIREMENT_EVIDENCE = {
    "w40_ad211": ("w61a_armctl.json", "RETIRE"),
}


def _fail(msg):
    print(f"  ⛔ {msg}")
    return 1


def _check_retirement(key, value):
    """w61: a RETIRED key must be backed by an artefact that says so, and quote its numbers.

    Retiring on evidence and deleting to pass a run look identical from the dict alone, so the
    evidence is checked here rather than trusted: the artefact exists, ITS OWN verdict is
    RETIRE, its failure count is zero, and every per-base delta it measured appears verbatim in
    the dict value. That last clause is what stops the value drifting into a summary of a
    measurement nobody can reproduce.
    """
    if key not in RETIREMENT_EVIDENCE:
        return _fail(f"{key} is RETIRED but RETIREMENT_EVIDENCE does not say what backs it. "
                     f"A retirement with no named artefact is a deletion.")
    art, want_verdict = RETIREMENT_EVIDENCE[key]
    path = os.path.join(HERE, art)
    if not os.path.exists(path):
        return _fail(f"{key} is RETIRED on {art}, which is not on disk.")
    with open(path) as fh:
        j = json.load(fh)
    bad = 0
    if j.get("verdict") != want_verdict:
        bad += _fail(f"{art} verdict is {j.get('verdict')!r}, not {want_verdict!r} — the key "
                     f"was retired against its own evidence.")
    if j.get("failures"):
        bad += _fail(f"{art} reports {len(j['failures'])} failure(s); a retirement may not "
                     f"rest on a run that did not pass its own gates.")
    for r in j.get("rows", []):
        if not r.get("in_criterion"):
            continue
        # the delta as the artefact reports it, to the precision the dict value quotes
        txt = f"{r['base']} {r['delta_e6']:+.3f}"
        if txt.replace("+", "") not in value.replace("+", ""):
            bad += _fail(f"{key}: the dict value does not quote {art}'s {r['base']} delta "
                         f"({r['delta_e6']:+.3f}e-6). w60_prereg requires the READING in the "
                         f"value, not a paraphrase of it.")
    return bad


def main() -> int:
    fails = 0

    spec = importlib.util.spec_from_file_location("_check_selection_w60", TARGET)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit as exc:
        print(f"⛔ importing check_selection.py raised SystemExit({exc.code}) — live WANTED "
              f"names a barred arm. Fix WANTED, not this guard.")
        return 1
    reg = mod.WANTED_INELIGIBLE
    ret = getattr(mod, "WANTED_RETIRED", {})
    print(f"registry: {len(reg)} live — {', '.join(sorted(reg)) or '(none)'}")
    print(f"          {len(ret)} retired — {', '.join(sorted(ret)) or '(none)'}\n")
    if set(reg) & set(ret):
        fails += _fail(f"key(s) in BOTH dicts: {sorted(set(reg) & set(ret))} — a bar cannot be "
                       f"live and retired at once, and `above_tier_reason` only reads the live "
                       f"one, so the retired copy would be a comment pretending to be a rule.")

    # ---- FORWARD -----------------------------------------------------------------------
    print("FORWARD — every declaring prereg has a live key, and still contains its clause:")
    declared = set()
    for fn, (keys, clause) in sorted(MAPPING.items()):
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            fails += _fail(f"{fn} is GONE. A registration cannot be enforced if it is deleted.")
            continue
        src = open(path).read()
        if clause not in src:
            fails += _fail(f"{fn} no longer contains its clause {clause!r:.70} — the rule was "
                           f"softened in the prereg rather than retired in the dict.")
            continue
        for k in keys:
            declared.add(k)
            if k in reg:
                print(f"  ✅ {fn:<20} -> {k} (live)")
            elif k in ret:
                # w61: retirement is allowed, but it has to be BACKED. A key that merely moved
                # dicts is a deletion with better manners.
                fails += _check_retirement(k, ret[k])
                print(f"  ✅ {fn:<20} -> {k} (RETIRED, evidence checked)")
            else:
                fails += _fail(f"{fn} declares {k} ineligible and neither WANTED_INELIGIBLE nor "
                               f"WANTED_RETIRED has such a key. THIS IS THE w60 BUG. Add the "
                               f"key, quoting the clause — or retire it with evidence.")

    # ---- SWEEP -------------------------------------------------------------------------
    print("\nSWEEP — no prereg declares an arm ineligible without MAPPING knowing about it:")
    unknown = []
    for path in sorted(glob.glob(os.path.join(HERE, "*prereg*.txt"))):
        fn = os.path.basename(path)
        hits = [ln.strip() for ln in open(path)
                if DECLARE.search(ln) and SCOPE in ln]
        if not hits:
            continue
        if fn in MAPPING:
            print(f"  ✅ {fn:<20} {len(hits)} declaration line(s), mapped")
        else:
            unknown.append((fn, hits))
    for fn, hits in unknown:
        fails += _fail(f"{fn} contains an ineligibility DECLARATION that MAPPING does not "
                       f"cover — this is the next omission, caught:")
        for h in hits[:3]:
            print(f"       {h[:110]}")
    if not unknown:
        print(f"  ✅ swept {len(glob.glob(os.path.join(HERE, '*prereg*.txt')))} prereg files, "
              f"no unmapped declaration")

    # ---- REVERSE -----------------------------------------------------------------------
    print("\nREVERSE — every key, live or retired, is backed by a registration on disk:")
    for k in sorted(set(reg) | set(ret)):
        where = "live" if k in reg else "retired"
        if k in declared:
            print(f"  ✅ {k:<12} backed ({where})")
        else:
            fails += _fail(f"{k} is in WANTED_{'INELIGIBLE' if k in reg else 'RETIRED'} but no "
                           f"prereg in MAPPING declares it. Either the registration moved or "
                           f"the key was invented.")

    # ---- EXERCISE ----------------------------------------------------------------------
    # A guard nobody has seen fire is decorative (w54). Both enforcement paths, by name.
    # ⚠ RE-POINTED w61, NOT SOFTENED — and the reason lives here, at the check. This block used
    # to fire on `w40_ad211stdcorr.csv`. w61 ran w60_prereg's registered four-base matched-
    # control test, it PASSED (worst |delta| 2.353e-6 against the ±4e-6 floor), and the ad211
    # bar was retired on that evidence — so an assertion that ad211 still fires would now be
    # asserting the retirement did not happen. The PROPERTY under test is unchanged: a live bar
    # must refuse a WANTED naming it. Only the subject moved, to `w42_ad217`, whose bar rests on
    # a SOURCE READ (w56a: an aggregator over 138 un-es-clearable streams) that no matched-
    # control test can discharge. ⛔ Do not point it back at ad211 to make this pass.
    print("\nEXERCISE — both enforcement paths bite on a real STILL-BARRED file:")
    fired = False
    try:
        mod.assert_wanted_eligible({"w42_ad217stdcorr.csv", "w36_ad199stdcorr.csv"})
    except SystemExit as exc:
        fired = exc.code == 3
    if fired:
        print("  ✅ assert_wanted_eligible fires on w42_ad217stdcorr.csv (exit 3)")
    else:
        fails += _fail("assert_wanted_eligible did NOT fire on w42_ad217stdcorr.csv.")

    # ...and the OTHER half of the same coin: the retired arm must now be let through, and
    # NOISILY. A retirement that leaves the caller unable to tell it happened is a deletion.
    import io as _io, contextlib as _cl
    buf = _io.StringIO()
    try:
        with _cl.redirect_stdout(buf):
            mod.assert_wanted_eligible({"w40_ad211stdcorr.csv"})
        admitted = True
    except SystemExit:
        admitted = False
    note = buf.getvalue()
    if admitted and "RETIRED" in note and "w40_ad211stdcorr.csv" in note:
        print("  ✅ w40_ad211stdcorr.csv is ADMITTED, with the retirement NOTICE printed")
    else:
        fails += _fail(f"the retired arm is not admitted-with-notice (admitted={admitted}, "
                       f"notice={note.strip()[:80]!r}). Either the bar did not lift or it "
                       f"lifted silently; both are wrong.")

    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, "agent"))
    import w26g_send as S  # noqa: E402

    class Row:  # the duck-type `above_tier_reason` reads: .fam, .file, .cv
        def __init__(self, file, fam, cv):
            self.file, self.fam, self.cv = file, fam, cv

    bar = 0.9701294160          # w59a's conditional bar; ad217stdcorr is 49.4e-6 ABOVE it
    # RE-POINTED w61 for the reason written above: ad211's bar is retired, ad217's is not, and
    # ad217 is the harder case anyway — its CV clears the bar by more than any file on disk, so
    # if eligibility is ever tested AFTER the CV bar rather than before it, this is what breaks.
    r = Row("w42_ad217stdcorr.csv", "h3", 0.9701788311219954)
    why = S.above_tier_reason(r, bar)
    if why and "ineligible" in why:
        print(f"  ✅ above_tier_reason blocks w42_ad217stdcorr ABOVE the CV bar: {why[:72]}...")
    else:
        fails += _fail(f"above_tier_reason waved w42_ad217stdcorr through above the tier "
                       f"(returned {why!r}). The ad211 form of this was live for two days.")

    # w61: and the retired arm is no longer blocked ON ELIGIBILITY. It may still be blocked on
    # CV — that is a different gate and this check must not accidentally assert it is not.
    r211 = Row("w40_ad211stdcorr.csv", "h3", 0.9701374732624002)
    why211 = S.above_tier_reason(r211, bar)
    if why211 is None:
        print("  ✅ above_tier_reason now ADMITS w40_ad211stdcorr (retired, and it clears the bar)")
    elif "ineligible" in why211:
        fails += _fail(f"w40_ad211stdcorr is still blocked ON ELIGIBILITY after the bar was "
                       f"retired: {why211!r}. The dict moved but the sender did not follow.")
    else:
        print(f"  ✅ w40_ad211stdcorr no longer blocked on eligibility (now on: {why211[:52]}...)")

    # ...and the control: an ELIGIBLE arm over the bar must still be allowed through, or the
    # fix has broken the lever w59 §7 is trying to arm rather than protecting it.
    ok = Row("w36_ad199stdcorr_ens4.csv", "ens4", 0.9701366)
    why_ok = S.above_tier_reason(ok, bar)
    if why_ok is None:
        print("  ✅ and w36_ad199stdcorr_ens4 (eligible, over the bar) is STILL allowed above it")
    else:
        fails += _fail(f"the fix over-blocks: an eligible ad199 arm was refused ({why_ok!r})")

    # ---- NEGATIVE CONTROL --------------------------------------------------------------
    # w61. `_check_retirement` is the ONLY thing separating "retired on evidence" from "deleted
    # to make a run pass", and it has just returned 0 for the one real key — which is exactly
    # what a function that always returns 0 would do. So make it FAIL, three ways, on copies:
    # a value that quotes a number the artefact does not measure, an artefact whose own verdict
    # is not RETIRE, and a key with no named evidence at all.
    print("\nNEGATIVE CONTROL — the retirement check is made to REFUSE:")
    import contextlib as _cl2, io as _io2, json as _json2
    real_key, real_val = next(iter(ret.items())) if ret else (None, None)
    if real_key is None:
        print("  (no retired keys — nothing to control)")
    else:
        art, _ = RETIREMENT_EVIDENCE[real_key]
        real_art = _json2.load(open(os.path.join(HERE, art)))
        cases = [
            ("a value quoting a delta the artefact does not measure",
             lambda: _check_retirement(real_key, real_val.replace("rankraw -2.353",
                                                                 "rankraw -0.001"))),
            ("a key with no entry in RETIREMENT_EVIDENCE",
             lambda: _check_retirement("w99_notreal", real_val)),
        ]
        for label, fn in cases:
            buf2 = _io2.StringIO()
            with _cl2.redirect_stdout(buf2):
                got = fn()
            if got:
                print(f"  ✅ refuses {label}")
            else:
                fails += _fail(f"_check_retirement ACCEPTED {label} — it is decorative.")
        # ...and the verdict clause, on a copy of the artefact written to a scratch name so the
        # real one is never touched.
        tmp_name = "w61a_armctl.NEGCTL.json"
        tmp = os.path.join(HERE, tmp_name)
        bad_art = dict(real_art, verdict="KEEP")
        try:
            with open(tmp, "w") as fh:
                _json2.dump(bad_art, fh)
            RETIREMENT_EVIDENCE["w99_negctl"] = (tmp_name, "RETIRE")
            buf2 = _io2.StringIO()
            with _cl2.redirect_stdout(buf2):
                got = _check_retirement("w99_negctl", real_val)
            if got:
                print("  ✅ refuses an artefact whose own verdict is KEEP")
            else:
                fails += _fail("_check_retirement ACCEPTED an artefact whose verdict is KEEP.")
        finally:
            RETIREMENT_EVIDENCE.pop("w99_negctl", None)
            if os.path.exists(tmp):
                os.remove(tmp)

    print(f"\nFAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
