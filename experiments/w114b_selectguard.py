"""w114 (2026-08-29) — no printed line of `check_selection.py` may name a clickable file
that is not WANTED.

WHY THIS GUARD EXISTS. `check_selection.py`'s NOTHING-IS-SELECTED branch carried, for twelve
days, a green-tick box reading "✅ RESOLVED. WANTED is now {w21_ad187corr.csv,
w20_ad187_h3.csv}" and a second block annotating `w16i_schemeavg` / `blend159av_h3` as
"current WANTED". All four are sent, so Kaggle offers them and the wrong click is reachable;
`w114a_misclick.py` prices those two pairs at +35.17e-6 and +81.92e-6 against the +4.5228e-6
that NOT clicking at all costs. The instrument built to prevent a 4.5e-6 error was the
largest reachable source of a 35-82e-6 one.

THE RULE. Walk `main()` with `ast` and collect every string LITERAL it prints. No literal may
contain the stem of a file this account can select — a stem is selectable iff
`submissions/<stem>.csv` exists — unless the stem is in `check_selection.WANTED` or in
`ALLOWED` below with a written reason.

Comments and module docstrings are deliberately NOT scanned: the file's top 460 lines are a
decision record and naming a superseded file there is correct. Only what it PRINTS can
mislead a reader who is about to click. `CLICK_HISTORY` is exempt for the same reason and
because it is printed by NAME, not as a literal, so the walk never sees it.

    .venv/bin/python experiments/w114b_selectguard.py                # rc 0 = clean
    .venv/bin/python experiments/w114b_selectguard.py --control      # must exit 0, having fired

Offline and deterministic: no API call, no leaderboard, no fold. Standing check #48.
"""
from __future__ import annotations

import ast, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUB = os.path.join(ROOT, "submissions")
TARGET_SRC = os.path.join(HERE, "check_selection.py")

# Stems a printed literal may name even though they are selectable, each with the reason.
ALLOWED = {
    "blend158_logit": "the named auto-selection hazard (w15j): the branch prints it to WARN "
                      "that it is in a tie, never to propose it.",
    "stack_pub74_logit": "named in the PAGINATION warning as evidence of a past truncation "
                         "bug, not as a candidate; public 0.97081, 37e-6 below tier 2.",
    "stack_pub88_mine_logit": "as stack_pub74_logit — the other file the page_size 50 default "
                              "hid from every API-reading script here (w17, 08-17).",
}


def selectable_stems() -> set:
    """Every stem this account could actually be talked into clicking."""
    return {f[:-4] for f in os.listdir(SUB)
            if f.endswith(".csv") and not f.startswith("oof_")}


def printed_literals(path: str) -> list:
    """(lineno, text) for every string literal `main()` passes to print(), f-strings too."""
    tree = ast.parse(open(path).read())
    main_fn = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "main")
    out = []
    for node in ast.walk(main_fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            continue
        for arg in node.args:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    out.append((getattr(sub, "lineno", node.lineno), sub.value))
    return out


def scan(path: str) -> list:
    sys.path.insert(0, HERE)
    import check_selection as cs
    wanted = {w[:-4] if w.endswith(".csv") else w for w in cs.WANTED}
    stems = selectable_stems() - wanted - set(ALLOWED)
    # longest first, so `w36_ad199stdcorr_ens4` is matched before `w36_ad199stdcorr`
    pat = re.compile("(?:" + "|".join(re.escape(s) for s in sorted(stems, key=len, reverse=True))
                     + r")(?![A-Za-z0-9_])")
    bad = []
    for lineno, text in printed_literals(path):
        for m in pat.finditer(text):
            bad.append((lineno, m.group(0), text.strip()[:90]))
    return bad


def main() -> int:
    if "--control" in sys.argv:
        # CONTROL-: plant the exact defect w114 removed and require the guard to catch it.
        # The copy is written NEXT TO the original: HERE/ROOT derive from __file__ and a copy
        # run from /tmp would test the relocation, not the rule (w113 §5).
        ctl = os.path.join(HERE, "_w114b_control_check_selection.py")
        src = open(TARGET_SRC).read()
        plant = ('        print("  WANTED is now {w21_ad187corr.csv, w20_ad187_h3.csv}")\n')
        needle = '        print(f"\\n*** NOTHING IS SELECTED for {COMP}. ***")\n'
        if needle not in src:
            print("*** CONTROL FAILED — the plant site is gone; the guard was not exercised.")
            return 1
        open(ctl, "w").write(src.replace(needle, needle + plant, 1))
        try:
            # ⚠ AGAINST THE BASELINE, NOT AGAINST ZERO. The first cut asserted len(bad) >= 2
            # on the perturbed copy, and the clean file already had three ALLOWED-pending
            # hits — so the control passed while proving nothing about the plant. A control
            # that counts must count the DELTA it planted.
            base = {(stem, text) for _, stem, text in scan(TARGET_SRC)}
            bad = scan(ctl)
            new = {(stem, text) for _, stem, text in bad} - base
            print("CONTROL- planted 'WANTED is now {w21_ad187corr.csv, w20_ad187_h3.csv}'")
            print(f"  baseline {len(base)} hit(s) on the clean file, "
                  f"{len(bad)} on the perturbed copy, {len(new)} NEW")
            for stem, text in sorted(new):
                print(f"  caught: {stem}   in  {text!r}")
            want = {"w21_ad187corr", "w20_ad187_h3"}
            if {stem for stem, _ in new} != want:
                print(f"\n*** CONTROL FAILED — new hits {sorted(s for s, _ in new)} != "
                      f"{sorted(want)}. The guard does not see the defect it exists to catch.")
                return 1
            print("\n✅ CONTROL- FIRED on exactly the two planted stems, and on nothing else.")
            return 0
        finally:
            os.remove(ctl)

    stems = selectable_stems()
    bad = scan(TARGET_SRC)
    print(f"w114b — {len(stems)} selectable stems on disk, "
          f"{len(printed_literals(TARGET_SRC))} printed literals in main()")
    if bad:
        print("\n⛔ A PRINTED LINE NAMES A CLICKABLE FILE THAT IS NOT WANTED:")
        for lineno, stem, text in bad:
            print(f"  line {lineno}: {stem}\n      {text!r}")
        print("\n  Move it into CLICK_HISTORY (printed by name, behind --history), or add it")
        print("  to ALLOWED with a reason. Do not weaken this check to make a run pass.")
        return 1
    print("✅ CLEAN — every printed literal names only WANTED or an ALLOWED stem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
