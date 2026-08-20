"""w37c — PRE-REGISTER the es-bias calibration sends, and wire them into the queue.

Written and committed BEFORE any of the seven files is submitted. Every prediction below is
computed from the w36f line as it stands today; none of the LB values exist yet.
"""
import numpy as np, pandas as pd, json

W = "/home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8/"
es = json.load(open(W + "experiments/w36f_esbias.json"))
A, B = es["beta"]                       # offset = A + B*oof
SHORT = 2.7294781234460386e-04          # zwr_realmlp's shortfall -- the n=1 es-on-val estimate
SIGMA = 3.764e-05                       # per-reading sd, from our 87 (cv,lb) points, 85 dof
GRID  = 1e-5                            # the public LB reporting grid

cal = pd.read_csv(W + "experiments/w37b_calfiles.csv")
cal["offset_line"] = A + B * cal.oof_auc
cal["pred_H0"] = cal.oof_auc + cal.offset_line              # honest: sits ON the line
cal["pred_H1"] = cal.pred_H0 - SHORT                        # es-inflated: sits BELOW it
cal["pred_lb"] = np.where(cal.role == "DIRTY", cal.pred_H1, cal.pred_H0)
cal["sep_steps"] = SHORT / GRID

print("PRE-REGISTERED PREDICTIONS  (line: offset = %.6f %+.6f * oof)\n" % (A, B))
print(f"{'order':>5} {'file':22s} {'role':13s} {'oof':>10} {'H0 (honest)':>12} {'H1 (es-bias)':>13} {'apart':>7}")
for r in cal.itertuples():
    print(f"{r.order:5d} {r.member:22s} {r.role:13s} {r.oof_auc:10.6f} "
          f"{r.pred_H0:12.6f} {r.pred_H1:13.6f} {r.sep_steps:6.0f} steps")

print(f"""
THE TEST. H0 and H1 are {SHORT/GRID:.0f} reporting steps apart. A single reading has sd
{SIGMA:.2e} = {SIGMA/GRID:.1f} steps, so EACH dirty send discriminates them on its own at
~{SHORT/SIGMA:.1f} sigma. Five of them do it five times independently, which is the point --
w36f had exactly one usable dirty reading and no way to tell a real effect from one member.

DECISION RULES, fixed now:

 R1  CLEAN-AUDIT `ram_hgb`. Its title's LB 0.96945 is one of only three points the line is
     fitted on. Our own send must return 0.96945 +/- 1 step. If it does not, the title was
     mis-attributed exactly as `tam_lkup`'s was, the w36f line is fitted on bad data, and
     EVERYTHING downstream of it -- including the +2.7e-4 es-on-val figure now in RESEARCH.md
     -- is withdrawn pending a refit. This send is checked FIRST.

 R2  CLEAN-ANCHOR `mkt_realmlp`. Predicted {cal[cal.member=='mkt_realmlp'].pred_H0.iloc[0]:.6f}.
     This is an extrapolation {(0.968026-0.9581337)*1e3:.1f}e-3 BELOW the fitted range, so it
     is a genuine out-of-sample test of the line's SHAPE, not just an extra point. If it lands
     within +/- 2*se the linear form survives and the low-AUC dirty readings are interpretable.
     If it misses badly, the offset is not linear in AUC over this range and only the dirty
     members ABOVE 0.966 can be priced. Either outcome is recorded; only the first licenses R3.

 R3  THE FIVE DIRTY READINGS. Report each shortfall = pred_H0 - actual_LB. Then:
       - mean shortfall, with the line error as a common floor (not averaged down);
       - sd ACROSS members, on 4 dof. THIS is the number that matters and it has never
         existed. Deflation of a quarantined member's OOF is only defensible if the
         inflation is roughly common; w37a can resolve a between-member sd above ~8.9e-5,
         i.e. 33% of the effect.
     Pre-registered reading: if sd < 33% of the mean, es-on-val is a roughly COMMON bias and
     a deflation arm is worth building. If sd is larger, the bias is member-specific, no
     single deflation is right, and THE QUARANTINE STANDS -- which is the outcome that costs
     us nothing and is named here because it is at least as likely.

 R4  NONE of these seven is a deadline candidate, ever. They are member vectors scoring
     0.958-0.969 against our 0.97118 and they exist to measure a bias. `check_selection.WANTED`
     is not touched by this experiment under any result.

 R5  If the day is cut short, order is authoritative: the CLEAN sends (orders 1 and 7) make
     the DIRTY ones interpretable, so `mkt_realmlp` goes first among the seven. Any dirty
     reading taken without it is still recorded but is priced with the wider pre-anchor se.
""")

# ---- wire into the send queue -------------------------------------------------------------
q = pd.read_csv(W + "experiments/w26d_queueprice.csv")
q = q[~q.file.str.startswith("w37_cal_")]          # idempotent: drop any earlier wiring
for c in ("send_rank", "msg"):
    if c not in q.columns:
        q[c] = np.nan
# the two CV leaders keep the head of the priority-1 band
q.loc[q.priority == 1, "send_rank"] = 1

new = []
for r in cal.itertuples():
    role = ("CLEAN ANCHOR - the enabling send" if r.role == "CLEAN-ANCHOR" else
            "CLEAN AUDIT of the line's own foundation" if r.role == "CLEAN-AUDIT" else
            "DIRTY reading")
    msg = (f"w37 es-bias CALIBRATION ({role}) — this is a MEASUREMENT, not an attempt on the "
           f"board. The file is the raw test-prediction vector of pack member `{r.member}`, "
           f"whose out-of-fold AUC on our frozen SKF5 seed-42 folds is {r.oof_auc:.10f}. "
           f"Expect it to score around {r.pred_lb:.5f} — far below this account's 0.97118 best, "
           f"which costs nothing because the public board shows best-of-all-submissions. "
           f"WHY: w36f priced early-stopping-on-the-scored-fold at +2.7e-4 of inflated OOF AUC "
           f"from a single usable reading, because the only other candidate's published score "
           f"turned out to belong to a blend rather than to that model. Submitting the vector "
           f"ourselves removes that ambiguity and returns an exact score for exactly these "
           f"predictions. Member's early-stopping mechanism: {r.es_mechanism}. Pre-registered "
           f"in experiments/w37c_prereg.py before any of these was sent: honest {r.pred_H0:.6f} "
           f"vs inflated {r.pred_H1:.6f}, {r.sep_steps:.0f} reporting steps apart. NOT a "
           f"deadline candidate under any outcome — final selection is on CV.")
    new.append(dict(file=r.file, sent=False, cv=r.oof_auc, rows=296302, md5=r.md5,
                    stem=r.file[:-4], fam="member", std=False, corr=False,
                    pred_lb=r.pred_lb, p_beat=0.0, priority=1,
                    send_rank=1 + r.order, msg=msg))

q = pd.concat([q, pd.DataFrame(new)], ignore_index=True)
q.to_csv(W + "experiments/w26d_queueprice.csv", index=False)
cal.to_csv(W + "experiments/w37c_prereg.csv", index=False)
print(f"queue rewritten: {len(new)} calibration rows added, {len(q)} rows total")
