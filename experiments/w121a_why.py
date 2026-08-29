"""w121a — WHY three Foundation-angle runs are invisible to the census reader.

The ANGLE INDEX's row 8 states the cause in prose: "w40/w58/w76 quote it without the word
`Foundation`". This measures the cause instead of restating it.
"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w117a_handcount as H

lines = open(H.JOURNAL, encoding="utf-8").read().split("\n")
for L in (17388, 21007, 25187):
    i = L - 1
    print("=" * 74)
    print("run:", lines[i][:72])
    for k in range(i, i + 45):
        joined = " ".join(lines[k:k + 3])
        if H.LABEL.search(lines[k]) and re.search(r"confirm the metric", joined, re.I):
            s = lines[k]
            for j in (k + 1, k + 2):
                if lines[j].strip():
                    s += " " + lines[j]
            print("  decl line", k + 1, ":", lines[k][:96])
            print("  contains 'Foundation'? ", bool(re.search(r"foundation", s, re.I)))
            print("  ANGLE_Q matches?       ", bool(H.ANGLE_Q.search(s)))
            for m in re.finditer(r"\b(?:angle|issued)\b", s, re.I):
                q = re.search(r"[“\"]", s[m.end():])
                gap = q.start() if q else None
                print(f"    label {m.group(0)!r:9} @{m.start():4d} -> next quote gap {gap}")
            print("  genera_of:", H.genera_of(s))
            print("  classify :", [H.classify(g) for g in H.genera_of(s)])
            break
