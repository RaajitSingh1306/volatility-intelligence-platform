import io, sys
P = "frontend/pages/index.tsx"
t = io.open(P, encoding="utf-8", newline="").read()
crlf = "\r\n" in t; t = t.replace("\r\n", "\n"); bad = False
for old, new, label in [
 ("      setSummary(sumRes);\n", "      setSummary({ ...DEFAULT_SUMMARY, ...sumRes });\n", "merge summary"),
 ("      setPredict(predRes);\n", "      setPredict({ ...DEFAULT_PREDICT, ...predRes });\n", "merge predict"),
]:
    if new in t: print("ALREADY".ljust(11), label); continue
    n = t.count(old)
    if n != 1: print(("MISSING" if n == 0 else "NOT-UNIQUE").ljust(11), label); bad = True; continue
    print("PATCHED".ljust(11), label); t = t.replace(old, new)
if bad: print("\nNOTHING WRITTEN (problems above)"); sys.exit(1)
io.open(P, "w", encoding="utf-8", newline="").write(t.replace("\n", "\r\n") if crlf else t)
print("\ndone")