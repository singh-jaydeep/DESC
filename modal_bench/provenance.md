# Working-tree state for ledger entries at git 4482846d0

Generated 2026-08-21T16:30:38Z

## git status (desc/)
```
 M desc/optimize/least_squares.py
```

## diff (desc/)
```diff
diff --git a/desc/optimize/least_squares.py b/desc/optimize/least_squares.py
index 642008859..6cb3ac308 100644
--- a/desc/optimize/least_squares.py
+++ b/desc/optimize/least_squares.py
@@ -458,7 +458,7 @@ def lsqtr(  # noqa: C901
                 del U, s, Vt
             elif tr_method == "cho":
                 del B_h
-            elif tr_method in ("qr", "qr_struct", "qr_slim"):
+            elif tr_method in ("qr", "qr-struct", "qr-slim"):
                 del R
             J = jac(x, *args)
             njev += 1
```
