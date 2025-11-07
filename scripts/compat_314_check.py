# scripts/compat_314_check.py
pkgs = [
    # base / ciencia de datos
    "numpy", "pandas", "matplotlib", "scipy", "sklearn", "statsmodels",
    # series/ML
    "pmdarima", "prophet", "cmdstanpy", "xgboost", "lightgbm",
    # finanzas/indicadores
    "yfinance", "ta", "talib",
    # deep learning (si usas)
    "torch", "tensorflow",
]

ok, bad = [], []
for name in pkgs:
    try:
        __import__(name)
        ok.append(name)
    except Exception as e:
        bad.append((name, repr(e)))

print("=== OK ===")
for p in ok:
    print(" ", p)

print("\n=== FALLÓ ===")
for p, err in bad:
    print(" ", p, "->", err)

