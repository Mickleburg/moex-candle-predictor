"""H6 — Macro / cross-asset reaction as a cross-sectional signal (sleeve S2).

ECONOMIC THESIS
    A name's sensitivity (rolling beta) to an exogenous driver (Brent oil, USDRUB) should let us
    TILT the cross-section: when the driver is moving, lean toward the names it moves. This is the
    CAUSE of relative strength that pure price (H1) lacked: oil up -> exporters (ROSN/LKOH/TATN/
    GAZP/NVTK/SNGS) over domestic (MGNT/MTSS); RUB weaker -> exporters over domestic.

WHAT IS (AND IS NOT) A SIGNAL  — the careful part
    "Always long exporters" is a STATIC ticker bias, not a dynamic signal: it would survive a
    time-shuffle (permuting a feature across time) because it never changes which names rank high.
    The honest dynamic signal is a CONDITIONAL tilt:
        score_i(t) = rolling_beta_i(driver)(t)  *  driver_momentum(t)
    which flips sign over time as the driver rises/falls. Under time-shuffle the alignment between
    "driver was rising" and "high-beta names then outperformed" must BREAK. We test both the static
    beta and the dynamic tilt so the contrast is explicit. A variant only counts as signal if its
    FORWARD rank-IC is positive, stable, AND clearly above its time-shuffled control.

NO-LOOKAHEAD
    All features at decision time t use returns through close[t] only (rolling betas/momenta end at
    t). The harness label uses close[t+H] purely as the outcome. Same rig as xsec_eval_harness, so
    numbers are directly comparable to the price-momentum benchmark printed there.

FX PROXY
    USD spot (USD000UTSTOM) is suspended since mid-2024. RTS index is the IMOEX basket priced in USD,
    so USDRUB_proxy ~= IMOEX / RTSI (rising = RUB depreciation = exporter tailwind). Used for the FX
    driver; works across the suspension because both indices keep printing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from src.data.load import to_moscow_time  # noqa: E402
from scripts.xsec_eval_harness import (  # noqa: E402
    UNIVERSE, FORWARD_START, load_daily_panel, evaluate_scores, momentum_score, print_row,
)

DATA_RAW = REPO_ROOT / "data" / "raw"
SHUFFLE_SEEDS = (1, 2, 3, 4, 5)


# ----------------------------- macro data loading -----------------------------

def _load_daily_close(symbol: str) -> pd.Series:
    """Daily last-close of a single instrument from its 1H parquet (Moscow tz)."""
    files = sorted(DATA_RAW.glob(f"{symbol}_1H_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no 1H parquet for {symbol}")
    df = pd.read_parquet(files[-1])
    df.columns = [c.lower() for c in df.columns]
    s = pd.Series(df["close"].to_numpy(float), index=to_moscow_time(df["begin"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.resample("1D").last()


def load_macro_returns(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Daily returns of the macro drivers, aligned (ffill) to the equity panel index.

    oil : Brent continuous future (BR_CONT)
    fx  : USDRUB proxy = IMOEX / RTSI  (rising = RUB weaker)
    rates: RGBI govt-bond index (rising = yields down = risk-on / banks tailwind)
    """
    brent = _load_daily_close("BR_CONT").reindex(index).ffill()
    imoex = _load_daily_close("IMOEX").reindex(index).ffill()
    rtsi = _load_daily_close("RTSI").reindex(index).ffill()
    rgbi = _load_daily_close("RGBI").reindex(index).ffill()
    fx_proxy = imoex / rtsi
    return {
        "oil": brent.pct_change(),
        "fx": fx_proxy.pct_change(),
        "rates": rgbi.pct_change(),
    }


# ----------------------------- feature construction ---------------------------

def rolling_beta_to(panel: pd.DataFrame, factor_ret: pd.Series, window: int) -> pd.DataFrame:
    """Past-only rolling beta of each ticker's daily return vs an external factor return."""
    r = panel.pct_change()
    cov = r.rolling(window).cov(factor_ret)
    var = factor_ret.rolling(window).var()
    return cov.div(var, axis=0)


def static_beta_score(panel: pd.DataFrame, factor_ret: pd.Series, window: int) -> pd.DataFrame:
    """Score matrix = rolling beta to the factor (a STATIC-ish sensitivity tilt)."""
    return rolling_beta_to(panel, factor_ret, window)


def tilt_score(panel: pd.DataFrame, factor_ret: pd.Series,
               beta_window: int, mom_lookback: int) -> pd.DataFrame:
    """Dynamic tilt = beta_i(t) * driver_momentum(t). Sign flips as the driver moves."""
    beta = rolling_beta_to(panel, factor_ret, beta_window)
    driver_mom = factor_ret.rolling(mom_lookback).sum()          # cumulative recent driver move
    return beta.mul(driver_mom, axis=0)


def combined_tilt(panel: pd.DataFrame, macro: dict[str, pd.Series],
                  keys: list[str], beta_window: int, mom_lookback: int) -> pd.DataFrame:
    """Sum of per-driver tilts, each cross-sectionally z-scored per date so drivers are comparable."""
    total = None
    for kk in keys:
        s = tilt_score(panel, macro[kk], beta_window, mom_lookback)
        z = s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1).replace(0, np.nan), axis=0)
        total = z if total is None else total.add(z, fill_value=0.0)
    return total


# ----------------------------- evaluation helpers -----------------------------

def matrix_score_fn(mat: pd.DataFrame):
    """Wrap a precomputed [time x ticker] score matrix as the harness score_fn (past-only row t)."""
    arr = mat.to_numpy(float)

    def fn(panel: pd.DataFrame, t: int):
        row = arr[t]
        return None if not np.all(np.isfinite(row)) else row
    return fn


def shuffle_columns(mat: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute each ticker's feature values across TIME (kills temporal alignment, keeps marginal)."""
    rng = np.random.default_rng(seed)
    out = mat.copy()
    for c in out.columns:
        col = out[c].to_numpy(float).copy()
        finite = np.isfinite(col)
        idx = np.where(finite)[0]
        perm = rng.permutation(idx)
        col[idx] = col[perm]
        out[c] = col
    return out


def shuffled_fwd_ic(panel: pd.DataFrame, mat: pd.DataFrame, horizon: int) -> float:
    """Mean FORWARD rank-IC under time-shuffle, averaged over seeds (the control number)."""
    vals = []
    for sd in SHUFFLE_SEEDS:
        m = evaluate_scores(panel, matrix_score_fn(shuffle_columns(mat, sd)), horizon=horizon,
                            label="shuf")
        vals.append(m["ic_fwd_mean"])
    return float(np.mean(vals))


def assess(panel: pd.DataFrame, mat: pd.DataFrame, horizon: int, label: str) -> dict:
    m = evaluate_scores(panel, matrix_score_fn(mat), horizon=horizon, label=label)
    m["ic_fwd_shuf"] = round(shuffled_fwd_ic(panel, mat, horizon), 4)
    m["fwd_edge_vs_shuf"] = round(m["ic_fwd_mean"] - m["ic_fwd_shuf"], 4)
    return m


def print_h6(m: dict) -> None:
    flag = "  <-- dynamic" if (m["ic_fwd_mean"] > 0 and m["fwd_edge_vs_shuf"] > 0.01
                               and m["ic_fwd_ir"] > 0) else ""
    print(f"{m['label']:26} H={m['horizon']:>2} | "
          f"IS={m['ic_is_mean']:+.4f}  FWD={m['ic_fwd_mean']:+.4f}(IR{m['ic_fwd_ir']:+.2f})  "
          f"shuf={m['ic_fwd_shuf']:+.4f}  edge={m['fwd_edge_vs_shuf']:+.4f} | "
          f"btFWD={m['bt_fwd_cum']:+.3f} win={m['bt_fwd_win']:.2f}{flag}")


# ----------------------------------- main -------------------------------------

def main() -> int:
    panel = load_daily_panel()
    macro = load_macro_returns(panel.index)
    print(f"H6 macro/cross-asset cross-sectional signal")
    print(f"  panel: {panel.shape[1]} tickers x {len(panel)} days, "
          f"{panel.index.min().date()}..{panel.index.max().date()}; forward from {FORWARD_START.date()}")
    # data sanity: macro coverage on the panel grid
    for kk, s in macro.items():
        cov = float(s.reindex(panel.index).notna().mean())
        print(f"  macro[{kk}]: coverage {cov:.2%}, "
              f"daily std {float(s.reindex(panel.index).std()):.4f}")
    print()

    print("Read: a variant is a DYNAMIC signal only if FWD IC > 0, IR > 0, AND edge over its")
    print("time-shuffled control (edge>0.01). 'shuf' is the static-bias null; 'edge' is real timing.\n")

    BETA_W = 60
    HORIZONS = (5, 10, 20)

    print("--- price-momentum benchmark (from harness, the number to beat) ---")
    for L in (20, 60):
        for H in HORIZONS:
            print_row(evaluate_scores(panel, momentum_score(L), horizon=H, k=3, label=f"mom_L{L}"))
    print()

    print("--- STATIC beta tilt (expected to be a static bias: small edge vs shuffle) ---")
    for drv in ("oil", "fx"):
        for H in HORIZONS:
            mat = static_beta_score(panel, macro[drv], BETA_W)
            print_h6(assess(panel, mat, H, f"beta_{drv}"))
    print()

    print("--- DYNAMIC tilt beta*driver_momentum (the H6 signal) ---")
    for drv in ("oil", "fx", "rates"):
        for mom in (10, 20):
            for H in HORIZONS:
                mat = tilt_score(panel, macro[drv], BETA_W, mom)
                print_h6(assess(panel, mat, H, f"tilt_{drv}_m{mom}"))
        print()

    print("--- COMBINED oil+fx tilt (z-scored sum) ---")
    for mom in (10, 20):
        for H in HORIZONS:
            mat = combined_tilt(panel, macro, ["oil", "fx"], BETA_W, mom)
            print_h6(assess(panel, mat, H, f"tilt_oilfx_m{mom}"))
    print()

    print("--- COMBINED oil+fx+rates tilt ---")
    for mom in (10, 20):
        for H in HORIZONS:
            mat = combined_tilt(panel, macro, ["oil", "fx", "rates"], BETA_W, mom)
            print_h6(assess(panel, mat, H, f"tilt_all_m{mom}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
