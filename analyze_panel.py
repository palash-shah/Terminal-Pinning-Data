"""
analyze_panel.py
================
Full empirical analysis pipeline for the bridge model paper, applied to
the polymarket_panel/ folder produced by collect_polymarket_panel.py v3.

Outputs:
  out/
    summary.json          -- per-market summary (beta_0, sigma_J, etc.)
    panel_table.tex       -- LaTeX-ready Table 1 contents
    sigma_pooled.tex      -- LaTeX-ready Table 2 (Sigma* convergence)
    sigma_pooled.json     -- raw cell ratios
    smile_table.tex       -- LaTeX-ready Table 3 (smile comparison)
    paths.png             -- all market paths
    logit_validation.png  -- logit overlay for first 6 informative markets
    smiles.png            -- smile comparison for first 4 informative markets
    pinning_rate.png      -- log-log P(1-P) vs (T-t)/T pooled

Run after the data folder is available:
    python3 analyze_panel.py --panel-dir polymarket_panel --out out/

Requires: numpy, scipy, matplotlib, pandas
"""

import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_iso(s):
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------
# Bridge: Sigma*(p, K)
# ----------------------------------------------------------------------
def Sigma_star_bridge(p, K):
    """The closed-form constant from Theorem 4.2."""
    if abs(K - p) < 1e-9:
        return 2 * norm.ppf(1 - p/2)
    target = p * (1 - K)
    log_pK = np.log(K / p)
    def c_bs(Sigma):
        d1 = -log_pK/Sigma + Sigma/2
        d2 = -log_pK/Sigma - Sigma/2
        return p*norm.cdf(d1) - K*norm.cdf(d2)
    try:
        return brentq(lambda S: c_bs(S) - target, 1e-3, 50)
    except ValueError:
        return None


def Sigma_at_price(p, K, c_target):
    """BS implied total variance such that c_BS_normalized(p, K, Sigma) = c_target."""
    if c_target <= max(p - K, 0) + 1e-12 or c_target >= p - 1e-12:
        return None
    log_pK = np.log(K / p) if K != p else 0.0
    def c_bs(Sigma):
        if K != p:
            d1 = -log_pK/Sigma + Sigma/2
            d2 = -log_pK/Sigma - Sigma/2
        else:
            d1 = Sigma/2
            d2 = -Sigma/2
        return p*norm.cdf(d1) - K*norm.cdf(d2)
    try:
        return brentq(lambda S: c_bs(S) - c_target, 1e-3, 50)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# Effective resolution time and beta_0 estimator
# ----------------------------------------------------------------------
def effective_resolution_time(df, expected_outcome, end_unix):
    """Earliest tick after which YES price stays within 0.01 of terminal value."""
    end_unix = end_unix if end_unix else df["t"].iloc[-1]
    if expected_outcome == "Yes":
        mask = df["p"] >= 0.99
    else:
        mask = df["p"] <= 0.01
    if mask.any():
        T_pin = df["t"].iloc[mask.idxmax()]
        return min(end_unix, T_pin)
    return end_unix


def beta0_QV(df, T_eff, delta_sec=300):
    """Quadratic-variation estimator of beta_0 from the bridge SDE."""
    sub = df[df["t"] <= T_eff - delta_sec].copy()
    if len(sub) < 10:
        sub = df.copy()
        T_eff = sub["t"].iloc[-1] + 60
    L = logit(sub["p"].values)
    QV = float(np.sum(np.diff(L)**2))
    t0 = sub["t"].iloc[0]
    t_end = sub["t"].iloc[-1]
    if T_eff - t_end <= 0:
        T_eff = t_end + 60
    Lambda_window = np.log((T_eff - t0) / (T_eff - t_end))
    if Lambda_window <= 0:
        return None
    return float(np.sqrt(max(QV / Lambda_window, 0)))


# ----------------------------------------------------------------------
# Baseline calibrations
# ----------------------------------------------------------------------
def calibrate_jacobi(df, T_eff, delta_sec=300):
    sub = df[df["t"] <= T_eff - delta_sec].copy()
    if len(sub) < 10:
        return None
    P = sub["p"].values
    dP = np.diff(P)
    P_mid = 0.5*(P[1:] + P[:-1])
    dt = np.diff(sub["t"].values)
    integrand = float(np.sum(P_mid*(1-P_mid)*dt))
    if integrand < 1e-9:
        return None
    return float(np.sqrt(min(np.sum(dP**2) / integrand, 100)))


def calibrate_logit_JD(df, T_eff, delta_sec=300):
    sub = df[df["t"] <= T_eff - delta_sec].copy()
    if len(sub) < 10:
        return None
    L = logit(sub["p"].values)
    dL = np.diff(L)
    dt = np.diff(sub["t"].values)
    horizon = sub["t"].iloc[-1] - sub["t"].iloc[0]
    if horizon <= 0:
        return None
    sorted_dL = np.sort(np.abs(dL))
    threshold = sorted_dL[int(0.95 * len(dL))] if len(dL) > 20 else sorted_dL[-1]
    is_jump = np.abs(dL) > threshold
    dL_diff = dL[~is_jump]
    dt_diff = dt[~is_jump]
    sigma_X_sq = (np.sum(dL_diff**2) / np.sum(dt_diff)) if np.sum(dt_diff) > 0 else 0
    n_jumps = int(is_jump.sum())
    lam = n_jumps / horizon
    jump_sd = float(np.sqrt(np.var(dL[is_jump]))) if n_jumps > 0 else 0.0
    return {"sigma_X": float(np.sqrt(max(sigma_X_sq, 0))),
            "lam": float(lam),
            "jump_sd": jump_sd,
            "n_jumps": n_jumps}


# ----------------------------------------------------------------------
# Simulators
# ----------------------------------------------------------------------
def jacobi_sim(p0, sigma_J, T_horizon, N=200_000, M=400, seed=20260428):
    rng = np.random.default_rng(seed)
    dt = T_horizon / M
    P = np.full(N, p0)
    half = N // 2
    for _ in range(M):
        Z_h = rng.standard_normal(half)
        Z = np.concatenate([Z_h, -Z_h])
        P = P + sigma_J*np.sqrt(np.maximum(P*(1-P), 0))*np.sqrt(dt)*Z
        P = np.clip(P, 0, 1)
    return P


def logit_JD_sim_recentered(p0, sigma_X, lam, jump_sd, T_horizon,
                             N=200_000, M=400, seed=20260428):
    rng = np.random.default_rng(seed)
    X0 = logit(p0)
    dt = T_horizon / M
    half = N // 2
    X = np.full(N, X0)
    Z_h = rng.standard_normal((M, half))
    Z = np.concatenate([Z_h, -Z_h], axis=1)
    n_jumps = rng.poisson(lam * T_horizon, N)
    for k in range(M):
        X = X + sigma_X*np.sqrt(dt)*Z[k]
    if jump_sd > 0:
        total_jump_var = jump_sd**2 * n_jumps
        X = X + rng.standard_normal(N)*np.sqrt(total_jump_var)
    def mean_p(c):
        return float(np.mean(sigmoid(X + c)))
    try:
        c_star = brentq(lambda c: mean_p(c) - p0, -10, 10)
    except ValueError:
        c_star = 0.0
    return sigmoid(X + c_star), c_star


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------
def load_panel(panel_dir):
    manifest_path = os.path.join(panel_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"ERROR: no manifest at {manifest_path}", file=sys.stderr)
        sys.exit(1)
    rows = list(csv.DictReader(open(manifest_path)))
    out = []
    for r in rows:
        csv_path = r.get("csv_path") or os.path.join(panel_dir, f"{r['slug']}_yes.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(panel_dir, os.path.basename(csv_path))
        if not os.path.exists(csv_path):
            print(f"  skipping {r['slug']}: csv missing")
            continue
        df = pd.read_csv(csv_path)
        df["t"] = df["t_unix"].astype(float)
        df["p"] = df["p"].astype(float)
        df = df.sort_values("t").reset_index(drop=True)
        end_unix = parse_iso(r.get("end_date"))
        out.append({"manifest": r, "df": df, "end_unix": end_unix})
    print(f"Loaded {len(out)} markets from {panel_dir}/")
    return out


# ----------------------------------------------------------------------
# Per-market analysis
# ----------------------------------------------------------------------
def analyze_market(market):
    df = market["df"]
    m = market["manifest"]
    end_unix = market["end_unix"]
    expected = m.get("expected_outcome", "")
    T_eff = effective_resolution_time(df, expected, end_unix)
    t0 = df["t"].iloc[0]
    horizon = T_eff - t0
    horizon_days = horizon / 86400.0
    p0 = float(df["p"].iloc[0])

    # interior fraction (all ticks)
    p = df["p"].values
    interior = float(np.mean((p >= 0.05) & (p <= 0.95)))

    beta0 = beta0_QV(df, T_eff) if interior >= 0.05 else None
    sigma_J = calibrate_jacobi(df, T_eff)
    jd = calibrate_logit_JD(df, T_eff)

    return {
        "slug": m["slug"],
        "question": m.get("question", ""),
        "topic": m.get("topic", "other"),
        "expected_outcome": expected,
        "p0": p0,
        "T_eff_unix": float(T_eff),
        "t0_unix": float(t0),
        "horizon_days": horizon_days,
        "n_ticks": len(df),
        "interior_fraction": interior,
        "beta0": beta0,
        "sigma_J": sigma_J,
        "logit_jd": jd,
        "volume": float(m.get("volume") or 0),
    }


# ----------------------------------------------------------------------
# Test P3: pooled Sigma_realized / Sigma* by time-bin and strike-offset
# ----------------------------------------------------------------------
def sigma_pooled_test(markets, summaries,
                      delta_days=1.0,
                      tau_bins=((0, 0.5), (0.5, 1.5), (1.5, 3),
                                (3, 5), (5, 8)),
                      anchor_bins=((0.05, 0.30), (0.30, 0.50),
                                   (0.50, 0.70), (0.70, 0.95)),
                      offsets=(-0.10, -0.05, 0.0, 0.05, 0.10),
                      min_pairs=10):
    """Pool realized (P_t, P_{t+Delta}) pairs across markets and compute
    Sigma_realized / Sigma*(p_anchor, K) ratios."""
    delta_sec = delta_days * 86400
    # Each cell:  (tau_bin, offset) -> list of ratios across (market, anchor_bin)
    cell_ratios = defaultdict(list)
    cell_pairs_count = defaultdict(int)

    for market, s in zip(markets, summaries):
        if s["interior_fraction"] < 0.05 or s["beta0"] is None:
            continue
        df = market["df"]
        T_eff = s["T_eff_unix"]
        t = df["t"].values
        p = df["p"].values
        # Tolerance: max(2h, 1.5x median tick spacing) so coarse-fidelity
        # markets (12h candles) still yield pairs
        if len(t) > 1:
            median_spacing = float(np.median(np.diff(t)))
        else:
            median_spacing = 0.0
        tol = max(2 * 3600, 1.5 * median_spacing)
        # Build (t_i, p_i, t_{i+Delta}, p_{i+Delta}) pairs.
        # For each i, find j such that t[j] is closest to t[i] + delta_sec.
        for i, t_i in enumerate(t):
            t_target = t_i + delta_sec
            if t_target >= T_eff:
                break
            j = np.searchsorted(t, t_target)
            if j >= len(t):
                continue
            if abs(t[j] - t_target) > tol:
                continue
            p_i = p[i]
            p_tplus = p[j]
            tau_minus_t_days = (T_eff - t_i) / 86400.0
            tau_bin = None
            for k, (lo, hi) in enumerate(tau_bins):
                if lo <= tau_minus_t_days < hi:
                    tau_bin = k
                    break
            if tau_bin is None:
                continue
            anchor_bin = None
            for k, (lo, hi) in enumerate(anchor_bins):
                if lo <= p_i < hi:
                    anchor_bin = k
                    break
            if anchor_bin is None:
                continue
            for offs in offsets:
                K = round(p_i + offs, 4)
                if K <= 0 or K >= 1:
                    continue
                cell_ratios[(tau_bin, anchor_bin, offs, s["slug"])].append(
                    (p_i, K, max(p_tplus - K, 0), tau_minus_t_days))
                cell_pairs_count[(tau_bin, anchor_bin, offs, s["slug"])] += 1

    # For each cell, compute realized payoff and its Sigma at the bin's
    # representative anchor; then ratio against Sigma* at that anchor.
    results = []
    for (tau_bin, anchor_bin, offs, slug), pairs in cell_ratios.items():
        if len(pairs) < min_pairs:
            continue
        p_anchor = float(np.mean([pp[0] for pp in pairs]))
        K = float(np.mean([pp[1] for pp in pairs]))
        c_realized = float(np.mean([pp[2] for pp in pairs]))
        sigma_imp = Sigma_at_price(p_anchor, K, c_realized)
        if sigma_imp is None:
            continue
        Sigma_realized = sigma_imp  # by definition Sigma_imp = sigma_imp * sqrt(Delta);
                                    # we want Sigma* to be the bridge's Theorem 4.2
                                    # constant evaluated at the same (p, K).
        # Note: Sigma* in the paper = sigma_imp * sqrt(tau-t). The pooled cell
        # already integrates over tau-t. We use the cell's mean tau-t for
        # comparison. Sigma_at_price returns BS total IV at that maturity.
        Sigma_star = Sigma_star_bridge(p_anchor, K)
        if Sigma_star is None or Sigma_star <= 0:
            continue
        # Convert sigma_imp at maturity Delta into Sigma_realized = sigma_imp*sqrt(Delta)
        # Actually Sigma_at_price returns BS Sigma which IS sigma*sqrt(Delta) in
        # our normalized BS formula (no maturity argument). So Sigma_realized
        # IS the Sigma at price; ratio against Sigma_star directly.
        ratio = Sigma_realized / Sigma_star
        results.append({
            "tau_bin": tau_bin,
            "tau_bin_range": tau_bins[tau_bin],
            "anchor_bin": anchor_bin,
            "anchor_bin_range": anchor_bins[anchor_bin],
            "K_offset": offs,
            "slug": slug,
            "p_anchor": p_anchor,
            "K": K,
            "c_realized": c_realized,
            "Sigma_realized": Sigma_realized,
            "Sigma_star": Sigma_star,
            "ratio": ratio,
            "n_pairs": len(pairs)
        })

    return results


def aggregate_pooled(results, tau_bins, offsets):
    """Pool across markets and anchor bins; produce a table cell mean ratio,
    cell SE, and cell n."""
    cell_data = defaultdict(list)
    for r in results:
        cell_data[(r["tau_bin"], r["K_offset"])].append(r["ratio"])
    table = {}
    for (tau_bin, offs), ratios in cell_data.items():
        n = len(ratios)
        mean = float(np.mean(ratios))
        if n > 1:
            se = float(np.std(ratios, ddof=1)) / np.sqrt(n)
        else:
            se = float("nan")
        table[(tau_bin, offs)] = {"mean": mean, "se": se, "n": n}
    return table


# ----------------------------------------------------------------------
# Smile comparison (across all interior-dynamics markets, pooled)
# ----------------------------------------------------------------------
def smile_comparison(markets, summaries, max_markets=None,
                     min_interior=0.20):
    """For each interior-dynamics market, compute Sigma_bridge, Sigma_jac,
    Sigma_jd at four strikes spanning the interior. Returns rows."""
    rows = []
    for market, s in zip(markets, summaries):
        if s["interior_fraction"] < min_interior or s["beta0"] is None:
            continue
        if s["sigma_J"] is None or s["logit_jd"] is None:
            continue
        p0 = s["p0"]
        horizon = s["T_eff_unix"] - s["t0_unix"]
        if horizon <= 0:
            continue
        # Simulate
        try:
            P_jac = jacobi_sim(p0, s["sigma_J"], horizon)
            jd = s["logit_jd"]
            P_jd, c_shift = logit_JD_sim_recentered(
                p0, jd["sigma_X"], jd["lam"], jd["jump_sd"], horizon)
        except Exception as e:
            print(f"  sim failed for {s['slug']}: {e}")
            continue
        # Strikes: interior of (0,1), spanning around p0
        if p0 < 0.5:
            strikes = sorted({0.05, p0, (1+p0)/2, 0.75})
        else:
            strikes = sorted({0.25, p0/2, p0, 0.95})
        for K in strikes:
            if K <= 0 or K >= 1:
                continue
            c_br = p0 * (1 - K) if K < 1 else 0
            Sb = Sigma_star_bridge(p0, K)
            c_ja = float(np.mean(np.maximum(P_jac - K, 0)))
            Sj = Sigma_at_price(p0, K, c_ja)
            c_lj = float(np.mean(np.maximum(P_jd - K, 0)))
            Sl = Sigma_at_price(p0, K, c_lj)
            rows.append({
                "slug": s["slug"], "question": s["question"], "p0": p0,
                "K": K, "horizon_days": s["horizon_days"],
                "beta0": s["beta0"], "sigma_J": s["sigma_J"],
                "lam_per_day": jd["lam"]*86400,
                "jump_sd": jd["jump_sd"],
                "c_bridge": c_br, "S_bridge": Sb,
                "c_jacobi": c_ja, "S_jacobi": Sj,
                "c_logit_jd": c_lj, "S_logit_jd": Sl,
            })
    return rows


# ----------------------------------------------------------------------
# LaTeX table writers
# ----------------------------------------------------------------------
def write_panel_table_tex(summaries, out_path, max_rows=12):
    """Top markets by volume, full panel in supplement."""
    sorted_s = sorted(summaries, key=lambda x: -x.get("volume", 0))
    show = sorted_s[:max_rows]
    lines = []
    for s in show:
        q = (s["question"] or s["slug"])[:55]
        if len(q) > 55:
            q = q[:52] + "..."
        q = q.replace("&", "\\&").replace("$", "\\$").replace("%", "\\%")
        beta0_str = f"{s['beta0']:.2f}" if s["beta0"] is not None else "—"
        horizon_str = f"{s['horizon_days']:.1f}" if s["horizon_days"] > 0 else "—"
        vol_m = (s.get("volume") or 0) / 1e6
        topic = s.get("topic", "other")
        lines.append(f"{q} & {topic} & {vol_m:.0f} & {s['expected_outcome']} & "
                     f"{s['p0']:.3f} & {horizon_str} & {s['interior_fraction']:.2f} & "
                     f"{beta0_str} \\\\")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_sigma_pooled_tex(table, tau_bins, offsets, out_path):
    """Render the (tau_bin x offset) table as TeX rows."""
    lines = []
    for ti, (lo, hi) in enumerate(tau_bins):
        center = (lo + hi) / 2
        cells = []
        for offs in offsets:
            cell = table.get((ti, offs))
            if cell is None or cell["n"] == 0:
                cells.append("---")
            else:
                cell_str = f"{cell['mean']:.3f}"
                if not np.isnan(cell["se"]):
                    cell_str += f"\\,({cell['se']:.2f})"
                cell_str += f" [{cell['n']}]"
                cells.append(cell_str)
        lines.append(f"$[{lo:g},{hi:g})$ & " + " & ".join(cells) + " \\\\")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_smile_table_tex(rows, out_path, max_markets=4):
    """Render top-N interior-dynamics markets in the smile comparison."""
    by_slug = defaultdict(list)
    for r in rows:
        by_slug[r["slug"]].append(r)
    chosen_slugs = list(by_slug.keys())[:max_markets]
    lines = []
    for slug in chosen_slugs:
        rs = sorted(by_slug[slug], key=lambda x: x["K"])
        first = rs[0]
        q = first["question"][:30].replace("&", "\\&")
        for i, r in enumerate(rs):
            beta0_str = f"{r['beta0']:.2f}" if i == 0 else "—"
            sj_str = f"{r['sigma_J']*1e3:.2f}" if i == 0 else "—"
            lam_str = f"{r['lam_per_day']:.1f}" if i == 0 else "—"
            Sb = f"{r['S_bridge']:.2f}" if r['S_bridge'] is not None else "—"
            Sj = f"{r['S_jacobi']:.2f}" if r['S_jacobi'] is not None else "—"
            Sl = f"{r['S_logit_jd']:.2f}" if r['S_logit_jd'] is not None else "—"
            row_label = q if i == 0 else ""
            lines.append(f"{row_label} & {r['K']:.2f} & {beta0_str} & {r['c_bridge']:.3f} & {Sb} & "
                         f"{sj_str} & {r['c_jacobi']:.3f} & {Sj} & "
                         f"{lam_str} & {r['c_logit_jd']:.3f} & {Sl} \\\\")
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines = lines[:-1]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_paths(markets, summaries, out_path, ncols=4):
    n = len(markets)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 2.2*nrows))
    if nrows == 1:
        axes = [axes]
    axes_flat = [a for row in axes for a in row] if nrows > 1 else axes
    for ax, m, s in zip(axes_flat, markets, summaries):
        df = m["df"]
        days = (df["t"] - df["t"].iloc[0]) / 86400
        ax.plot(days, df["p"], lw=0.6)
        if s["T_eff_unix"]:
            T_days = (s["T_eff_unix"] - df["t"].iloc[0]) / 86400
            ax.axvline(T_days, color="red", ls="--", lw=0.5)
        ax.set_ylim(-0.05, 1.05)
        title = (m["manifest"].get("question") or m["manifest"]["slug"])[:35]
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes_flat[len(markets):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_logit_validation(markets, summaries, out_path,
                          max_markets=6, min_interior=0.30):
    selected = [(m, s) for m, s in zip(markets, summaries)
                if s["interior_fraction"] >= min_interior and s["beta0"] is not None]
    selected = selected[:max_markets]
    if not selected:
        print("  no markets pass interior filter for logit_validation")
        return
    n = len(selected)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.2*nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for ax, (m, s) in zip(axes_flat, selected):
        df = m["df"]
        T_eff = s["T_eff_unix"]
        t0 = s["t0_unix"]
        beta0 = s["beta0"]
        p0 = s["p0"]
        L0 = logit(p0)
        x = 1.0 if s["expected_outcome"] == "Yes" else 0.0
        eps = 300
        sub = df[(df["t"] >= t0) & (df["t"] <= T_eff - eps)]
        if len(sub) < 5:
            continue
        Lambda_t = np.log((T_eff - t0) / np.maximum(T_eff - sub["t"].values, 1))
        L_emp = logit(sub["p"].values)
        mu = L0 + (2*x - 1) * beta0**2 * Lambda_t
        sd = beta0 * np.sqrt(np.maximum(Lambda_t, 0))
        days = (sub["t"].values - t0) / 86400
        ax.plot(days, L_emp, lw=0.7, label="empirical $L_t$")
        ax.plot(days, mu, color="C1", lw=1.2, label=f"bridge mean ($X_T={int(x)}$)")
        ax.fill_between(days, mu - 2*sd, mu + 2*sd, color="C1", alpha=0.2,
                        label="$\\pm 2\\sigma$")
        title = (m["manifest"].get("question") or m["manifest"]["slug"])[:42]
        ax.set_title(f"{title}\n($\\beta_0$={beta0:.2f})", fontsize=8)
        ax.set_xlabel("days from start", fontsize=8)
        ax.set_ylabel("$L_t$", fontsize=8)
        ax.legend(fontsize=6, loc="best")
        ax.tick_params(labelsize=7)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_smiles(rows, out_path, max_markets=4):
    by_slug = defaultdict(list)
    for r in rows:
        by_slug[r["slug"]].append(r)
    chosen = list(by_slug.keys())[:max_markets]
    if not chosen:
        print("  no markets for smile plot")
        return
    n = len(chosen)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for ax, slug in zip(axes_flat, chosen):
        rs = sorted(by_slug[slug], key=lambda x: x["K"])
        K_arr = [r["K"] for r in rs]
        Sb = [r["S_bridge"] for r in rs]
        Sj = [r["S_jacobi"] for r in rs]
        Sl = [r["S_logit_jd"] for r in rs]
        ax.plot(K_arr, Sb, "o-", color="C0", label=f"Bridge ($\\beta_0$={rs[0]['beta0']:.2f})")
        # filter Nones for Jacobi and JD
        K_j = [r["K"] for r in rs if r["S_jacobi"] is not None]
        S_j = [r["S_jacobi"] for r in rs if r["S_jacobi"] is not None]
        K_l = [r["K"] for r in rs if r["S_logit_jd"] is not None]
        S_l = [r["S_logit_jd"] for r in rs if r["S_logit_jd"] is not None]
        ax.plot(K_j, S_j, "s-", color="C1", label=f"Jacobi ($\\sigma_J$={rs[0]['sigma_J']:.4f})")
        ax.plot(K_l, S_l, "^-", color="C2", label=f"Logit JD ($\\lambda$={rs[0]['lam_per_day']:.1f}/d)")
        ax.axvline(rs[0]["p0"], color="gray", ls=":", lw=0.8, label=f"$p_0$={rs[0]['p0']:.3f}")
        title = (rs[0]["question"])[:42]
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("strike $K$", fontsize=9)
        ax.set_ylabel("$\\Sigma$", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_pinning_rate(markets, summaries, out_path, min_interior=0.10):
    """Pool log P(1-P) vs log((T-t)/T) across all markets."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, s in zip(markets, summaries):
        if s["interior_fraction"] < min_interior or s["beta0"] is None:
            continue
        df = m["df"]
        T_eff = s["T_eff_unix"]
        t0 = s["t0_unix"]
        sub = df[df["t"] <= T_eff - 300]
        if len(sub) < 20:
            continue
        T_total = T_eff - t0
        log_x = np.log(np.maximum(T_eff - sub["t"].values, 1) / T_total)
        p_vals = sub["p"].values
        log_y = np.log(np.maximum(p_vals * (1 - p_vals), 1e-10))
        ax.scatter(log_x, log_y, s=2, alpha=0.3)
    ax.set_xlabel("$\\log((T_{\\rm eff} - t)/T_{\\rm eff})$")
    ax.set_ylabel("$\\log P_t(1-P_t)$")
    ax.set_title("Pooled pinning-rate scatter (all interior-dynamics markets)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-dir", default="polymarket_panel")
    ap.add_argument("--out", default="out")
    ap.add_argument("--delta-days", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load
    markets = load_panel(args.panel_dir)
    if not markets:
        print("ERROR: no markets loaded", file=sys.stderr)
        sys.exit(1)

    # Per-market analysis
    print("Analyzing markets...")
    summaries = [analyze_market(m) for m in markets]
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    # Statistics
    informative = [s for s in summaries
                   if s["interior_fraction"] >= 0.20 and s["beta0"] is not None]
    all_with_beta = [s for s in summaries if s["beta0"] is not None]
    print(f"\n=== Sample summary ===")
    print(f"  Total markets: {len(summaries)}")
    print(f"  Interior-dynamics (>=0.20): {len(informative)}")
    print(f"  With valid beta_0: {len(all_with_beta)}")
    if all_with_beta:
        betas = [s["beta0"] for s in all_with_beta]
        print(f"  beta_0 stats: mean={np.mean(betas):.3f}, median={np.median(betas):.3f}, "
              f"std={np.std(betas):.3f}, range=[{min(betas):.3f}, {max(betas):.3f}]")
    if informative:
        betas_i = [s["beta0"] for s in informative]
        print(f"  beta_0 (interior only): mean={np.mean(betas_i):.3f}, "
              f"median={np.median(betas_i):.3f}, std={np.std(betas_i):.3f}")
    topic_counts = defaultdict(int)
    for s in summaries:
        topic_counts[s["topic"]] += 1
    print(f"  topic breakdown: {dict(topic_counts)}")

    # Test P3 (pooled Sigma ratio)
    print("\nRunning Sigma* convergence test...")
    tau_bins = ((0, 0.5), (0.5, 1.5), (1.5, 3), (3, 5), (5, 8))
    offsets = (-0.10, -0.05, 0.0, 0.05, 0.10)
    p3_results = sigma_pooled_test(markets, summaries,
                                    delta_days=args.delta_days,
                                    tau_bins=tau_bins,
                                    offsets=offsets)
    p3_table = aggregate_pooled(p3_results, tau_bins, offsets)
    print(f"  cells: {len(p3_table)}")
    for (ti, offs), cell in sorted(p3_table.items()):
        lo, hi = tau_bins[ti]
        se_str = f" se={cell['se']:.3f}" if not np.isnan(cell['se']) else ""
        print(f"    tau in [{lo},{hi}], K-p={offs:+.2f}: ratio={cell['mean']:.3f}{se_str} (n={cell['n']})")
    with open(os.path.join(args.out, "sigma_pooled.json"), "w") as f:
        json.dump({"results": p3_results,
                   "table": {f"{ti}_{offs}": cell
                             for (ti, offs), cell in p3_table.items()}},
                  f, indent=2, default=str)
    write_sigma_pooled_tex(p3_table, tau_bins, offsets,
                           os.path.join(args.out, "sigma_pooled.tex"))

    # Smile comparison
    print("\nRunning smile comparison...")
    smile_rows = smile_comparison(markets, summaries, min_interior=0.20)
    print(f"  rows: {len(smile_rows)}")
    with open(os.path.join(args.out, "smile_rows.json"), "w") as f:
        json.dump(smile_rows, f, indent=2, default=str)
    write_smile_table_tex(smile_rows, os.path.join(args.out, "smile_table.tex"))

    # Panel table
    write_panel_table_tex(summaries, os.path.join(args.out, "panel_table.tex"))

    # Plots
    print("\nMaking plots...")
    plot_paths(markets, summaries, os.path.join(args.out, "paths.png"))
    plot_logit_validation(markets, summaries,
                          os.path.join(args.out, "logit_validation.png"))
    plot_smiles(smile_rows, os.path.join(args.out, "smiles.png"))
    plot_pinning_rate(markets, summaries,
                      os.path.join(args.out, "pinning_rate.png"))

    print(f"\n=== Done. All outputs in {args.out}/ ===")


if __name__ == "__main__":
    main()
