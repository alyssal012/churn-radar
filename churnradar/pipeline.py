"""Score current customers, pick who to target within a budget, and write the summary."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from . import features as F
from .models import RiskModel, TimeToChurn, UpliftModel


def early_warnings(f: pd.DataFrame, drop: float = 0.4, min_baseline: float = 5) -> pd.Series:
    """Customers whose usage fell sharply vs. *their own* normal, even if they still look
    average compared with everyone else."""
    return (f["_baseline_usage"] >= min_baseline) & (f["usage_trend"] <= 1 - drop)


def choose_targets(scored: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Spend the budget where it saves the most revenue.

    expected value of an offer = uplift x (value of keeping them) - cost of the offer
    Rank by value per dollar and take customers until the budget runs out. Anyone
    with negative expected value is skipped even if budget remains.
    """
    s = scored[scored["net_value"] > 0].sort_values("value_per_dollar", ascending=False)
    picked = s[s["offer_cost"].cumsum() <= budget]
    return picked


def run(cfg: F.Config, folder: pathlib.Path, budget: float, out: pathlib.Path) -> dict:
    data = F.load(cfg, folder)
    train = F.build(cfg, data, cfg.train_as_of)
    score = F.build(cfg, data, cfg.score_as_of)
    if score.empty:
        raise ValueError("no active customers on score_as_of")

    risk = RiskModel().fit(train)
    ttc = TimeToChurn().fit(train)
    uplift = None
    if "campaign" in data:
        offered = data["campaign"].set_index(cfg.id)[cfg.offered]
        uplift = UpliftModel().fit(train, offered)

    s = pd.DataFrame(index=score.index)
    s["churn_risk_90d"] = risk.predict(score)
    s = s.join(ttc.predict(score))
    s["monthly_price"] = score["monthly_price"]
    s["value_if_kept"] = score["monthly_price"] * cfg.value_months
    s["revenue_at_risk"] = s["churn_risk_90d"] * s["value_if_kept"]
    s["offer_cost"] = score["monthly_price"] * cfg.offer_cost_months
    raw = uplift.predict(score) if uplift else s["churn_risk_90d"] * 0.3  # fallback assumption, flagged in summary
    # a negative estimate means "the offer won't help"; and an offer can't cut churn
    # by more than the customer's total chance of leaving
    s["uplift"] = np.minimum(np.clip(raw, 0, None), s["churn_risk_90d"].values)
    s["net_value"] = s["uplift"] * s["value_if_kept"] - s["offer_cost"]
    s["value_per_dollar"] = s["net_value"] / s["offer_cost"]
    s["segment"] = np.select(
        [s.uplift >= 0.08, (s.churn_risk_90d >= 0.5) & (s.uplift < 0.08), s.churn_risk_90d < 0.15],
        ["persuadable", "likely lost", "safe"], "watch")
    s["early_warning"] = early_warnings(score)
    s["reasons"] = ""
    targets = choose_targets(s, budget)
    s["target"] = s.index.isin(targets.index)
    focus = s.index[(s.churn_risk_90d >= 0.3) | s.early_warning | s.target]
    if len(focus):
        s.loc[focus, "reasons"] = risk.reasons(score.loc[focus])
    targets = s[s.target]

    out.mkdir(parents=True, exist_ok=True)
    cols = ["segment", "target", "churn_risk_90d", "churn_prob_30d", "median_days_to_churn", "uplift",
            "monthly_price", "revenue_at_risk", "net_value", "early_warning", "reasons"]
    s.sort_values(["target", "net_value"], ascending=False)[cols].round(3).to_csv(out / "customers_scored.csv")
    stats = summarize(s, targets, budget, risk, ttc, uplift, train, cfg)
    (out / "summary.md").write_text(stats["markdown"])
    return stats


def summarize(s, targets, budget, risk, ttc, uplift, train, cfg) -> dict:
    n = len(s)
    high = s[s.churn_risk_90d >= 0.5]
    spend = float(targets.offer_cost.sum())
    saved_customers = float(targets.uplift.sum())
    saved_revenue = float((targets.uplift * targets.value_if_kept).sum())
    roi = saved_revenue / spend if spend else 0
    naive = s.sort_values("churn_risk_90d", ascending=False)
    naive = naive[naive.offer_cost.cumsum() <= budget]
    naive_saved = float((naive.uplift * naive.value_if_kept).sum())
    lost = s[s.segment == "likely lost"]
    warn = s[s.early_warning & ~s.target].sort_values("revenue_at_risk", ascending=False)
    soon = s[s.churn_prob_30d >= 0.25]
    seg = s.segment.value_counts()

    def money(x):
        return f"${x:,.0f}"

    ev = uplift.evaluation if uplift else None
    lines = [
        "# Retention summary",
        f"*Scored {n:,} active customers as of {cfg.score_as_of}. Budget: {money(budget)}.*",
        "",
        "## Key points",
        f"- **{len(high):,} customers ({len(high) / n:.0%}) are likely to cancel in the next 90 days**, "
        f"putting {money(high.revenue_at_risk.sum())} of yearly revenue at risk ({money(s.revenue_at_risk.sum())} across all customers).",
        f"- **{len(soon):,} have a 25%+ chance of leaving in the next 30 days.** These are the most urgent.",
        f"- **Recommended plan: send the offer to {len(targets):,} customers for {money(spend)}.** "
        f"Expected result: about {saved_customers:.0f} customers kept and {money(saved_revenue)} in revenue saved, "
        f"or **{roi:.1f}x the cost**.",
        f"- Targeting the {len(naive):,} *highest-risk* customers with the same budget would save only about "
        f"{money(naive_saved)}. Many of them would leave no matter what, so the money is better spent on persuadable customers.",
        f"- **{len(warn):,} customers show an early warning sign** (usage down 40%+ vs. their own normal) but aren't high risk yet. "
        "Worth a check-in, not a discount.",
        f"- Segments: {seg.get('persuadable', 0):,} persuadable, {seg.get('likely lost', 0):,} likely lost, "
        f"{seg.get('watch', 0):,} watch, {seg.get('safe', 0):,} safe.",
        "",
        "## Pros and cons of the recommended plan",
        "**Pros**",
        f"- Spends only on customers the offer is expected to change, so it skips the {len(lost):,} who are likely to leave anyway "
        "and the customers who would stay without a discount.",
        f"- Expected return of {roi:.1f}x, about {money(saved_revenue - naive_saved)} more than spending the same budget on the riskiest customers.",
        "- Every targeted customer has written reasons, so the retention team knows what to say to each one.",
        "",
        "**Cons and risks**",
    ]
    if uplift:
        lines.append(f"- Uplift is learned from one past campaign ({uplift.n_campaign:,} customers). "
                     "If the new offer or season differs, the effect may too.")
    else:
        lines.append("- **No past campaign data was provided**, so uplift is a rough assumption (30% of churn risk). "
                     "Run a small randomized test before trusting the targeting.")
    lines += [
        f"- {len(lost):,} high-risk customers get no offer under this plan. "
        f"That's a deliberate choice, but it means accepting about {money(lost.revenue_at_risk.sum())} of revenue at risk.",
        "- Discounts can train customers to wait for deals. Track whether kept customers renew at full price.",
        "- Predictions are estimates. Hold back a random 10% of targets as a control group to measure the real effect.",
        "",
        "## How much to trust the models",
        f"- Churn risk model: AUC {risk.auc:.2f} on held-out customers (0.5 = coin flip, 1.0 = perfect).",
        f"- Time-to-churn model: concordance {ttc.c_index:.2f} (how often it correctly orders who leaves first).",
    ]
    if ev:
        lines.append(f"- Uplift model, checked on held-out campaign customers: the top 30% it picked really did respond more "
                     f"(the offer cut their churn by {100 * ev['actual_uplift_top30']:.1f} points vs. {100 * ev['actual_uplift_rest']:.1f} for everyone else).")
    lines += ["", "## Top 10 customers to contact", "",
              "| Customer | Risk (90d) | Likely leaves in | Offer cuts risk by | Why |", "|---|---|---|---|---|"]
    for cid, r in targets.sort_values("net_value", ascending=False).head(10).iterrows():
        days = "> 6 months" if pd.isna(r.median_days_to_churn) else f"~{r.median_days_to_churn:.0f} days"
        lines.append(f"| {cid} | {r.churn_risk_90d:.0%} | {days} | {100 * r.uplift:.0f} pts | {s.loc[cid, 'reasons'] or '-'} |")
    lines += ["", "Full list: `customers_scored.csv`."]
    return {"markdown": "\n".join(lines) + "\n", "targets": len(targets), "spend": spend, "roi": roi,
            "saved_revenue": saved_revenue, "naive_saved_revenue": naive_saved, "auc": risk.auc,
            "c_index": ttc.c_index, "uplift_eval": ev, "high_risk": len(high), "early_warnings": len(warn)}
