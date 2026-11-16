"""Turn any company's customer table + activity log into model features.

Your data only needs:
  customers: an ID, a signup date, a monthly price, and a churn date (blank if still active)
  events:    an ID, a date, and a usage count (logins, orders, sessions...)
Optional columns (plan, support tickets, price increase date) are used when present.
Column names are mapped in a small config, so nothing has to be renamed.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Config:
    customers_file: str = "customers.csv"
    events_file: str = "events.csv"
    campaign_file: str | None = "campaign.csv"
    id: str = "customer_id"
    signup_date: str = "signup_date"
    price: str = "monthly_price"
    churn_date: str = "churn_date"
    plan: str | None = "plan"
    price_increase_date: str | None = "price_increase_date"
    event_date: str = "date"
    usage: str = "usage"
    tickets: str | None = "support_tickets"
    offered: str = "offered"
    train_as_of: str = ""
    score_as_of: str = ""
    horizon_days: int = 90
    offer_cost_months: float = 1.0     # the offer costs one month of the customer's price
    value_months: float = 12.0         # value of keeping a customer = 12 months of revenue
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Config":
        return cls(**json.loads(pathlib.Path(path).read_text()))


def load(cfg: Config, folder: pathlib.Path) -> dict[str, pd.DataFrame]:
    c = pd.read_csv(folder / cfg.customers_file)
    e = pd.read_csv(folder / cfg.events_file)
    for col in [cfg.signup_date, cfg.churn_date] + ([cfg.price_increase_date] if cfg.price_increase_date in c else []):
        c[col] = pd.to_datetime(c[col], errors="coerce")
    e[cfg.event_date] = pd.to_datetime(e[cfg.event_date])
    out = {"customers": c, "events": e}
    if cfg.campaign_file and (folder / cfg.campaign_file).exists():
        out["campaign"] = pd.read_csv(folder / cfg.campaign_file)
    return out


FEATURE_LABELS = {
    "tenure_days": "time as a customer",
    "usage_per_week_recent": "recent usage",
    "usage_trend": "change in usage vs. their own normal",
    "active_weeks_recent": "weeks active out of the last 12",
    "days_since_active": "days since last activity",
    "tickets_recent": "support tickets in the last 90 days",
    "monthly_price": "monthly price",
    "recent_price_increase": "a recent price increase",
}


def build(cfg: Config, data: dict[str, pd.DataFrame], as_of: str) -> pd.DataFrame:
    """One row per customer who was active on `as_of`, using only data from before it."""
    t = pd.Timestamp(as_of)
    c, e = data["customers"], data["events"]
    active = c[(c[cfg.signup_date] < t) & (c[cfg.churn_date].isna() | (c[cfg.churn_date] >= t))].copy()
    past = e[(e[cfg.event_date] < t) & e[cfg.id].isin(active[cfg.id])]

    recent = past[past[cfg.event_date] >= t - pd.Timedelta(days=28)]
    base = past[(past[cfg.event_date] < t - pd.Timedelta(days=28)) & (past[cfg.event_date] >= t - pd.Timedelta(days=112))]
    last12 = past[past[cfg.event_date] >= t - pd.Timedelta(weeks=12)]
    g = lambda df, col, fn: df.groupby(cfg.id)[col].agg(fn)

    f = pd.DataFrame(index=active[cfg.id])
    f["tenure_days"] = (t - active.set_index(cfg.id)[cfg.signup_date]).dt.days
    f["usage_per_week_recent"] = g(recent, cfg.usage, "sum") / 4
    base_rate = g(base, cfg.usage, "sum") / 12
    f["usage_trend"] = (f["usage_per_week_recent"] + 1) / (base_rate.reindex(f.index) + 1)
    f["active_weeks_recent"] = g(last12[last12[cfg.usage] > 0], cfg.usage, "count")
    last_active = g(past[past[cfg.usage] > 0], cfg.event_date, "max")
    f["days_since_active"] = (t - last_active.reindex(f.index)).dt.days
    if cfg.tickets and cfg.tickets in e:
        f["tickets_recent"] = g(past[past[cfg.event_date] >= t - pd.Timedelta(days=90)], cfg.tickets, "sum")
    f["monthly_price"] = active.set_index(cfg.id)[cfg.price]
    if cfg.price_increase_date and cfg.price_increase_date in c:
        pi = active.set_index(cfg.id)[cfg.price_increase_date]
        f["recent_price_increase"] = ((pi < t) & (pi >= t - pd.Timedelta(days=120))).astype(int)
    if cfg.plan and cfg.plan in c:
        for p in sorted(c[cfg.plan].dropna().unique()):
            f[f"plan_{p}"] = (active.set_index(cfg.id)[cfg.plan] == p).astype(int)
    f = f.fillna({"usage_per_week_recent": 0, "active_weeks_recent": 0, "tickets_recent": 0,
                  "usage_trend": 1.0, "days_since_active": 999})

    # baseline for the early-warning check: their own usage over the prior 12 weeks
    f["_baseline_usage"] = base_rate.reindex(f.index).fillna(0)

    # labels and survival time (only meaningful for the training snapshot)
    churn = active.set_index(cfg.id)[cfg.churn_date]
    end = c[cfg.churn_date].max() if c[cfg.churn_date].notna().any() else t
    end = max(end, e[cfg.event_date].max())
    f["_churned_in_horizon"] = (churn.notna() & (churn < t + pd.Timedelta(days=cfg.horizon_days))).astype(int)
    f["_duration_days"] = np.where(churn.notna(), (churn - t).dt.days, (end - t).days).clip(min=1)
    f["_event"] = churn.notna().astype(int)
    return f


def model_columns(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if not c.startswith("_")]


def label(col: str) -> str:
    if col.startswith("plan_"):
        return f"being on the {col[5:]} plan"
    return FEATURE_LABELS.get(col, col.replace("_", " "))
