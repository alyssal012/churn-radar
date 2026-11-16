"""Demo data for a subscription business: customers, weekly activity, and a past
retention campaign where a random group got a discount offer.

The simulation builds in the patterns that real retention teams see:
- usage drops for weeks before someone cancels
- price increases and support problems push people out
- a discount only changes the mind of *some* customers: very engaged people
  stay anyway, and people who have already checked out leave anyway

All data is synthetic. Point the tool at your own CSVs for real use.
"""
from __future__ import annotations

import datetime as dt
import pathlib

import numpy as np
import pandas as pd

START = dt.date(2025, 1, 6)  # a Monday
WEEKS = 78
CAMPAIGN_WEEK = 52            # past campaign ran at the training snapshot
PLANS = {"basic": 19.0, "pro": 49.0, "team": 129.0}


def simulate(n: int = 6000, seed: int = 5) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    customers, events, campaign = [], [], []
    for cid in range(1, n + 1):
        signup_week = int(rng.integers(0, 40))
        plan = rng.choice(list(PLANS), p=[0.55, 0.35, 0.10])
        engagement = rng.beta(2.2, 1.6)                   # 0 = checked out, 1 = power user
        price_hike = rng.random() < 0.25
        hike_week = int(rng.integers(signup_week + 8, WEEKS)) if price_hike else None
        price = PLANS[plan]
        ticket_rate = rng.uniform(0.02, 0.25)
        price_sensitive = rng.random() < 0.5
        offered = None
        churn_week = None

        for w in range(signup_week, WEEKS):
            # engagement drifts; it slowly erodes for some customers
            engagement = float(np.clip(engagement + rng.normal(-0.006, 0.03), 0, 1))
            p_now = price * (1.15 if hike_week is not None and w >= hike_week else 1)
            tickets = int(rng.poisson(ticket_rate * (2.0 if engagement < 0.3 else 1.0)))
            usage = int(rng.poisson(1 + 24 * engagement))
            events.append((cid, START + dt.timedelta(weeks=w), usage, tickets))

            if w == CAMPAIGN_WEEK:
                offered = bool(rng.random() < 0.5)

            # weekly cancellation hazard
            h = 0.004 + 0.05 * (1 - engagement) ** 3 + 0.012 * tickets
            if hike_week is not None and hike_week <= w < hike_week + 10:
                h += 0.012 if price_sensitive else 0.004
            if plan == "team":
                h *= 0.6
            # the offer (applied for 13 weeks after the campaign) only moves the middle
            if offered and CAMPAIGN_WEEK <= w < CAMPAIGN_WEEK + 13:
                recent_hike = hike_week is not None and hike_week <= w < hike_week + 17
                persuadable = 0.15 < engagement < 0.7 and (price_sensitive or recent_hike or ticket_rate > 0.15)
                h *= 0.2 if persuadable else 0.97
            if rng.random() < h:
                churn_week = w
                break
            # usage fades in the weeks leading up to a cancellation decision
            if h > 0.03:
                engagement = max(0.0, engagement - 0.04)

        customers.append({
            "customer_id": cid,
            "signup_date": (START + dt.timedelta(weeks=signup_week)).isoformat(),
            "plan": plan,
            "monthly_price": round(price * (1.15 if hike_week is not None and (churn_week or WEEKS) > hike_week else 1), 2),
            "price_increase_date": (START + dt.timedelta(weeks=hike_week)).isoformat() if hike_week is not None else "",
            "churn_date": (START + dt.timedelta(weeks=churn_week)).isoformat() if churn_week is not None else "",
        })
        if offered is not None:
            campaign.append({"customer_id": cid, "offered": int(offered)})

    ev = pd.DataFrame(events, columns=["customer_id", "date", "usage", "support_tickets"])
    ev["date"] = ev["date"].astype(str)
    return {"customers": pd.DataFrame(customers), "events": ev, "campaign": pd.DataFrame(campaign)}


TRAIN_AS_OF = (START + dt.timedelta(weeks=CAMPAIGN_WEEK)).isoformat()
SCORE_AS_OF = (START + dt.timedelta(weeks=WEEKS)).isoformat()


def write(folder: pathlib.Path, n: int = 6000) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in simulate(n).items():
        df.to_csv(folder / f"{name}.csv", index=False)
