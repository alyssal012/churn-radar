import pathlib

import numpy as np
import pandas as pd
import pytest

from churnradar import simulate
from churnradar.features import Config, build, load
from churnradar.pipeline import choose_targets, early_warnings, run


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    folder = tmp_path_factory.mktemp("data")
    simulate.write(folder)  # full size: uplift needs enough campaign customers
    return folder


@pytest.fixture(scope="module")
def cfg():
    return Config(train_as_of=simulate.TRAIN_AS_OF, score_as_of=simulate.SCORE_AS_OF)


def test_features_use_only_the_past(demo, cfg):
    data = load(cfg, demo)
    f = build(cfg, data, cfg.train_as_of)
    # adding future events must not change the features
    later = data["events"].copy()
    later.loc[later["date"] >= cfg.train_as_of, "usage"] = 999
    g = build(cfg, {**data, "events": later}, cfg.train_as_of)
    pd.testing.assert_frame_equal(f.drop(columns=["_churned_in_horizon", "_duration_days", "_event"]),
                                  g.drop(columns=["_churned_in_horizon", "_duration_days", "_event"]))


def test_only_active_customers_are_scored(demo, cfg):
    data = load(cfg, demo)
    f = build(cfg, data, cfg.score_as_of)
    c = data["customers"].set_index("customer_id").loc[f.index]
    assert (c["churn_date"].isna() | (c["churn_date"] >= cfg.score_as_of)).all()


def test_works_with_renamed_columns_and_no_optional_data(demo, tmp_path):
    """Bring-your-own-data: different column names, no plan/tickets/campaign."""
    c = pd.read_csv(demo / "customers.csv").rename(columns={"customer_id": "acct", "monthly_price": "mrr"})
    e = pd.read_csv(demo / "events.csv").rename(columns={"customer_id": "acct", "date": "week", "usage": "logins"})
    c.drop(columns=["plan", "price_increase_date"]).to_csv(tmp_path / "c.csv", index=False)
    e.drop(columns=["support_tickets"]).to_csv(tmp_path / "e.csv", index=False)
    cfg = Config(customers_file="c.csv", events_file="e.csv", campaign_file=None, id="acct", price="mrr",
                 event_date="week", usage="logins", plan=None, price_increase_date=None, tickets=None,
                 train_as_of=simulate.TRAIN_AS_OF, score_as_of=simulate.SCORE_AS_OF)
    stats = run(cfg, tmp_path, budget=2000, out=tmp_path / "out")
    assert "No past campaign data" in stats["markdown"]
    assert (tmp_path / "out" / "customers_scored.csv").exists()


def test_budget_is_respected_and_negative_value_skipped():
    s = pd.DataFrame({"offer_cost": [10, 10, 10, 10], "net_value": [50, 30, -5, 20],
                      "value_per_dollar": [5, 3, -0.5, 2]}, index=[1, 2, 3, 4])
    t = choose_targets(s, budget=25)
    assert list(t.index) == [1, 2] and t.offer_cost.sum() <= 25
    assert 3 not in choose_targets(s, budget=1000).index


def test_early_warning_is_relative_to_own_baseline():
    f = pd.DataFrame({"_baseline_usage": [20, 20, 2], "usage_trend": [0.5, 0.9, 0.3]})
    assert early_warnings(f).tolist() == [True, False, False]  # third: baseline too small to judge


def test_full_demo(demo, cfg, tmp_path):
    stats = run(cfg, demo, budget=5000, out=tmp_path)
    md = stats["markdown"]
    assert "## Key points" in md and "**Pros**" in md and "**Cons and risks**" in md
    assert stats["spend"] <= 5000
    assert stats["auc"] > 0.65
    # the uplift model's top picks should really respond more than everyone else
    ev = stats["uplift_eval"]
    assert ev["actual_uplift_top30"] > ev["actual_uplift_rest"]
    # spending on persuadable customers should beat spending on the riskiest ones
    assert stats["saved_revenue"] > stats["naive_saved_revenue"]
    scored = pd.read_csv(tmp_path / "customers_scored.csv")
    assert (scored.uplift <= scored.churn_risk_90d + 1e-9).all()
    assert scored.loc[scored.target, "reasons"].notna().all()
