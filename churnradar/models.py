"""The four models: churn risk, time to churn, offer uplift, and per-customer reasons."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .features import label, model_columns


def _gbm(seed=0):
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=seed)


class RiskModel:
    """Probability a customer cancels within the horizon (default 90 days)."""

    def fit(self, f: pd.DataFrame) -> "RiskModel":
        self.cols = model_columns(f)
        X, y = f[self.cols], f["_churned_in_horizon"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
        self.auc = float(roc_auc_score(yte, _gbm().fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
        self.model = _gbm().fit(X, y)
        self.medians = X.median()
        self.base_rate = float(y.mean())
        return self

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(f[self.cols])[:, 1]

    def reasons(self, f: pd.DataFrame, top: int = 3) -> list[str]:
        """Which features push each customer's risk up the most?

        For each feature, swap in the typical customer's value and see how much
        the risk drops. Model-agnostic, and easy to explain to a non-technical team.
        """
        base = self.predict(f)
        drops = {}
        for col in self.cols:
            if col.startswith("plan_"):
                continue
            g = f.copy()
            g[col] = self.medians[col]
            drops[col] = base - self.predict(g)
        D = pd.DataFrame(drops, index=f.index)
        out = []
        for cid, row in D.iterrows():
            parts = []
            for col, d in row.sort_values(ascending=False).items():
                if d < 0.02 or len(parts) == top:
                    break
                v, med = f.loc[cid, col], self.medians[col]
                if not _risky_direction(col, v, med):
                    continue  # don't explain risk with something that's actually a good sign
                parts.append(_describe(col, v, med))
            out.append("; ".join(parts) if parts else "no single strong driver")
        return out


RISKY_IF_LOW = {"usage_trend", "usage_per_week_recent", "active_weeks_recent", "tenure_days"}
RISKY_IF_HIGH = {"days_since_active", "tickets_recent", "recent_price_increase"}


def _risky_direction(col, v, med) -> bool:
    if col in RISKY_IF_LOW:
        return v < med * (0.9 if col == "usage_trend" else 1)
    if col in RISKY_IF_HIGH:
        return v > med
    return True


def _describe(col, v, med) -> str:
    if col == "usage_trend":
        return f"usage {'down' if v < 1 else 'up'} {abs(1 - v):.0%} vs. their own normal"
    if col == "usage_per_week_recent":
        return f"low recent usage ({v:.0f}/week vs. typical {med:.0f})"
    if col == "days_since_active":
        return f"inactive for {v:.0f} days"
    if col == "tickets_recent":
        return f"{v:.0f} support tickets in the last 90 days"
    if col == "recent_price_increase":
        return "a recent price increase"
    if col == "tenure_days":
        return f"new customer ({v / 30:.0f} months)" if v < med else f"long-time customer ({v / 30:.0f} months)"
    if col == "active_weeks_recent":
        return f"active only {v:.0f} of the last 12 weeks"
    return f"{label(col)} ({v:g})"


class TimeToChurn:
    """Cox proportional hazards model: *when* a customer is likely to leave."""

    def fit(self, f: pd.DataFrame) -> "TimeToChurn":
        self.cols = [c for c in model_columns(f) if f[c].std() > 0]
        df = f[self.cols + ["_duration_days", "_event"]].copy()
        self.mu, self.sd = df[self.cols].mean(), df[self.cols].std()
        df[self.cols] = (df[self.cols] - self.mu) / self.sd
        self.cph = CoxPHFitter(penalizer=0.05).fit(df, "_duration_days", "_event")
        self.c_index = float(concordance_index(df["_duration_days"], -self.cph.predict_partial_hazard(df), df["_event"]))
        self.max_days = int(f["_duration_days"].max())
        return self

    def predict(self, f: pd.DataFrame) -> pd.DataFrame:
        X = (f[self.cols] - self.mu) / self.sd
        surv = self.cph.predict_survival_function(X)
        med = self.cph.predict_median(X)
        p30 = 1 - surv.iloc[(surv.index - 30).to_series().abs().argmin()]
        return pd.DataFrame({"churn_prob_30d": p30.values,
                             "median_days_to_churn": med.replace(np.inf, np.nan).values}, index=f.index)


class UpliftModel:
    """Who does the offer actually change? (two-model 'T-learner')

    Trained on a past campaign where some customers randomly got the offer.
    uplift = P(cancel | no offer) - P(cancel | offer). High uplift means persuadable.
    Near zero means they'd stay anyway, or leave anyway.
    """

    def fit(self, f: pd.DataFrame, offered: pd.Series) -> "UpliftModel":
        self.cols = model_columns(f)
        d = f.join(offered.rename("_offered"), how="inner")
        tr, te = train_test_split(d, test_size=0.3, random_state=1, stratify=d["_offered"])
        self._fit(tr)
        self.evaluation = self._evaluate(te)
        self._fit(d)
        self.n_campaign = len(d)
        return self

    @staticmethod
    def _learner(seed):
        # shallower and smoother than the risk model: uplift is a small difference
        # between two noisy probabilities, so overfitting is the main danger
        return GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                          min_samples_leaf=60, subsample=0.8, random_state=seed)

    def _fit(self, d):
        self.m_treat = self._learner(1).fit(d.loc[d._offered == 1, self.cols], d.loc[d._offered == 1, "_churned_in_horizon"])
        self.m_ctrl = self._learner(2).fit(d.loc[d._offered == 0, self.cols], d.loc[d._offered == 0, "_churned_in_horizon"])

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        X = f[self.cols]
        return self.m_ctrl.predict_proba(X)[:, 1] - self.m_treat.predict_proba(X)[:, 1]

    def _evaluate(self, te) -> dict:
        """On held-out campaign customers: does the top 30% by predicted uplift really respond more?"""
        te = te.assign(u=self.predict(te))
        top = te[te.u >= te.u.quantile(0.7)]
        rest = te[te.u < te.u.quantile(0.7)]

        def real(df):
            return float(df[df._offered == 0]._churned_in_horizon.mean() - df[df._offered == 1]._churned_in_horizon.mean())
        return {"actual_uplift_top30": real(top), "actual_uplift_rest": real(rest), "actual_uplift_all": real(te)}
