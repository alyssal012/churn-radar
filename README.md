# Churn Radar

A customer retention toolkit for any subscription business. Point it at your customer list and activity log, give it a budget, and it tells you **who is likely to leave, how soon, which customers an offer would actually change, and how to spend the budget.** It finishes with a plain-English summary of key points and the pros and cons of its recommended plan.

```
python -m churnradar demo
```

```markdown
## Key points
- 317 customers (15%) are likely to cancel in the next 90 days, putting $79,340 of yearly revenue at risk.
- 29 have a 25%+ chance of leaving in the next 30 days. These are the most urgent.
- Recommended plan: send the offer to 350 customers for $9,955. Expected result: about 44 customers
  kept and $14,567 in revenue saved, or 1.5x the cost.
- Targeting the 303 highest-risk customers with the same budget would save only about $1,615.
  Many of them would leave no matter what, so the money is better spent on persuadable customers.
- 154 customers show an early warning sign (usage down 40%+ vs. their own normal) but aren't high
  risk yet. Worth a check-in, not a discount.

## Pros and cons of the recommended plan
Pros: spends only on customers the offer is expected to change ...
Cons and risks: uplift is learned from one past campaign ...; 292 high-risk customers get no offer ...
```

Full example: [`reports/summary.md`](reports/summary.md). Per-customer output: `reports/customers_scored.csv`.

## Why this is different from a typical churn model

Most churn projects stop at "this customer has a 70% chance of leaving." That doesn't tell a business what to *do*. The riskiest customers are often the worst place to spend money: many are already gone in their heads, and no discount will change that. This toolkit answers the questions a retention team actually has:

| Question | Feature | Method |
|---|---|---|
| Who's likely to leave? | **Churn risk** (next 90 days) | Gradient-boosted classifier |
| How soon? | **Time to churn** | Cox proportional hazards survival model |
| Will an offer change their mind? | **Uplift**: persuadable vs. lost vs. safe | Two-model "T-learner" trained on a past randomized campaign |
| Who do we target with $X? | **Budget optimizer** | Rank by expected revenue saved per dollar, skip negative-value offers |
| Why is this customer at risk? | **Plain-English reasons** per customer | Swap each feature to the typical value and measure the drop in risk |
| Who's quietly fading? | **Early warnings** | Usage vs. the customer's *own* baseline, not everyone else's |
| What should leadership know? | **Summary with key points + pros/cons** | Generated from the results, with every number traceable |

The demo shows the main point. Spending $10,000 on the **persuadable** customers saves about **9x more revenue** than spending it on the **highest-risk** customers.

## How much to trust it

The summary always reports model quality so nobody over-trusts it:

- **Churn risk:** AUC 0.74 on held-out customers.
- **Time to churn:** concordance 0.70.
- **Uplift:** checked on held-out campaign customers. The top 30% the model picked cut their churn by 8.2 points when given the offer, compared with 0.6 points for everyone else, so the targeting is real. (My first version failed this check because the uplift model was overfitting noise. I fixed it by using shallower, more regularized models and capping uplift at the customer's total risk.)

## Use your own data

You need two CSVs. The column names can be anything; you map them in a config file.

- **customers:** ID, signup date, monthly price, churn date (blank if active). *Optional:* plan, price increase date.
- **events:** ID, date, usage count (logins, orders, sessions...). *Optional:* support tickets.
- **campaign** (optional): ID and whether they got an offer in a past *randomized* campaign. Without it, uplift is a rough assumption, and the summary tells you to run a small test first.

```bash
python -m churnradar run path/to/data examples/config.json --budget 10000 --out reports/
```

`train_as_of` is a past date used to learn patterns (the tool looks at what happened in the 90 days after it). `score_as_of` is today. Features only use data from before each date, and a test checks that future data can't leak in.

## About the demo data

The demo company is **simulated** (`churnradar/simulate.py`): 6,000 subscribers across three plans over 18 months. It includes usage that fades before people cancel, price increases, support problems, and a past campaign where half of customers randomly got a one-month discount that only worked on some of them. Real customer data is private, so the simulation stands in for it. The numbers above describe the demo company, not a real business.

## Run it

```bash
pip install -r requirements.txt
python -m churnradar demo      # about 10 seconds
pytest -q
```

## Stack

Python, pandas, scikit-learn, lifelines (survival analysis), pytest
