# Retention summary
*Scored 2,088 active customers as of 2026-07-06. Budget: $10,000.*

## Key points
- **317 customers (15%) are likely to cancel in the next 90 days**, putting $79,340 of yearly revenue at risk ($245,951 across all customers).
- **29 have a 25%+ chance of leaving in the next 30 days.** These are the most urgent.
- **Recommended plan: send the offer to 350 customers for $9,955.** Expected result: about 44 customers kept and $14,567 in revenue saved, or **1.5x the cost**.
- Targeting the 303 *highest-risk* customers with the same budget would save only about $1,615. Many of them would leave no matter what, so the money is better spent on persuadable customers.
- **154 customers show an early warning sign** (usage down 40%+ vs. their own normal) but aren't high risk yet. Worth a check-in, not a discount.
- Segments: 516 persuadable, 292 likely lost, 463 watch, 817 safe.

## Pros and cons of the recommended plan
**Pros**
- Spends only on customers the offer is expected to change, so it skips the 292 who are likely to leave anyway and the customers who would stay without a discount.
- Expected return of 1.5x, about $12,952 more than spending the same budget on the riskiest customers.
- Every targeted customer has written reasons, so the retention team knows what to say to each one.

**Cons and risks**
- Uplift is learned from one past campaign (3,479 customers). If the new offer or season differs, the effect may too.
- 292 high-risk customers get no offer under this plan. That's a deliberate choice, but it means accepting about $75,452 of revenue at risk.
- Discounts can train customers to wait for deals. Track whether kept customers renew at full price.
- Predictions are estimates. Hold back a random 10% of targets as a control group to measure the real effect.

## How much to trust the models
- Churn risk model: AUC 0.74 on held-out customers (0.5 = coin flip, 1.0 = perfect).
- Time-to-churn model: concordance 0.70 (how often it correctly orders who leaves first).
- Uplift model, checked on held-out campaign customers: the top 30% it picked really did respond more (the offer cut their churn by 8.2 points vs. 0.6 for everyone else).

## Top 10 customers to contact

| Customer | Risk (90d) | Likely leaves in | Offer cuts risk by | Why |
|---|---|---|---|---|
| 5384 | 16% | ~147 days | 12 pts | low recent usage (0/week vs. typical 13); a recent price increase |
| 4921 | 49% | ~98 days | 26 pts | usage down 69% vs. their own normal; low recent usage (0/week vs. typical 13) |
| 2823 | 17% | > 6 months | 15 pts | usage down 18% vs. their own normal |
| 4146 | 16% | > 6 months | 15 pts | low recent usage (10/week vs. typical 13) |
| 4959 | 22% | > 6 months | 22 pts | 5 support tickets in the last 90 days; low recent usage (11/week vs. typical 13) |
| 4274 | 45% | ~91 days | 22 pts | low recent usage (2/week vs. typical 13); 4 support tickets in the last 90 days; active only 11 of the last 12 weeks |
| 845 | 17% | > 6 months | 14 pts | low recent usage (10/week vs. typical 13) |
| 395 | 37% | ~77 days | 21 pts | low recent usage (0/week vs. typical 13); usage down 40% vs. their own normal; a recent price increase |
| 3861 | 17% | > 6 months | 14 pts | usage down 18% vs. their own normal; low recent usage (10/week vs. typical 13) |
| 3941 | 40% | ~91 days | 21 pts | low recent usage (0/week vs. typical 13); usage down 36% vs. their own normal |

Full list: `customers_scored.csv`.
