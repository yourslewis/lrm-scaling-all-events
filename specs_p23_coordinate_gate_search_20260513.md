# P23 — Coordinate Search for Ad-Anchor Event-Type Gates

## Motivation

The best balanced ad-anchor treatment so far is `p20_s300_page07`:

```text
sigma_seconds = 300
PageTitle gate = 0.7
Strong event gate = 1.0
MSN gate = 0.1
OutlookSenderDomain gate = 0.0
```

It gives a better Ads/overall tradeoff than P18/P19B/P21 semantic baselines:

```text
p20_s300_page07 latest: Overall HR@10 ~= 0.6056, Ads HR@10 ~= 0.2121
p20_s300_page07 best Ads HR@10 ~= 0.2424
```

Full combinatorial search over all event-type weights is too expensive. Instead, use coordinate search: tune one gate family at a time with step size 0.1, pick the best balanced setting, then move to the next gate.

## Scoring

Primary selection score:

```text
score = 0.4 * Overall_HR@10 + 0.6 * Ads_HR@10
```

Guardrails:

```text
Overall_HR@10 >= 0.58      # avoid Ads-only collapses
Ads_HR@10 >= 0.2121        # must match/beat p20_s300_page07 latest Ads recall
```

Report all candidates, but only update the coordinate baseline if the candidate passes guardrails and improves score.

## Fixed parameters

```text
Architecture: P14 stabilized group residual
Ad-anchor max weight: 4.0
Ad-anchor sigma: 300s
Pre/post windows: +/-10 minutes
Strong Search/UET/Shopping event group default: 1.0
PageTitle default: 0.7
MSN default: 0.1
OutlookSenderDomain default: 0.0
```

## Gate groups

### Strong event group

```text
OrganicSearchQuery
EdgeSearchQuery
UET
UETShoppingView
UETShoppingCart
EdgeShoppingCart
EdgeShoppingPurchase
AbandonCart
```

### PageTitle group

```text
EdgePageTitle
ChromePageTitle
```

### MSN group

```text
MSN
```

### Outlook group

```text
OutlookSenderDomain
```

## Coordinate-search stages

### Stage 1 — PageTitle gate

Baseline: `page=0.7`.

Candidates:

```text
page = 0.5, 0.6, 0.8, 0.9
```

Keep:

```text
strong=1.0, msn=0.1, outlook=0.0
```

### Stage 2 — MSN gate

Use best PageTitle value from Stage 1.

Candidates:

```text
msn = 0.0, 0.2, 0.3, 0.4
```

### Stage 3 — Strong event gate

Use best PageTitle/MSN values.

Candidates:

```text
strong = 0.7, 0.8, 0.9
```

The current baseline `strong=1.0` remains in the comparison set.

### Stage 4 — Outlook gate

Use best previous values.

Candidates:

```text
outlook = 0.1, 0.2
```

The current baseline `outlook=0.0` remains in the comparison set.

## Early stopping

Each candidate trains up to 26k batches.

Stop early if:

```text
batch >= 12k and best Ads HR@10 < 0.145
batch >= 20k and best Ads HR@10 < 0.1744  # below P18 best
NaN/invalid metrics appear
```

Do not select final checkpoint blindly. Use validation monitor and report latest/best.

## Reporting

During training, cron should render a Telegram table with:

```text
Run
Stage
Gate values
Status
Batch
Overall HR@10
Ads HR@10
Overall NDCG@10
Ads NDCG@10
Score
Decision
```

Include baselines:

```text
P14 best
P18 best
P19B best
p20_s300_page07 selected baseline
current coordinate best
active candidates
completed candidates
```
