# Proposed 1: All-Events Vanilla HSTU

Self-contained HSTU trained on all 14 event types.
Tests whether non-ads signals improve ad prediction without architectural changes.

## Hypothesis
Non-ad events (browsing, search, purchase) provide useful sequential context that improves next-event prediction for ads.

## Expected Outcome
+5-15% Ad NDCG@10 over baseline, with non-ads NDCG@10 as bonus.
