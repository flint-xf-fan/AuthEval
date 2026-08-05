# Proposal-dependence of the authority gap

Gate held fixed: `capacity_ceiling` / `calibrated` / frozen `metadata` gate proxy.
All arms share **one canonical cluster-shared authority context**
(`build_authority_context`, seed 20260706): same cases, same private context,
same clusters, same local utility. Only the recommender varies.

Whole-cluster bootstrap: 5000 replicates, seed 20260711, quantiles 0.025/0.975.

| proposal | test bacc | p | delta | image-weighted | cluster-equal |
|---|---:|---:|---:|---|---|
| LLaVA-Med head | 0.249 | 0.163 | +0.08 | +0.014 [-0.056, +0.086] | -0.096 [-0.156, -0.037] |
| MedGemma readout | 0.420 | 0.341 | +2.18 | +0.744 [+0.491, +1.010] | +0.056 [-0.107, +0.218] |
| CLIP probe | 0.546 | 0.276 | +2.99 | +0.824 [+0.621, +1.033] | +0.337 [+0.256, +0.422] |

## Paired contrast (identical clusters in every replicate)

| unit | probe minus LLaVA | 95% CI | excludes zero |
|---|---|---|---|
| image_weighted | +0.810 | [+0.629, +0.999] | True |
| cluster_equal | +0.433 | [+0.345, +0.524] | True |

## Reading

Under one identical authority process, the signed evaluation error varies with the
recommender being evaluated. The mechanism is the ceiling's asymmetry: a capacity
ceiling substitutes predominantly downwards, so a recommender that rarely escalates is rarely
bound by it. This is a statement about escalation propensity under this gate, not an
identified causal effect of competence: the three systems differ in architecture,
readout, and action distribution, and n=3 is an ordering, not a competence ladder.
