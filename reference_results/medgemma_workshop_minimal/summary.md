# Workshop-Minimal MedGemma Authority Result

## What Is Measured

Each ISIC image is one benchmark instance. Images in the same frozen cluster share 
the same simulated authority context, and uncertainty resamples whole clusters.
The fixed zero-bias MedGemma-derived policy is scored before and after one fixed 
simulated local authority process under the same declared local utility.

## Primary Image-Weighted Estimate

- Images / clusters: `3967` / `2090`
- Proposal score `J_auth`: `-2.938475`
- Executed score `J`: `-3.312955`
- Intervention rate `p`: `0.470129`
- Conditional score contrast `delta`: `0.796547`
- Authority gap `Delta_authority`: `0.374480`
- Cluster 95% CI: `[0.091313, 0.662679]`
- Identity residual `Delta_authority - p*delta`: `0.000e+00`

## Cluster-Equal Sensitivity

- Authority gap: `-0.205630`
- Cluster 95% CI: `[-0.373466, -0.037471]`

## Claim Boundary

This is a post-hoc controlled simulator analysis. It establishes a signed score 
discrepancy under declared image weighting, not clinical benefit, harm, competence, 
policy reversal, prevalence, or deployment validity. The cluster-equal sensitivity
is retained because the sign depends on the declared evaluation unit.
