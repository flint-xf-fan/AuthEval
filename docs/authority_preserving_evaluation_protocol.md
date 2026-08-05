# Applying Authority-Preserving Evaluation

## When To Use It

Use this protocol when a medical assistant recommends a workflow action and a
later accountable process may accept, substitute, defer, or reject it. Examples
include triage, referral, follow-up, image acquisition, and intervention planning.
Ordinary model-output metrics remain necessary; this protocol adds the workflow
stage rather than replacing model evaluation.

Do not use the full decomposition for a purely descriptive task with no downstream
action claim, or when the utility of the unexecuted proposal cannot be identified.

## Minimum Log

| Column | Required | Meaning |
|---|:---:|---|
| `case_id` | Yes | Unique evaluation case identifier |
| `cluster_id` | No | Lesion, patient, or duplicate component used for uncertainty |
| `proposed_action` | Yes | Model recommendation before local review |
| `executed_action` | Yes | Action selected by the accountable workflow; the field name does not imply completed care |
| `utility_executed_local` | Yes | Declared local score of the selected action |
| `utility_proposed_local` | Full counterfactual record | Same-rule score of the proposal |
| `utility_proposed_global` | Optional | External/global proposal score, when a broader metric comparison is required |

Supply `utility_proposed_local` to identify the authority discrepancy. The external
proposal score is independent and optional; when supplied, the evaluator also reports
the broader global-to-local metric terms. Omitting the local proposal score leaves
a role-separated record, even if an external proposal score is available.

Also version the model, prompt, action vocabulary, gate/policy, utility function,
and timestamps in the surrounding study metadata. Raw private context is not an
input to the exported evaluator. Sites may export the selected action, an approved
coarse intervention reason, and aggregate statistics while retaining private
patient and workflow variables locally.

## Four-Step Checklist

1. **Separate roles.** Record what the assistant proposed and what the accountable
   workflow selected. Do not overwrite one with the other.
2. **Declare scoring.** State who defines each static action score, which reference
   state it uses, and whether the proposal score is observed, computed, or adjudicated.
3. **Respect identification.** If the declined proposal cannot be scored, report the
   selected-action score and intervention rate only. Estimate the conditional intervention
   value only with a same-rule proposal score, adjudication, randomization, or a justified
   structural model. When no departures occur, the discrepancy is zero and the conditional
   departure value is undefined.
4. **Report both stages.** Give proposal score, selected-action score, intervention rate,
   available bias terms, cluster-aware intervals, and workflow/subgroup summaries.

## Worked Teledermatology Example

Illustrative only, not clinical advice: a VLM proposes urgent specialist routing
from a dermoscopy image and public metadata. A local service instead selects
expedited review because it knows pathway eligibility and appointment capacity.
The log retains both actions. If expert adjudication can score both under the same
declared local scoring rule, the full proposal-selection gap is estimable. Without that
counterfactual review, the honest report contains the selected-action score and override
rate but marks the conditional intervention value unavailable.

## Run The Evaluator

```bash
python scripts/evaluate_authority_log.py \
  --input examples/authority_log_example.csv \
  --output-dir results/authority_log_example
```

The command writes `summary.json` for reuse and `summary.md` for inspection. See
`examples/authority_log_example.csv` for a complete synthetic log.
