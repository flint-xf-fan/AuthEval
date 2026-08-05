# AuthEval: Minimal Camera-Ready Reproduction Package

This package accompanies **Authority-Preserving Evaluation of Medical
Vision-Language Assistants**. It contains the smallest executable subset needed
to:

1. evaluate a role-separated recommendation log with AuthEval;
2. prepare the public ISIC 2019 data with the frozen duplicate-aware split;
3. create the public-metadata gate proxy and frozen MedGemma likelihood cache;
4. reproduce the mixed, capacity, safety, and proposal-dependence analyses; and
5. reconstruct every released numerical result from retained case-level traces.

The code is for research evaluation. The simulated actions, local review rules,
private factors, and utilities are not clinical recommendations or validated
models of a health service.

## What Is Included

- AuthEval's generic log evaluator and worked CSV example;
- exact data-preparation, gate, utility, and cluster-bootstrap code;
- the frozen ISIC split assignment and exact-pHash grouping;
- the exact MedGemma prompt, model revision, and fitted action readout;
- compact frozen action heads, thresholds, and model metadata for the two
  proposal controls;
- final case traces, summaries, and receipts under `reference_results/`;
- scripts for full reruns and data-free verification.

The ISIC images, model weights, and generated MedGemma score caches are **not**
included. Downloading them remains subject to their respective terms.

## 1. Verify The Released Results Without Data Or A GPU

Python 3.10 or newer is required.

```bash
sha256sum -c PACKAGE_SHA256SUMS.txt

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python scripts/verify_reference_results.py
python -m pytest -q
python scripts/evaluate_authority_log.py \
  --input examples/authority_log_example.csv \
  --output-dir results/authority_log_example
```

`verify_reference_results.py` recomputes point estimates and all 5,000-draw
whole-cluster bootstrap intervals from the packaged traces. It also checks
`Delta_authority = p * delta`, the paired proposal contrast, and the recorded
output hashes. This path does not require ISIC or any model weights.

## 2. Download ISIC 2019 Yourself

Review the ISIC challenge data terms before downloading:

- Data page: <https://challenge.isic-archive.com/data/>
- 2019 challenge: <https://challenge.isic-archive.com/landing/2019/>
- Archive collection: <https://api.isic-archive.com/collections/65/>

From this package root, download the official training metadata, ground truth,
and image archive:

```bash
mkdir -p data/raw/isic2019

curl -L -C - \
  https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv \
  -o data/raw/isic2019/ISIC_2019_Training_Metadata.csv
curl -L -C - \
  https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv \
  -o data/raw/isic2019/ISIC_2019_Training_GroundTruth.csv
curl -L -C - \
  https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip \
  -o data/raw/isic2019/ISIC_2019_Training_Input.zip

unzip -q data/raw/isic2019/ISIC_2019_Training_Input.zip \
  -d data/raw/isic2019
find data/raw/isic2019/ISIC_2019_Training_Input -maxdepth 1 -name '*.jpg' | wc -l
```

The final command should report **25,331** JPEG files. The image archive is large
(approximately 9 GB compressed and 19 GB extracted), so use a filesystem with
adequate free space.

The files used for the released run have these SHA-256 checksums:

```text
d93994a8ed201d474de9a7af7e17ec30929cbc4a5220659ecbde17cbe83e9316  ISIC_2019_Training_Metadata.csv
aa88e9638fe4df9ef330dc4ba22fa4bd475c44af692fffb870c73091ab25cdcd  ISIC_2019_Training_GroundTruth.csv
5075020b720c8f1b9b7f3ff85326c55ce0435e5906c41fbd338994feef886df1  ISIC_2019_Training_Input.zip
```

## 3. Prepare The Frozen Split And Gate Proxy

Install the image-processing extra, then prepare the manifests:

```bash
python -m pip install -e ".[data]"
python scripts/prepare_isic.py --config configs/data/isic.yaml
python scripts/check_image_duplicates.py \
  --config configs/data/isic.yaml --use-final-split
python scripts/fit_gate_proxy.py --config configs/model/gate_proxy.yaml
```

Expected split sizes are **17,629 / 3,735 / 3,967** train/validation/test
images. `split_freeze.csv` fixes the assignment, while
`phash_exact_duplicate_components.csv` keeps exact pHash groups in one split.
The threshold-4 near-duplicate report is diagnostic and is retained for audit;
those broader components are not used to redefine the split.

## 4. Download And Score MedGemma

The primary proposal uses `google/medgemma-1.5-4b-it` at revision
`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. Accept the model's access terms on
Hugging Face, authenticate, and download that exact revision:

```bash
python -m pip install -e ".[vlm]"
hf auth login
hf download google/medgemma-1.5-4b-it \
  --revision 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b

python scripts/score_medgemma.py --split test --progress-every 25
```

The scorer uses only the image and declared public metadata. It is resumable:
completed rows are validated and preserved when the same command is rerun. A
CUDA GPU with sufficient memory is required by the frozen configuration. Small
floating-point differences across GPU and software stacks are possible; all
paper estimates can still be independently checked from `reference_results/`.

The package uses the validation-fitted readout stored in
`artifacts/medgemma_policy_lock.yaml`. It does not refit or select that readout on
test data.

## 5. Reproduce The Paper Analyses

After Steps 2-4:

```bash
python scripts/run_authority_analysis.py \
  --config configs/experiments/authority_analysis.yaml

python scripts/run_proposal_dependence_check.py \
  --config configs/experiments/proposal_dependence.yaml
```

The first command evaluates the same MedGemma-derived proposal under the mixed,
capacity-ceiling, and safety-floor review processes. The second keeps the cases,
gate proxy, simulated private context, utility, clusters, and capacity gate fixed
while changing the frozen proposal mechanism.

The camera-ready reference values include:

| Result | `p` | `delta` | image-weighted `Delta_authority` |
|---|---:|---:|---:|
| Mixed MedGemma readout | 0.470 | 0.797 | +0.374 [0.091, 0.663] |
| Capacity ceiling | 0.341 | 2.182 | +0.744 [0.491, 1.010] |
| Safety floor | 0.281 | -0.180 | -0.051 [-0.262, 0.158] |

Cluster-equal weighting changes the mixed estimate to
`-0.206 [-0.373, -0.037]`. These are simulator-score quantities, not patient
outcomes or estimates of real clinical practice.

## Directory Guide

```text
artifacts/          frozen readout, test receipt, and compact proposal heads
configs/            data, model, simulator, and experiment specifications
data/processed/     frozen split and duplicate-audit artifacts only
docs/               AuthEval logging protocol
examples/           complete synthetic role-separated log
reference_results/  immutable camera-ready traces, summaries, and receipts
scripts/            reproduction scripts
src/apeval/         minimal imported implementation
tests/              package-level evaluator checks
```

## Reproducibility Boundary

The retained traces reproduce the paper's controlled evidence exactly. A raw-data
rerun additionally depends on the official ISIC release, the gated MedGemma
weights, CUDA behavior, and the specified Python dependencies. The LLaVA-Med and
CLIP controls are represented by their frozen operational action heads because
upstream model training and weight redistribution are outside this minimal
package; the matched authority analysis itself is fully rerunnable.

ISIC data and model weights are not redistributed. Users must obtain them under
the applicable dataset and model-provider terms.
