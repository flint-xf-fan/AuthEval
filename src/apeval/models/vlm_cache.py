"""Frozen VLM score caching utilities.

These caches contain proposal scores only. Gate proxies are generated and
versioned separately so proposal mechanisms can be compared under a fixed
authority observation. Labels, risk strata, synthetic site identifiers,
ideal actions, and utility columns are excluded from model inputs.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

RISK_LEVELS = ("very_low", "low", "moderate", "high")
RISK_TO_SCORE = {"very_low": 0.10, "low": 0.38, "moderate": 0.64, "high": 0.90}
TRIAGE_DESCRIPTIONS = {
    "very_low": "no immediate escalation; routine observation with safety-netting",
    "low": "routine, non-urgent dermatology review",
    "moderate": "expedited dermatology assessment; a clinician may consider biopsy",
    "high": "urgent specialist assessment through a suspected skin-cancer pathway",
}
MC_PROMPT_SPECS = {
    "isic_triage_abcd_v2": {
        "labels": dict(zip(RISK_LEVELS, ("A", "B", "C", "D"))),
        "visual_focus": False,
    },
    "isic_triage_dcba_v2": {
        "labels": dict(zip(RISK_LEVELS, ("D", "C", "B", "A"))),
        "visual_focus": False,
    },
    "isic_triage_visual_abcd_v3": {
        "labels": dict(zip(RISK_LEVELS, ("A", "B", "C", "D"))),
        "visual_focus": True,
    },
}
MEDGEMMA_PROMPT_SPECS = {
    "medgemma_isic_actions_v1": tuple(RISK_LEVELS),
    "medgemma_isic_actions_reversed_v1": tuple(reversed(RISK_LEVELS)),
    "medgemma_isic_visual_actions_v2": tuple(RISK_LEVELS),
    "medgemma_isic_risk_v3": tuple(RISK_LEVELS),
}
FORBIDDEN_SCORE_INPUTS = {
    "y", "label", "labels", "diagnosis", "dx", "target", "risk_stratum",
    "site_id", "site_profile", "ideal_action_level", "r_true", "ell_star", "utility",
}
PUBLIC_COLUMNS = ("image_id", "image_path", "age", "sex", "anatomic_site")
DEFAULT_STRATA_COLUMNS = ("risk_stratum", "site_profile")


class BackendUnavailable(RuntimeError):
    """Raised when a configured VLM backend cannot be loaded in this runtime."""


@dataclass(frozen=True)
class ScoreResult:
    probs: dict[str, float]
    s_assistant: float
    model_name: str
    backend: str
    prompt_id: str
    cache_status: str
    raw_response: str = ""
    parse_status: str = "not_applicable"


@dataclass(frozen=True)
class ParsedRisk:
    probs: dict[str, float]
    status: str


def public_view(row: Mapping) -> dict:
    """Return only public inputs allowed for assistant/gate scoring."""
    leaked = FORBIDDEN_SCORE_INPUTS & set(row)
    if leaked:
        # The manifest may contain labels, but the scoring row passed into a
        # backend must not. This turns accidental leakage into a testable error.
        raise ValueError(f"Forbidden scoring columns present: {sorted(leaked)}")
    return {c: row.get(c) for c in PUBLIC_COLUMNS if c in row}


def row_to_public_view(row: Mapping) -> dict:
    return {c: row.get(c) for c in PUBLIC_COLUMNS if c in row}


def assert_no_forbidden_backend_inputs(row: Mapping) -> None:
    leaked = FORBIDDEN_SCORE_INPUTS & set(row)
    if leaked:
        raise ValueError(f"Forbidden scoring columns present: {sorted(leaked)}")


def _stable_unit(*parts: object) -> float:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12 - 1)


def select_manifest_subset(
    df: pd.DataFrame,
    limit: int | None,
    *,
    method: str = "head",
    seed: int = 20260706,
    strata_columns: tuple[str, ...] = DEFAULT_STRATA_COLUMNS,
) -> pd.DataFrame:
    """Select a deterministic manifest subset before scoring.

    Stratification is for smoke/QC subset construction only. Backend scoring
    still receives public columns only.
    """
    if limit is None or limit >= len(df):
        return df.copy()
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return df.head(0).copy()
    if method == "head":
        return df.head(limit).copy()
    if method != "stratified":
        raise ValueError(f"Unknown sampling method: {method}")

    present = [c for c in strata_columns if c in df.columns]
    if not present:
        return df.assign(
            _sample_key=df["image_id"].map(lambda x: _stable_unit(seed, x))
        ).sort_values("_sample_key").drop(columns=["_sample_key"]).head(limit).copy()

    work = df.copy()
    work["_stratum"] = work[present].fillna("__missing__").astype(str).agg("|".join, axis=1)
    work["_sample_key"] = work["image_id"].map(lambda x: _stable_unit(seed, x))
    work = work.sort_values(["_stratum", "_sample_key", "image_id"]).copy()
    groups = {key: grp.drop(columns=["_stratum", "_sample_key"]) for key, grp in work.groupby("_stratum", sort=True)}
    selected = []
    positions = {key: 0 for key in groups}
    keys = sorted(groups, key=lambda k: (-len(groups[k]), k))
    while len(selected) < limit:
        advanced = False
        for key in keys:
            pos = positions[key]
            group = groups[key]
            if pos >= len(group):
                continue
            selected.append(group.iloc[[pos]])
            positions[key] = pos + 1
            advanced = True
            if len(selected) == limit:
                break
        if not advanced:
            break
    return pd.concat(selected, ignore_index=True) if selected else df.head(0).copy()


def _softmax(logits: list[float]) -> dict[str, float]:
    arr = np.asarray(logits, dtype=float)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    probs = exp / exp.sum()
    return {k: float(v) for k, v in zip(RISK_LEVELS, probs)}


def probs_to_score(probs: Mapping[str, float]) -> float:
    return float(sum(float(probs[k]) * RISK_TO_SCORE[k] for k in RISK_LEVELS))


def normalize_probs(probs: Mapping[str, float]) -> dict[str, float]:
    vals = {k: max(0.0, float(probs.get(k, 0.0))) for k in RISK_LEVELS}
    total = sum(vals.values())
    if total <= 0:
        return {k: 1.0 / len(RISK_LEVELS) for k in RISK_LEVELS}
    return {k: v / total for k, v in vals.items()}


def parse_risk_distribution_with_status(text: str) -> ParsedRisk:
    """Parse a generated answer into four risk probabilities.

    Supports either explicit numeric fields such as "high: 0.7" or a single
    class mention. This is deterministic and conservative for smoke caching.
    """
    lower = text.lower().replace("-", "_")
    explicit = {}
    level_patterns = {
        "very_low": r"very[\s_]*low",
        "low": r"(?<!very\s)(?<!very_)low",
        "moderate": r"moderate",
        "high": r"high",
    }
    for level in RISK_LEVELS:
        pattern = rf"\b{level_patterns[level]}\b\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%)?"
        match = re.search(pattern, lower)
        if match:
            value = float(match.group(1))
            if match.group(2) == "%" or value > 1.0:
                value /= 100.0
            explicit[level] = value
    if explicit:
        return ParsedRisk(normalize_probs(explicit), "explicit_probs")

    aliases = {
        "very_low": (r"\bvery[\s_]*low\b", r"\bbenign[\s_]*no[\s_]*action\b"),
        "low": (r"(?<!very\s)(?<!very_)\blow\b", r"\broutine\b", r"\bnon[\s_]*urgent\b"),
        "moderate": (r"\bmoderate\b", r"\brefer\b", r"\bbiopsy\b"),
        "high": (r"\bhigh\b", r"\burgent\b", r"\bmelanoma\b"),
    }
    hits = [level for level, patterns in aliases.items() if any(re.search(p, lower) for p in patterns)]
    if hits:
        chosen = hits[-1]
        probs = normalize_probs({k: (0.85 if k == chosen else 0.05) for k in RISK_LEVELS})
        return ParsedRisk(probs, f"class_mention:{chosen}")
    return ParsedRisk({k: 1.0 / len(RISK_LEVELS) for k in RISK_LEVELS}, "unparsed_uniform")


def parse_risk_distribution(text: str) -> dict[str, float]:
    return parse_risk_distribution_with_status(text).probs


def build_prompt(row: Mapping, prompt_id: str) -> str:
    age = row.get("age")
    sex = row.get("sex")
    site = row.get("anatomic_site")
    return (
        "You are a frozen medical vision-language assistant used for research, "
        "not clinical care. Inspect the dermoscopy image and public metadata only. "
        "Return probabilities for management-risk levels: very_low, low, moderate, high. "
        "Definitions: very_low = reassure/no urgent management; low = routine follow-up; "
        "moderate = dermatology referral or biopsy consideration; high = urgent escalation. "
        f"Public metadata: age={age}, sex={sex}, anatomic_site={site}. "
        "Answer exactly as: very_low: <p>, low: <p>, moderate: <p>, high: <p>."
    )


def mc_candidate_text(prompt_id: str) -> dict[str, str]:
    """Return candidate continuations indexed by semantic risk level."""
    spec = MC_PROMPT_SPECS.get(prompt_id)
    if spec is None:
        return {level: f" {level}" for level in RISK_LEVELS}
    return {level: f" {spec['labels'][level]}" for level in RISK_LEVELS}


def build_mc_prompt(row: Mapping, prompt_id: str) -> str:
    age = row.get("age")
    sex = row.get("sex")
    site = row.get("anatomic_site")
    spec = MC_PROMPT_SPECS.get(prompt_id)
    if spec is not None:
        label_to_level = {label: level for level, label in spec["labels"].items()}
        options = " ".join(
            f"{label} = {TRIAGE_DESCRIPTIONS[label_to_level[label]]};"
            for label in sorted(label_to_level)
        )
        visual_instruction = ""
        if spec["visual_focus"]:
            visual_instruction = (
                "Assess visible dermoscopic concern, including asymmetry, border "
                "irregularity, and colour or structure variation. Evolution, symptoms, "
                "palpation, and patient preference are unavailable; do not assume them. "
            )
        return (
            "You are a frozen medical vision-language assistant in a research simulation, "
            "not a source of patient-specific clinical advice. Inspect the dermoscopy "
            "image and public metadata only. "
            f"{visual_instruction}Choose one stylized teledermatology triage-intensity "
            "category. The eventual action remains subject to local clinician review. "
            f"Options: {options} "
            f"Public metadata: age={age}, sex={sex}, anatomic_site={site}. "
            "Answer with exactly one option letter:"
        )
    return (
        "You are a frozen medical vision-language assistant used for research, "
        "not clinical care. Inspect the dermoscopy image and public metadata only. "
        "Choose the most appropriate management-risk level. "
        "Options: very_low, low, moderate, high. "
        "Definitions: very_low = reassure/no urgent management; low = routine follow-up; "
        "moderate = dermatology referral or biopsy consideration; high = urgent escalation. "
        f"Public metadata: age={age}, sex={sex}, anatomic_site={site}. "
        "Answer with exactly one option:"
    )


def medgemma_candidate_text(
    candidate_text: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Full semantic continuations used instead of selector letters."""
    if candidate_text is not None:
        missing = sorted(set(RISK_LEVELS) - set(candidate_text))
        extra = sorted(set(candidate_text) - set(RISK_LEVELS))
        if missing or extra:
            raise ValueError(
                f"MedGemma candidates must match {RISK_LEVELS}; "
                f"missing={missing}, extra={extra}"
            )
        return {level: str(candidate_text[level]) for level in RISK_LEVELS}
    return {level: f" {TRIAGE_DESCRIPTIONS[level]}" for level in RISK_LEVELS}


def build_medgemma_prompt(
    row: Mapping,
    prompt_id: str,
    prompt_text: str | None = None,
) -> str:
    age = row.get("age")
    sex = row.get("sex")
    site = row.get("anatomic_site")
    if prompt_text is not None:
        if prompt_id not in MEDGEMMA_PROMPT_SPECS:
            raise ValueError(f"Unknown MedGemma prompt_id: {prompt_id}")
        return str(prompt_text).format(
            age=age,
            sex=sex,
            anatomic_site=site,
        )
    order = MEDGEMMA_PROMPT_SPECS.get(prompt_id)
    if order is None:
        raise ValueError(f"Unknown MedGemma prompt_id: {prompt_id}")
    visual_instruction = ""
    if prompt_id == "medgemma_isic_visual_actions_v2":
        visual_instruction = (
            "First assess only visible dermoscopic concern such as asymmetry, border "
            "irregularity, and colour or structure variation. Do not assume evolution, "
            "symptoms, palpation findings, or patient preference. "
        )
    options = " | ".join(TRIAGE_DESCRIPTIONS[level] for level in order)
    return (
        "This is a research simulation, not patient-specific clinical advice. "
        "Inspect the dermoscopy image and public metadata only. "
        f"{visual_instruction}Select exactly one stylized teledermatology "
        "management-intensity category. Local clinicians retain authority over any "
        "eventual action. "
        f"Categories: {options}. "
        f"Public metadata: age={age}, sex={sex}, anatomic_site={site}. "
        "Selected category:"
    )


class BaseScoreBackend:
    backend = "base"

    def __init__(self, cfg: Mapping):
        self.cfg = cfg
        self.prompt_id = cfg.get("prompt_id", "isic_risk_mc_v1")
        self.model_name = cfg.get("model_name", self.backend)
        self.revision = cfg.get("revision")
        self.seed = int(cfg.get("seed", 20260706))

    def score_public(self, row: Mapping) -> ScoreResult:
        raise NotImplementedError


class FallbackImageMetadataScoreBackend(BaseScoreBackend):
    """Deterministic public-image/metadata fallback, marked honestly in output."""

    backend = "fallback_image_metadata_score"

    def __init__(self, cfg: Mapping):
        super().__init__(cfg)
        self.model_name = cfg.get("fallback_model_name", self.backend)

    def score_public(self, row: Mapping) -> ScoreResult:
        assert_no_forbidden_backend_inputs(row)
        age = pd.to_numeric(pd.Series([row.get("age")]), errors="coerce").iloc[0]
        age = 55.0 if pd.isna(age) else float(age)
        anatomic = str(row.get("anatomic_site", "")).lower()
        sex = str(row.get("sex", "")).lower()
        image_path = Path(str(row.get("image_path", "")))
        image_signal = _stable_unit(row.get("image_id", ""), image_path.name, self.seed)
        size_signal = 0.0
        if image_path.exists():
            size_signal = math.log1p(image_path.stat().st_size) / 12.0

        age_signal = np.clip((age - 30.0) / 60.0, 0.0, 1.0)
        site_signal = 0.15 if any(s in anatomic for s in ("torso", "head", "neck")) else 0.0
        sex_signal = 0.05 if sex == "male" else 0.0
        public_risk = np.clip(0.20 + 0.30 * age_signal + site_signal + sex_signal
                              + 0.20 * image_signal + 0.08 * size_signal, 0.02, 0.98)
        logits = [
            1.6 - 3.0 * public_risk,
            1.2 - 1.1 * abs(public_risk - 0.35),
            1.0 - 1.3 * abs(public_risk - 0.62),
            -0.2 + 3.0 * public_risk,
        ]
        probs = _softmax(logits)
        s_assistant = probs_to_score(probs)
        return ScoreResult(
            probs=probs,
            s_assistant=s_assistant,
            model_name=self.model_name,
            backend=self.backend,
            prompt_id=self.prompt_id,
            cache_status="fallback",
            raw_response=f"fallback_public_risk={public_risk:.6f}",
            parse_status="fallback_probs",
        )


class LlavaMedMcLogitsBackend(BaseScoreBackend):
    backend = "llava_med_mc_logits"

    def __init__(self, cfg: Mapping):
        super().__init__(cfg)
        if cfg.get("loader", "native_llava") != "native_llava":
            raise BackendUnavailable("LLaVA-Med MC logits backend requires loader=native_llava.")
        self.native = LlavaMedGenerateParseBackend(cfg)

    def score_public(self, row: Mapping) -> ScoreResult:
        assert_no_forbidden_backend_inputs(row)
        n = self.native
        qs = build_mc_prompt(row, self.prompt_id)
        if n.model.config.mm_use_im_start_end:
            qs = n.DEFAULT_IM_START_TOKEN + n.DEFAULT_IMAGE_TOKEN + n.DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = n.DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = n.conv_templates[n.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt_ids = n.tokenizer_image_token(
            prompt, n.tokenizer, n.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        image = n.Image.open(row["image_path"]).convert("RGB")
        image_tensor = n.process_images([image], n.image_processor, n.model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).half().cuda()

        candidate_text = mc_candidate_text(self.prompt_id)
        logits = []
        raw_parts = []
        with n.torch.inference_mode():
            for level in RISK_LEVELS:
                candidate = candidate_text[level]
                cand_ids = n.tokenizer(
                    candidate, add_special_tokens=False, return_tensors="pt"
                ).input_ids[0]
                full_ids = n.torch.cat([prompt_ids, cand_ids]).unsqueeze(0).cuda()
                outputs = n.model(input_ids=full_ids, images=image_tensor)
                start = int(prompt_ids.shape[0])
                token_logits = outputs.logits[0, start - 1:start - 1 + cand_ids.shape[0], :]
                log_probs = n.torch.log_softmax(token_logits, dim=-1)
                token_scores = log_probs.gather(
                    1, cand_ids.to(log_probs.device).unsqueeze(1)
                ).squeeze(1)
                score = float(token_scores.mean().item())
                logits.append(score)
                raw_parts.append(f"{level}[{candidate.strip()}]={score:.6f}")

        probs = _softmax(logits)
        s_assistant = probs_to_score(probs)
        return ScoreResult(
            probs=probs,
            s_assistant=s_assistant,
            model_name=self.model_name,
            backend=self.backend,
            prompt_id=self.prompt_id,
            cache_status="ok",
            raw_response="mc_logprobs:" + ",".join(raw_parts),
            parse_status="mc_logits",
        )


class MedGemmaMcLogitsBackend(BaseScoreBackend):
    """Frozen MedGemma scorer over full semantic action continuations."""

    backend = "medgemma_mc_logits"

    def __init__(self, cfg: Mapping):
        super().__init__(cfg)
        if not bool(cfg.get("local_files_only", True)):
            raise BackendUnavailable("MedGemma backend requires local_files_only=true")
        self.prompt_text = cfg.get("prompt_text")
        self.candidates = medgemma_candidate_text(cfg.get("candidate_text"))
        self._init_runtime(cfg)

    def _init_runtime(self, cfg: Mapping) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise BackendUnavailable(f"Missing MedGemma runtime dependencies: {exc}") from exc

        if cfg.get("device", "cuda") == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailable("MedGemma requested CUDA but torch.cuda.is_available() is false")
        dtype_name = str(cfg.get("torch_dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise BackendUnavailable(f"Unknown torch dtype: {dtype_name}")
        try:
            processor_kwargs = {
                "local_files_only": True,
                "use_fast": bool(cfg.get("use_fast", False)),
            }
            if self.revision:
                processor_kwargs["revision"] = self.revision
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                **processor_kwargs,
            )
            model_kwargs = {
                "local_files_only": True,
                "torch_dtype": dtype,
            }
            if self.revision:
                model_kwargs["revision"] = self.revision
            device_map = cfg.get("device_map", "auto")
            if device_map:
                model_kwargs["device_map"] = device_map
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            self.model.eval()
        except Exception as exc:
            raise BackendUnavailable(f"Could not load local MedGemma model: {exc}") from exc
        self.torch = torch
        self.Image = Image
        self.device = next(self.model.parameters()).device

    def _base_inputs(self, row: Mapping):
        image = self.Image.open(row["image_path"]).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": build_medgemma_prompt(
                        row,
                        self.prompt_id,
                        prompt_text=self.prompt_text,
                    ),
                },
            ],
        }]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def _candidate_scores(self, base_inputs, candidates: Mapping[str, str]) -> list[float]:
        torch = self.torch
        prompt_ids = base_inputs["input_ids"]
        prompt_length = int(prompt_ids.shape[1])
        candidate_ids = [
            self.processor.tokenizer(
                candidates[level],
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids
            for level in RISK_LEVELS
        ]
        max_candidate_length = max(int(ids.shape[1]) for ids in candidate_ids)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id
        batch_size = len(candidate_ids)
        full_length = prompt_length + max_candidate_length
        input_ids = torch.full(
            (batch_size, full_length),
            int(pad_id),
            dtype=prompt_ids.dtype,
        )
        attention_mask = torch.zeros_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        for index, ids in enumerate(candidate_ids):
            candidate_length = int(ids.shape[1])
            input_ids[index, :prompt_length] = prompt_ids[0]
            input_ids[index, prompt_length:prompt_length + candidate_length] = ids[0]
            attention_mask[index, :prompt_length + candidate_length] = 1
            if "token_type_ids" in base_inputs:
                token_type_ids[index, :prompt_length] = base_inputs["token_type_ids"][0]

        pixel_values = base_inputs["pixel_values"].repeat(batch_size, 1, 1, 1)
        model_inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "pixel_values": pixel_values.to(self.device, dtype=self.model.dtype),
        }
        if "token_type_ids" in base_inputs:
            model_inputs["token_type_ids"] = token_type_ids.to(self.device)
        with torch.inference_mode():
            outputs = self.model(**model_inputs, use_cache=False)

        scores = []
        for level in RISK_LEVELS:
            index = RISK_LEVELS.index(level)
            ids = candidate_ids[index]
            token_logits = outputs.logits[
                index,
                prompt_length - 1:prompt_length - 1 + ids.shape[1],
                :,
            ]
            log_probs = torch.log_softmax(token_logits, dim=-1)
            token_scores = log_probs.gather(
                1,
                ids[0].to(log_probs.device).unsqueeze(1),
            ).squeeze(1)
            scores.append(float(token_scores.mean().item()))
        return scores

    def score_public(self, row: Mapping) -> ScoreResult:
        assert_no_forbidden_backend_inputs(row)
        candidates = getattr(self, "candidates", medgemma_candidate_text())
        scores = self._candidate_scores(self._base_inputs(row), candidates)
        probs = _softmax(scores)
        raw = ",".join(
            f"{level}[{candidates[level].strip()}]={score:.6f}"
            for level, score in zip(RISK_LEVELS, scores)
        )
        return ScoreResult(
            probs=probs,
            s_assistant=probs_to_score(probs),
            model_name=self.model_name,
            backend=self.backend,
            prompt_id=self.prompt_id,
            cache_status="ok",
            raw_response="semantic_logprobs:" + raw,
            parse_status="semantic_mc_logits",
        )


class LlavaMedGenerateParseBackend(BaseScoreBackend):
    backend = "llava_med_generate_parse"

    def __init__(self, cfg: Mapping):
        super().__init__(cfg)
        if cfg.get("loader", "native_llava") == "native_llava":
            try:
                self._init_native_llava(cfg)
                self._score_impl = self._score_native_llava
                return
            except BackendUnavailable:
                raise
            except Exception as exc:
                raise BackendUnavailable(f"Could not load native LLaVA-Med backend: {exc}") from exc

        self._init_transformers_pipeline(cfg)
        self._score_impl = self._score_transformers_pipeline

    def _init_native_llava(self, cfg: Mapping):
        try:
            from threading import Thread

            import torch
            from PIL import Image
            from transformers import TextIteratorStreamer
            from llava.constants import (
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_END_TOKEN,
                DEFAULT_IM_START_TOKEN,
                IMAGE_TOKEN_INDEX,
            )
            from llava.conversation import SeparatorStyle, conv_templates
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import (
                KeywordsStoppingCriteria,
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
            )
            from llava.utils import disable_torch_init
        except Exception as exc:
            raise BackendUnavailable(f"Missing native LLaVA-Med runtime dependencies: {exc}") from exc

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_name)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            self.model_name,
            cfg.get("model_base"),
            model_name,
            device=cfg.get("device", "cuda"),
        )
        model.eval()
        self.torch = torch
        self.Thread = Thread
        self.TextIteratorStreamer = TextIteratorStreamer
        self.Image = Image
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_mode = cfg.get("conv_mode", "mistral_instruct")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 64))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.top_p = cfg.get("top_p")
        self.num_beams = int(cfg.get("num_beams", 1))

    def _init_transformers_pipeline(self, cfg: Mapping):
        try:
            import torch
            from PIL import Image
            from transformers import pipeline
        except Exception as exc:
            raise BackendUnavailable(f"Missing VLM runtime dependencies: {exc}") from exc
        try:
            dtype_name = cfg.get("torch_dtype", "auto")
            dtype = getattr(torch, dtype_name) if dtype_name != "auto" else "auto"
            pipe_kwargs = {
                "model": self.model_name,
                "torch_dtype": dtype,
                "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
            }
            if cfg.get("device_map"):
                pipe_kwargs["device_map"] = cfg.get("device_map")
            elif cfg.get("pipeline_device") is not None:
                pipe_kwargs["device"] = int(cfg.get("pipeline_device"))
            self.pipe = pipeline("image-text-to-text", **pipe_kwargs)
            self.Image = Image
            self.max_new_tokens = int(cfg.get("max_new_tokens", 64))
        except Exception as exc:
            raise BackendUnavailable(f"Could not load {self.model_name} with Transformers pipeline: {exc}") from exc

    def score_public(self, row: Mapping) -> ScoreResult:
        return self._score_impl(row)

    def generate_public_text(self, row: Mapping, instruction: str) -> str:
        """Generate text from public image/metadata input with the native backend."""
        assert_no_forbidden_backend_inputs(row)
        if not hasattr(self, "model"):
            raise BackendUnavailable("Direct public generation requires loader=native_llava")
        qs = instruction
        if self.model.config.mm_use_im_start_end:
            qs = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = self.DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenizer_image_token(
            prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        image = self.Image.open(row["image_path"]).convert("RGB")
        image_tensor = self.process_images([image], self.image_processor, self.model.config)[0]
        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        generate_kwargs = {
            "inputs": input_ids,
            "attention_mask": self.torch.ones_like(input_ids),
            "images": image_tensor.unsqueeze(0).half().cuda(),
            "do_sample": self.temperature > 0,
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            generate_kwargs["top_p"] = self.top_p
        with self.torch.inference_mode():
            output_ids = self.model.generate(**generate_kwargs)
        input_length = int(input_ids.shape[1])
        if int(output_ids.shape[1]) > input_length:
            generated_ids = output_ids[:, input_length:]
        else:
            generated_ids = output_ids
        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if not text:
            text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if stop_str and text.endswith(stop_str):
            text = text[:-len(stop_str)].strip()
        return text

    def _score_transformers_pipeline(self, row: Mapping) -> ScoreResult:
        assert_no_forbidden_backend_inputs(row)
        image = self.Image.open(row["image_path"]).convert("RGB")
        prompt = build_prompt(row, self.prompt_id)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        output = self.pipe(text=messages, max_new_tokens=self.max_new_tokens)
        text = str(output[0].get("generated_text", output[0]) if isinstance(output, list) else output)
        parsed = parse_risk_distribution_with_status(text)
        probs = parsed.probs
        s_assistant = probs_to_score(probs)
        return ScoreResult(
            probs=probs,
            s_assistant=s_assistant,
            model_name=self.model_name,
            backend=self.backend,
            prompt_id=self.prompt_id,
            cache_status="ok" if parsed.status != "unparsed_uniform" else "unparsed_uniform",
            raw_response=text,
            parse_status=parsed.status,
        )

    def _score_native_llava(self, row: Mapping) -> ScoreResult:
        text = self.generate_public_text(row, build_prompt(row, self.prompt_id))
        parsed = parse_risk_distribution_with_status(text)
        probs = parsed.probs
        s_assistant = probs_to_score(probs)
        return ScoreResult(
            probs=probs,
            s_assistant=s_assistant,
            model_name=self.model_name,
            backend=self.backend,
            prompt_id=self.prompt_id,
            cache_status="ok" if parsed.status != "unparsed_uniform" else "unparsed_uniform",
            raw_response=text,
            parse_status=parsed.status,
        )


BACKENDS = {
    "medgemma_mc_logits": MedGemmaMcLogitsBackend,
    "llava_med_mc_logits": LlavaMedMcLogitsBackend,
    "llava_med_generate_parse": LlavaMedGenerateParseBackend,
    "fallback_image_metadata_score": FallbackImageMetadataScoreBackend,
}


def build_backend(cfg: Mapping) -> BaseScoreBackend:
    preference = cfg.get("preference", ["llava_med_generate_parse", "fallback_image_metadata_score"])
    errors = []
    for name in preference:
        cls = BACKENDS.get(name)
        if cls is None:
            errors.append(f"{name}: unknown backend")
            continue
        try:
            return cls(cfg)
        except BackendUnavailable as exc:
            errors.append(f"{name}: {exc}")
    raise BackendUnavailable("; ".join(errors))


def score_manifest(
    df: pd.DataFrame,
    backend: BaseScoreBackend,
    seed: int,
    *,
    progress_every: int = 0,
) -> pd.DataFrame:
    rows = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        public_row = row_to_public_view(row)
        result = backend.score_public(public_row)
        rows.append({
            "image_id": row["image_id"],
            "split": row["split"],
            "site_id": row.get("site_id"),
            "site_profile": row.get("site_profile"),
            "image_path": row.get("image_path"),
            "s_assistant": round(result.s_assistant, 8),
            "p_very_low": round(result.probs["very_low"], 8),
            "p_low": round(result.probs["low"], 8),
            "p_moderate": round(result.probs["moderate"], 8),
            "p_high": round(result.probs["high"], 8),
            "model_name": result.model_name,
            "backend": result.backend,
            "prompt_id": result.prompt_id,
            "seed": seed,
            "cache_status": result.cache_status,
            "parse_status": result.parse_status,
            "raw_response": result.raw_response.replace("\r", "\\r").replace("\n", "\\n"),
        })
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            print(f"[score_manifest] backend={backend.backend} rows={idx}/{total}", flush=True)
    return pd.DataFrame(rows)
