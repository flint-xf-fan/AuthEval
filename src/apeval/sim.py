"""Authority-preserving evaluation — toy simulator.
INVARIANT: the gate reads only public riskproxy(x) + private z. It NEVER sees y/r_true.
Utility uses r_true (from y) only post hoc, after the executed action is chosen.
Site/patient utility weights are loaded from a frozen, disclosed config.
"""
from pathlib import Path
import numpy as np, yaml

STRATA  = ["very_low", "low", "moderate", "high"]
IDEAL   = {"very_low": 0, "low": 1, "moderate": 2, "high": 3}
SEV     = {"very_low": 0.5, "low": 1.0, "moderate": 2.0, "high": 4.0}
ACTIONS = ["a0", "a1", "a2", "a3"]
LEVEL   = {"a0": 0, "a1": 1, "a2": 2, "a3": 3}
CENTERS = {"very_low": 0.12, "low": 0.38, "moderate": 0.62, "high": 0.88}
RISK_SCORE_VECTOR = np.asarray([0.10, 0.38, 0.64, 0.90], dtype=float)

def _load_config():
    for c in [Path.cwd() / "configs/simulator/site_heterogeneity.yaml",
              Path(__file__).resolve().parents[2] / "configs/simulator/site_heterogeneity.yaml"]:
        if c.exists():
            return yaml.safe_load(c.read_text())
    raise FileNotFoundError("site_heterogeneity.yaml not found")

CFG      = _load_config()
GLOBAL_U = CFG["global_utility"]
SITES    = CFG["site_profiles"]                 # name -> {cap, conservatism, w_under_site, w_over_site, ...}
PMULT    = CFG["patient_multipliers"]
W_BAR    = {"w_under": GLOBAL_U["w_under_bar"], "w_over": GLOBAL_U["w_over_bar"]}

# ---------- utility (uses r_true; scored POST-HOC only) ----------
def utility(r_true, action, w_under, w_over):
    g = LEVEL[action] - IDEAL[r_true]
    if g < 0:   return -w_under * SEV[r_true] * (-g)
    if g > 0:   return -w_over * g
    return 0.0

def effective_weights(site, sens_pref, burden, aversion):
    s = SITES[site]
    wu = s["w_under_site"] * PMULT["sensitivity_preference"][sens_pref]["w_under_mult"]
    wo = (s["w_over_site"]
          * PMULT["sensitivity_preference"][sens_pref]["w_over_mult"]
          * PMULT["follow_up_burden"][burden]["w_over_mult"]
          * PMULT["invasiveness_aversion"][aversion]["w_over_mult"])
    return wu, wo

# ---------- public risk proxy (from s_gate ONLY) ----------
def riskproxy_bin(s_gate, thr=(0.25, 0.5, 0.75)):
    return STRATA[sum(s_gate > t for t in thr)]

def floor_level(proxy_bin, conservatism):
    base = {"very_low": 0, "low": 0, "moderate": 1, "high": 2}[proxy_bin]
    base += {"lenient": -1, "standard": 0, "strict": 1}[conservatism]
    return int(np.clip(base, 0, 3))

# ---------- assistant: score -> preferred level (via tau) -> ordered slate ----------
def preferred_level(s_assistant, tau):
    """Map one expected-risk score to the evaluated action level."""
    return int(operational_head_from_score(s_assistant, tau))


def operational_score(probs):
    """Map four ordered option probabilities to the cached scalar score."""
    arr = np.asarray(probs, dtype=float)
    if arr.shape[-1] != len(RISK_SCORE_VECTOR):
        raise ValueError(f"Expected probability vectors of length 4, got shape {arr.shape}")
    return arr @ RISK_SCORE_VECTOR


def operational_head_from_score(scores, tau):
    """Canonical proposal head used by calibration and authority evaluation."""
    score_arr = np.asarray(scores, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    if tau_arr.shape != (3,) or not np.all(np.diff(tau_arr) > 0):
        raise ValueError(f"tau must contain three strictly increasing thresholds, got {tau}")
    return (score_arr[..., None] > tau_arr).sum(axis=-1).astype(int)


def operational_head(probs, tau):
    """Canonical probability-to-action proposal consumed by the simulator."""
    return operational_head_from_score(operational_score(probs), tau)

def make_slate(pref):
    return sorted(ACTIONS, key=lambda a: (abs(LEVEL[a] - pref), -LEVEL[a]))

# ---------- gate: y-blind by construction (signature excludes any outcome) ----------
def admissible_band(regime, proxy_bin, cap, conservatism):
    floor = floor_level(proxy_bin, conservatism)
    if regime == "capacity_ceiling":
        return [a for a in ACTIONS if LEVEL[a] <= cap]
    if regime == "safety_floor":
        return [a for a in ACTIONS if LEVEL[a] >= floor]
    if regime == "mixed":
        return [a for a in ACTIONS if floor <= LEVEL[a] <= cap]
    raise ValueError(regime)

def execute(slate, band):
    for a in slate:
        if a in band:
            return a, ("none" if a == slate[0]
                      else ("down_substitution" if LEVEL[a] < LEVEL[slate[0]] else "up_substitution"))
    if band:
        a = max(band, key=lambda x: LEVEL[x]); return a, "fallback"
    # An empty band must never silently execute the proposal: that would record
    # intervention_type "none" for a case the gate actually refused, i.e. exactly the
    # authority collapse this module exists to measure. Callers repair empty bands in
    # apply_patient_overrides; reaching here means that contract was broken.
    raise ValueError(
        "empty admissible band: the gate permits no action. Callers must supply a "
        "declared local default (see apply_patient_overrides) before calling execute()."
    )

TAU = {"calibrated": (0.25, 0.50, 0.75),
       "over_escalating": (0.12, 0.30, 0.55),
       "under_escalating": (0.45, 0.72, 0.90)}

# ---------- ISIC stratum map (populates all four strata) ----------
ISIC_TO_STRATUM = {"NV": "very_low", "VASC": "very_low", "DF": "very_low",
                   "BKL": "low", "AK": "moderate", "AKIEC": "moderate",
                   "BCC": "moderate", "SCC": "high", "MEL": "high"}

def patient_override(band, proxy_bin, follow_up_burden):
    """Proxy-only, y-blind."""
    if follow_up_burden != "high" or "a1" not in band:
        return list(band)
    if proxy_bin in ("very_low", "low") and "a0" in band:
        return [a for a in band if a != "a1"]
    if proxy_bin in ("moderate", "high") and "a2" in band:
        return [a for a in band if a != "a1"]
    return list(band)

def make_cases(n=8000, seed=20260706, sigma=0.08):
    rng = np.random.default_rng(seed)
    r    = rng.choice(STRATA, size=n, p=[0.45, 0.30, 0.15, 0.10])
    site = rng.choice(list(SITES), size=n)
    sens = rng.choice(list(PMULT["sensitivity_preference"]), size=n)
    burd = rng.choice(list(PMULT["follow_up_burden"]), size=n)
    inv  = rng.choice(list(PMULT["invasiveness_aversion"]), size=n)
    ctr  = np.array([CENTERS[x] for x in r])
    s_gate      = np.clip(ctr + rng.normal(0, sigma, n), 0, 1)
    s_assistant = np.clip(ctr + rng.normal(0, sigma, n), 0, 1)
    return dict(r=r, site=site, sens=sens, burden=burd, aversion=inv,
                s_gate=s_gate, s_assistant=s_assistant)

def apply_patient_overrides(band, proxy_bin, follow_up_burden, invasiveness_aversion):
    """Proxy-only, y-blind patient override applied within the gate path.
    Never reads y/r_true/utility. Returns (band, reasons)."""
    band = list(band); reasons = []
    if invasiveness_aversion == "averse" and "a3" in band:
        band = [a for a in band if a != "a3"]; reasons.append("patient_refusal")
    if follow_up_burden == "high" and "a1" in band:
        if proxy_bin in ("very_low", "low") and "a0" in band:
            band = [a for a in band if a != "a1"]; reasons.append("patient_burden")
        elif proxy_bin in ("moderate", "high") and "a2" in band:
            band = [a for a in band if a != "a1"]; reasons.append("patient_burden")
    if not band:                       # proxy-only fallback if override empties the band
        band = ["a0"] if proxy_bin in ("very_low", "low") else ["a2"]
    return band, reasons
