# Filters + scoring + ranking. Uses precomputed job embeddings; embeds the resume only.

import numpy as np
from typing import Dict, Any, List, Tuple
from src.config import openai_client, EMBED_MODEL, EMBED_DIM

def _embed_batch_resume(texts: List[str], model: str = EMBED_MODEL) -> List[np.ndarray]:
    client = openai_client()
    clean = [(t or "").strip() for t in texts]
    if not any(clean):
        return [np.zeros(EMBED_DIM, dtype=np.float32) for _ in texts]
    resp = client.embeddings.create(model=model, input=clean)
    out: List[np.ndarray] = []
    di = 0
    for t in clean:
        if not t:
            out.append(np.zeros(EMBED_DIM, dtype=np.float32))
        else:
            out.append(np.array(resp.data[di].embedding, dtype=np.float32))
            di += 1
    return out

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _jaccard(a: List[str], b: List[str]) -> float:
    A = {s.strip().lower() for s in (a or []) if isinstance(s, str) and s.strip()}
    B = {s.strip().lower() for s in (b or []) if isinstance(s, str) and s.strip()}
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

# --- Degree policy: BS/MS qualifies unless PhD-only explicitly required ---
def _resume_has_phd(edu_list):
    text = " ".join(edu_list or []).lower()
    return ("phd" in text) or ("ph.d" in text) or ("doctoral" in text) or ("doctorate" in text) or ("doctor " in text)

def _norm_degree_token(s: str):
    if not isinstance(s, str): return None
    t = s.strip().lower()
    if "phd" in t or "ph.d" in t or "doctor" in t: return "phd"
    if "master" in t or "m.s" in t or "msc" in t:  return "masters"
    if "bachelor" in t or "b.s" in t or "bsc" in t: return "bachelors"
    return None

def _education_ok_per_rule(required_edu_list, resume_edu_list) -> bool:
    req_norm = {_norm_degree_token(e) for e in (required_edu_list or [])}
    req_norm.discard(None)
    phd_only = (req_norm == {"phd"})
    if phd_only:
        return _resume_has_phd(resume_edu_list)
    return True

def smart_filter(job_pre: Dict[str,Any], resume_ie: Dict[str,Any]) -> Tuple[bool, Dict[str,Any]]:
    job_ie = job_pre.get("llm_ie") or {}

    # years
    need = ((job_ie.get("required_experience") or {}).get("years_min"))
    have = ((resume_ie.get("experience") or {}).get("years_overall"))
    if isinstance(need, float): need = int(need)
    if isinstance(have, float): have = int(have)
    years_ok = True if (need is None or have is None) else (have >= need)

    # education: explicit + inferred (PhD-only rule)
    req_edu = (job_ie.get("required_education") or []) + (job_pre.get("required_education_inferred") or [])
    edu_ok = _education_ok_per_rule(req_edu, resume_ie.get("education") or [])

    # region overlap (soft pass: filter only when both are explicit)
    jr = set(job_pre.get("region_restrictions_inferred") or job_ie.get("region_restrictions") or [])
    rc = set((resume_ie.get("countries") or []))
    if not rc:
        loc_text = " ".join(resume_ie.get("location") or []).lower()
        if "united states" in loc_text or "usa" in loc_text or "u.s" in loc_text: rc.add("US")
        if "united kingdom" in loc_text or "uk" in loc_text: rc.add("UK")
        if "canada" in loc_text: rc.add("Canada")
        if "india" in loc_text: rc.add("India")
    location_ok = True if (not jr or not rc) else (not rc.isdisjoint(jr))

    passed = years_ok and edu_ok and location_ok
    return passed, {
        "years_ok": years_ok, "years_need": need, "years_have": have,
        "education_ok": edu_ok, "required_education_evaluated": list(dict.fromkeys(req_edu)),
        "location_ok": location_ok, "required_regions": sorted(jr), "resume_countries": sorted(rc)
    }

_SENIORITY_TO_NUM = {
    "Intern":0,"Junior":1,"Mid":2,"Senior":3,"Lead":4,"Staff":4,"Principal":5,
    "Manager":5,"Director":6,"VP":7,"C-level":8,"Multiple/Unclear":2
}

def _seniority_gap(job_level: str, res_level: str) -> int:
    j = _SENIORITY_TO_NUM.get((job_level or "Multiple/Unclear"), 2)
    r = _SENIORITY_TO_NUM.get((res_level or "Multiple/Unclear"), 2)
    return abs(j - r)

def auto_match_score(
    *, resume_ie: Dict[str,Any], job_pre: Dict[str,Any],
    skills_overlap: float, industry_sim: float,
    e_resume_title: np.ndarray, e_resume_resp: np.ndarray,
    weights: Dict[str,float] = None
) -> Tuple[float, Dict[str,Any]]:
    weights = weights or {"title":0.25,"responsibilities":0.20,"family":0.15,"skills":0.15,"industry":0.25}

    job_ie = job_pre["llm_ie"]
    sim_title  = _cos(e_resume_title, job_pre["e_title"])
    sim_resp   = _cos(e_resume_resp,  job_pre["e_text"])

    pref = (resume_ie.get("primary_families_ranked") or [])
    fam  = job_ie.get("job_family") or "other"
    try:
        idx = pref.index(fam); fam_score = {0:1.0, 1:0.8, 2:0.6}.get(idx, 0.3)
    except ValueError:
        fam_score = 0.3

    gap = _seniority_gap(job_ie.get("seniority_level"), resume_ie.get("seniority_level"))
    pen_seniority = 0.0 if gap <= 1 else (0.1 if gap == 2 else 0.2)

    years_have = ((resume_ie.get("experience") or {}).get("years_overall")) or 0
    pen_intern = 0.15 if (job_ie.get("is_intern") and years_have >= 3) else 0.0

    res_countries = set((resume_ie.get("countries") or []))
    job_regions   = set(job_pre.get("region_restrictions_inferred") or job_ie.get("region_restrictions") or [])
    pen_region = 0.12 if (job_regions and res_countries and res_countries.isdisjoint(job_regions)) else 0.0

    base = (
        weights["title"]             * max(0, min(1, sim_title))        +
        weights["responsibilities"]  * max(0, min(1, sim_resp))         +
        weights["family"]            * max(0, min(1, fam_score))        +
        weights["skills"]            * max(0, min(1, skills_overlap))   +
        weights["industry"]          * max(0, min(1, industry_sim))
    )
    penalties = pen_intern + pen_region + pen_seniority
    score = max(0.0, base - penalties) * 100.0

    return score, {
        "sim_title": round(sim_title,3),
        "sim_responsibilities": round(sim_resp,3),
        "fam_score": round(fam_score,3),
        "industry_sim": round(industry_sim,3),
        "skills_overlap": round(skills_overlap,3),
        "penalties": {
            "intern": pen_intern,
            "region": pen_region,
            "seniority_gap": gap,
            "seniority_penalty": pen_seniority
        }
    }

def rank_listings_for_resume(
    *, resume_enriched: Dict[str,Any],
    listings: Dict[str,Dict[str,Any]],
    precomputed: Dict[str,Dict[str,Any]],
    top_k: int = 20,
    weights: Dict[str,float] = None,
) -> List[Dict[str,Any]]:
    resume_ie = resume_enriched.get("resume_ie") or {}
    latest = (resume_ie.get("latest_experience") or {})
    resume_title = latest.get("title") or ""
    resume_resp_text = " ".join((resume_ie.get("responsibilities") or []))[:4000]

    e_resume_title, e_resume_resp = _embed_batch_resume([resume_title, resume_resp_text])

    results = []
    for lid, job_pre in precomputed.items():
        passed, why = smart_filter(job_pre, resume_ie)
        if not passed:
            continue
        job_ie = job_pre["llm_ie"]
        job_row = listings.get(lid, {})
        skills_overlap = _jaccard(
            resume_ie.get("technologies", []),
            (job_ie.get("required_technologies") or []) + (job_ie.get("preferred_technologies") or [])
        )
        industry_sim = _jaccard(resume_ie.get("industry_tags", []), job_ie.get("industry_tags", []))
        score, dbg = auto_match_score(
            resume_ie=resume_ie, job_pre=job_pre,
            skills_overlap=skills_overlap, industry_sim=industry_sim,
            e_resume_title=e_resume_title, e_resume_resp=e_resume_resp,
            weights=weights
        )
        results.append({
            "listing_id": lid,
            "title": job_pre.get("title") or job_row.get("title") or "",
            "detail_url": job_pre.get("detail_url") or job_row.get("detail_url"),
            "score": round(score, 2),
            "debug": {**dbg, "filter_reasons": why}
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# comparison.py  (drop-in replacement for _why_row)

def _why_row(job_pre: dict, resume_ie: dict, dbg: dict) -> list[str]:
    """
    Deterministic explanation bullets for a single (job, resume) row.
    - Never throws on odd types (None, set, mixed types)
    - Short, readable bullets with light truncation
    """
    def _as_list_str(x):
        """Coerce to a de-duplicated list[str] preserving order."""
        out, seen = [], set()
        if isinstance(x, (list, tuple, set)):
            it = list(x)
        elif x is None:
            it = []
        else:
            it = [x]
        for v in it:
            if isinstance(v, str):
                s = v.strip()
                k = s.lower()
                if s and k not in seen:
                    out.append(s); seen.add(k)
        return out

    def _safe_join(x, sep=", ", limit=8):
        xs = _as_list_str(x)
        if not xs:
            return ""
        xs = xs[:limit]
        s = sep.join(xs)
        return s + (" …" if len(_as_list_str(x)) > limit else "")

    bullets = []
    ji = job_pre.get("llm_ie") or {}
    fr = (dbg or {}).get("filter_reasons") or {}

    # 1) Title & responsibilities similarity (if present)
    sim_title = (dbg or {}).get("sim_title")
    if isinstance(sim_title, (int, float)) and sim_title > 0:
        bullets.append(f"Title similarity: {sim_title:.2f}")
    sim_resp = (dbg or {}).get("sim_responsibilities")
    if isinstance(sim_resp, (int, float)) and sim_resp > 0:
        bullets.append(f"Responsibilities similarity: {sim_resp:.2f}")

    # 2) Skill overlap (required + preferred vs resume tech)
    req_sk = _as_list_str((ji.get("required_technologies") or []) + (ji.get("preferred_technologies") or []))
    res_sk = _as_list_str((resume_ie or {}).get("technologies") or [])
    if req_sk and res_sk:
        req_l = {s.lower() for s in req_sk}
        overlap = [s for s in res_sk if s.lower() in req_l]
        if overlap:
            bullets.append(f"✅ Skill overlap: {_safe_join(overlap)}")

    # 3) Industry tag overlap
    job_ind = _as_list_str(ji.get("industry_tags") or [])
    res_ind = _as_list_str((resume_ie or {}).get("industry_tags") or [])
    if job_ind and res_ind:
        jl = {s.lower() for s in job_ind}
        ind_olap = [s for s in res_ind if s.lower() in jl]
        if ind_olap:
            bullets.append(f"Industry alignment: {_safe_join(ind_olap)}")

    # 4) Family alignment
    pref = _as_list_str((resume_ie or {}).get("primary_families_ranked") or [])
    fam  = (ji.get("job_family") or "other")
    if fam and pref:
        if fam in pref:
            bullets.append(f"Job family match: {fam}")

    # 5) Years / education / location confirmations
    if fr:
        if fr.get("years_ok") is True:
            need = fr.get("years_need"); have = fr.get("years_have")
            if isinstance(need, (int, float)) and isinstance(have, (int, float)):
                bullets.append(f"Experience OK: {int(have)}y meets {int(need)}y minimum")
        if fr.get("education_ok") is True and fr.get("required_education"):
            bullets.append(f"Education OK for: {_safe_join(fr.get('required_education'))}")
        if fr.get("location_ok") is True and fr.get("required_regions"):
            bullets.append(f"Region OK: overlaps with {_safe_join(fr.get('required_regions'))}")

    # 6) Penalties (explain why score is not higher)
    pens = (dbg or {}).get("penalties") or {}
    pen_bits = []
    if pens.get("seniority_penalty", 0) > 0:
        gap = pens.get("seniority_gap")
        pen_bits.append(f"seniority gap {gap}")
    if pens.get("region", 0) > 0:
        pen_bits.append("region mismatch")
    if pens.get("intern", 0) > 0:
        pen_bits.append("intern role vs mid+ experience")
    if pen_bits:
        bullets.append("Score penalties applied: " + ", ".join(pen_bits))

    # De-duplicate bullets & keep compact
    seen, final = set(), []
    for b in bullets:
        k = b.lower()
        if b and k not in seen:
            final.append(b); seen.add(k)
    return final
