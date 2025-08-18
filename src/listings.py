# LLM extraction for jobs + precompute embeddings + save/load.

import os, re, json, pickle, numpy as np
from typing import Dict, Any, List
from copy import deepcopy
from src.config import openai_client, IE_MODEL, EMBED_MODEL, EMBED_DIM

# --------- LLM IE for job listings ---------
_JOB_IE_SCHEMA = {
    "name": "JobIE",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "required_technologies": {"type": "array","items":{"type":"string"}},
            "required_experience": {
                "type":"object","additionalProperties":False,
                "properties":{"years_min":{"type":["integer","null"],"minimum":0},"areas":{"type":"array","items":{"type":"string"}}},
                "required":["years_min","areas"]
            },
            "required_education": {"type":"array","items":{"type":"string"}},
            "required_certifications": {"type":"array","items":{"type":"string"}},
            "required_languages_spoken": {"type":"array","items":{"type":"string"}},
            "required_languages_programming": {"type":"array","items":{"type":"string"}},
            "required_location": {"type":"array","items":{"type":"string"}},
            "required_work_authorization": {"type":"array","items":{"type":"string"}},
            "required_clearance": {"type":"array","items":{"type":"string"}},
            "preferred_technologies": {"type":"array","items":{"type":"string"}},
            "preferred_experience": {"type":"array","items":{"type":"string"}},
            "preferred_education": {"type":"array","items":{"type":"string"}},
            "preferred_certifications": {"type":"array","items":{"type":"string"}},
            "responsibilities": {"type":"array","items":{"type":"string"}},
            "strong_fit_description": {"type":"string","maxLength":400},
            "seniority_level": {"type":"string","enum": [
                "Intern","Junior","Mid","Senior","Lead","Staff","Principal","Manager","Director","VP","C-level","Multiple/Unclear"
            ]},
            "industry_tags": {"type":"array","items":{"type":"string"}},
            "job_family": {"type":"string","enum":[
                "data_science","ml_engineer","software_engineer","analyst","evaluator",
                "product","design","ops","sales","legal","other"
            ]},
            "region_restrictions": {"type":"array","items":{"type":"string"}},
            "is_intern": {"type":"boolean"}
        },
        "required": [
            "required_technologies","required_experience","required_education","required_certifications",
            "required_languages_spoken","required_languages_programming","required_location",
            "required_work_authorization","required_clearance",
            "preferred_technologies","preferred_experience","preferred_education","preferred_certifications",
            "responsibilities","strong_fit_description","seniority_level","industry_tags",
            "job_family","region_restrictions","is_intern"
        ]
    },
    "strict": True
}

_SYSTEM_INSTRUCTIONS = """You are an expert recruitment analyst.
Return ONLY JSON that conforms EXACTLY to the provided JSON Schema.

HARD CONSTRAINTS (NO HALLUCINATIONS):
- Use ONLY facts explicitly present in the input job text. If an item is not stated, use [] for arrays and null for nullable fields.
- Do NOT restate or guess title, company, pay, hours, duration, or work mode. Include locations ONLY when the text explicitly states them as hard requirements (e.g., “must be located in…”, “US-only”, “based in Canada”).
- ‘required_*’ fields are hard filters; include an item ONLY if the text clearly implies “must/required/need to/only if/eligible only/located in/US-only/no sponsorship/etc.”.
- Separate spoken languages vs programming languages. If ambiguous, leave both empty.
- ‘responsibilities’ = SHORT, verb-first lines copied or minimally trimmed from the text (no new content). Extract ALL clear duties; cap to the most specific 12 if the list is very long.
- ‘strong_fit_description’ ≤ 60 words and MUST paraphrase only what appears in the text. Populate this ONLY if the posting has language like “You’re a strong fit if…”, “ideal candidate…”, “nice to have…”, “we’re looking for…”. Otherwise return "".
- Deduplicate lists; keep original casing/acronyms as in the text.

META LABELS (deterministic; no extra calls, no guessing beyond text):
- ‘job_family’: infer a generic family from title/description if CLEARLY indicated; else 'other'.
  Map using keyword presence (case-insensitive):
    * data_science  -> any of ["data scientist","data science","data analyst (ML context)","analytics engineer (ML/DS context)"]
    * ml_engineer   -> any of ["ml engineer","machine learning engineer","mlops","research scientist (ml)","ai researcher (engineering)"]
    * software_engineer -> any of ["software engineer","backend","full-stack","frontend","swe","platform","infrastructure"]
    * analyst       -> any of ["analyst","business analyst","financial analyst","investment analyst"] (non-ML context)
    * evaluator     -> any of ["evaluator","rater","grader","judge","task author","assessment","labeling","annotation"]
    * product       -> any of ["product manager","pm","product owner"]
    * design        -> any of ["designer","ux","ui","product designer"]
    * ops           -> any of ["operations","ops","talent success","program manager","project manager"]
    * sales         -> any of ["sales","account executive","growth (sales)","customer success (quota-tied)"]
    * legal         -> any of ["lawyer","legal","counsel","judge","magistrate"]
    * healthcare    -> any of ["health care", "medical", "biology"]
  If multiple families match, choose the best single family that dominates the role; otherwise 'other'.
- ‘region_restrictions’: normalize explicit geographic limits to country/region names (e.g., ["US"], ["US","Canada"], ["India"], ["EU"], ["UK"]). If the text is global/remote without explicit limits, return [].
  Examples:
    * “US-only”, “US-based”, “eligible to work in the US” -> ["US"]
    * “US or Canada” -> ["US","Canada"]
    * “India only”, “based in India” -> ["India"]
    * “UK only” -> ["UK"]
- ‘is_intern’: true ONLY if the role is explicitly an internship (e.g., “Intern”, “Internship”). Part-time/contract ≠ internship.

ABSOLUTE RULES:
- Never fabricate requirements or preferences.
- If a field is not explicitly supported by the text, leave it empty ([], null, or false per schema).
"""

def llm_enrich_jobs(listings: Dict[str, Dict[str, Any]], model: str = IE_MODEL) -> Dict[str, Dict[str, Any]]:
    client = openai_client()
    out = {}
    for lid, row in listings.items():
        txt = (row.get("raw_text") or "")[:12000]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":_SYSTEM_INSTRUCTIONS},
                          {"role":"user","content":txt}],
                response_format={"type":"json_schema","json_schema":_JOB_IE_SCHEMA},
                temperature=0.0, top_p=1
            )
            msg = resp.choices[0].message
            parsed = getattr(msg, "parsed", None)
            ie = parsed if parsed is not None else json.loads(msg.content)
        except Exception:
            # fail-closed defaults
            ie = {
                "required_technologies": [],
                "required_experience": {"years_min": None, "areas": []},
                "required_education": [],
                "required_certifications": [],
                "required_languages_spoken": [],
                "required_languages_programming": [],
                "required_location": [],
                "required_work_authorization": [],
                "required_clearance": [],
                "preferred_technologies": [],
                "preferred_experience": [],
                "preferred_education": [],
                "preferred_certifications": [],
                "responsibilities": [],
                "strong_fit_description": "",
                "seniority_level": "Multiple/Unclear",
                "industry_tags": [],
                "job_family": "other",
                "region_restrictions": [],
                "is_intern": False
            }
        new_row = dict(row)
        new_row["llm_ie"] = ie
        out[lid] = new_row
    return out

# --------- Embedding precompute for job listings ---------
_EMBED_CACHE: Dict[str, np.ndarray] = {}
_EMBED_DISK = ".embed_cache.pkl"

def _load_disk_cache():
    global _EMBED_CACHE
    if os.path.exists(_EMBED_DISK):
        try:
            with open(_EMBED_DISK,"rb") as f:
                _EMBED_CACHE.update(pickle.load(f))
        except Exception:
            pass

def _save_disk_cache():
    try:
        with open(_EMBED_DISK,"wb") as f:
            pickle.dump(_EMBED_CACHE, f)
    except Exception:
        pass

def _embed_batch(texts: List[str], model: str = EMBED_MODEL) -> List[np.ndarray]:
    client = openai_client()
    keys = [(model + "||" + (t or "").strip())[:4096] for t in texts]
    out: List[np.ndarray] = [None] * len(texts)
    to_idx, to_texts = [], []
    for i, (k, t) in enumerate(zip(keys, texts)):
        t = (t or "").strip()
        if not t:
            out[i] = np.zeros(EMBED_DIM, dtype=np.float32)
        elif k in _EMBED_CACHE:
            out[i] = _EMBED_CACHE[k]
        else:
            to_idx.append(i); to_texts.append(t)
    if to_idx:
        resp = client.embeddings.create(model=model, input=to_texts)
        for j, data in enumerate(resp.data):
            i = to_idx[j]
            vec = np.array(data.embedding, dtype=np.float32)
            _EMBED_CACHE[keys[i]] = vec
            out[i] = vec
    for i in range(len(out)):
        if out[i] is None:
            out[i] = np.zeros(EMBED_DIM, dtype=np.float32)
    return out

# Light inference for constraints (education/region) one-time
_EDU_RX = {
    "phd": re.compile(r"\b(phd|ph\.?d\.?|doctoral|doctorate|statistics\s+phds?)\b", re.I),
    "masters": re.compile(r"\b(master'?s|m\.?s\.?|msc)\b", re.I),
    "bachelors": re.compile(r"\b(bachelor'?s|b\.?s\.?|bsc)\b", re.I),
}
_COUNTRY_CANON = {
    "us":"US","u.s.":"US","u.s":"US","usa":"US","united states":"US","america":"US",
    "uk":"UK","u.k.":"UK","united kingdom":"UK","england":"UK","britain":"UK",
    "canada":"Canada","india":"India",
}
_COUNTRY_RX = re.compile(r"\b(US|U\.S\.|USA|United States|UK|U\.K\.|United Kingdom|Canada|India)\b", re.I)

def _canon_country(s: str) -> str:
    s = (s or "").strip().lower()
    return _COUNTRY_CANON.get(s, s.upper())

def _countries_from_text(txt: str) -> List[str]:
    hits = set()
    for m in _COUNTRY_RX.finditer(txt or ""):
        hits.add(_canon_country(m.group(0)))
    return sorted(hits)

def _infer_required_education_from_text(title: str, raw: str, existing: List[str]) -> List[str]:
    edu = list(existing or [])
    text = f"{title}\n{raw}"
    if _EDU_RX["phd"].search(text): edu.append("PhD")
    elif _EDU_RX["masters"].search(text): edu.append("Masters")
    elif _EDU_RX["bachelors"].search(text): edu.append("Bachelors")
    # dedupe preserving order
    seen, out = set(), []
    for e in edu:
        k = e.strip().lower()
        if k and k not in seen:
            out.append(e); seen.add(k)
    return out

def _infer_region_limits_from_text(raw: str, required_location: List[str], region_restrictions: List[str]) -> List[str]:
    regions = set(region_restrictions or [])
    for loc in (required_location or []):
        cc = _canon_country(loc)
        if cc: regions.add(cc)
    for c in _countries_from_text(raw):
        regions.add(c)
    return sorted(regions)

def _job_text_with_fallback(job_ie: Dict[str,Any], job_row: Dict[str,Any]) -> str:
    resp_lines = job_ie.get("responsibilities") or []
    strong_fit = job_ie.get("strong_fit_description") or ""
    text = " ".join(resp_lines + ([strong_fit] if strong_fit else []))
    if not text.strip():
        text = (job_row.get("raw_text") or "")
    return text[:2000]

def precompute_listings(listings_with_ie: Dict[str,Dict[str,Any]], embed_model: str = EMBED_MODEL) -> Dict[str, Dict[str, Any]]:
    """
    INPUT:
      listings_with_ie: {lid: {title, raw_text, detail_url, llm_ie: {...}}}
    OUTPUT:
      pre: {lid: {"title","text","e_title","e_text","llm_ie","required_education_inferred","region_restrictions_inferred"}}
    """
    _load_disk_cache()
    lids, titles, texts, req_edus, req_regions, ies = [], [], [], [], [], []
    for lid, row in listings_with_ie.items():
        ie = row.get("llm_ie") or {}
        title = row.get("title") or ""
        raw   = row.get("raw_text") or ""
        text  = _job_text_with_fallback(ie, row)
        req_edu    = _infer_required_education_from_text(title, raw, ie.get("required_education") or [])
        req_region = _infer_region_limits_from_text(raw, ie.get("required_location") or [], ie.get("region_restrictions") or [])
        lids.append(lid); titles.append(title); texts.append(text)
        req_edus.append(req_edu); req_regions.append(req_region); ies.append(ie)
    e_titles = _embed_batch(titles, model=embed_model)
    e_texts  = _embed_batch(texts,  model=embed_model)
    out = {}
    for lid, t, x, et, ex, edu, regs, ie in zip(lids, titles, texts, e_titles, e_texts, req_edus, req_regions, ies):
        out[lid] = {
            "title": t, "text": x,
            "e_title": et, "e_text": ex,
            "llm_ie": ie,
            "required_education_inferred": edu,
            "region_restrictions_inferred": regs,
            "detail_url": listings_with_ie[lid].get("detail_url")
        }
    _save_disk_cache()
    return out

def normalize_comp(listings_with_ie: dict) -> dict:
    """
    Normalize compensation to hourly:
      - 'year'  -> divide by 2080 (40 hrs * 52 wks)
      - 'month' -> divide by 160  (40 hrs * 4 wks)
      - 'week'  -> divide by 40   (40 hrs / week)
      - 'hour'  -> unchanged
    Updates comp_low, comp_high, comp_unit.
    """
    factors = {
        "year": 2080.0,
        "month": 160.0,
        "week": 40.0
    }
    
    # listings is dict keyed by listing_id -> listing dict
    for lid, job in listings_with_ie.items():
        unit = (job.get("comp_unit") or "").lower()
        if unit in factors:
            factor = factors[unit]
            try:
                if job.get("comp_low") is not None:
                    job["comp_low"] = round(float(job["comp_low"]) / factor)
                if job.get("comp_high") is not None:
                    job["comp_high"] = round(float(job["comp_high"]) / factor)
            except Exception:
                # if values are not numeric
                pass
            job["comp_unit"] = "hour"
    return listings_with_ie    

def serialize_precomputed(precomputed: Dict[str,Dict[str,Any]], path: str):
    to_save = {}
    for lid, v in precomputed.items():
        row = dict(v)
        row["e_title"] = v["e_title"].tolist()
        row["e_text"]  = v["e_text"].tolist()
        to_save[lid] = row
    with open(path,"wb") as f:
        pickle.dump(to_save, f)

def load_precomputed(path: str) -> Dict[str,Dict[str,Any]]:
    with open(path,"rb") as f:
        raw = pickle.load(f)
    for lid, v in raw.items():
        v["e_title"] = np.array(v["e_title"], dtype=np.float32)
        v["e_text"]  = np.array(v["e_text"],  dtype=np.float32)
    return raw
