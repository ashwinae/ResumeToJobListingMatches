# resume.py
# Parse resumes (PDF/DOCX), LLM IE for resumes.

import os, re, json
from typing import Dict, Any, List
from copy import deepcopy
from src.config import openai_client, IE_MODEL, gemini_client,GEMINI_MODEL

def parse_resume_content(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        raw = "\n".join([p.extract_text() or "" for p in reader.pages])
    elif ext in (".docx", ".doc"):
        from docx import Document
        doc = Document(path)
        raw = "\n".join([p.text for p in doc.paragraphs])
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

    EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE_RX = re.compile(r"(?:\+?\d{1,2}\s*)?(?:\(?\d{3}\)?[\s.-]*)?\d{3}[\s.-]*\d{4}")
    LOC_RX   = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*),\s*[A-Z]{2,}\b")

    contact = {
        "email": EMAIL_RX.search(raw).group(0) if EMAIL_RX.search(raw) else None,
        "phone": PHONE_RX.search(raw).group(0) if PHONE_RX.search(raw) else None,
        "locations": list(dict.fromkeys(LOC_RX.findall(raw)))[:3]
    }
    return {"path": path, "raw_text": raw, "contact": contact}

_RESUME_SYSTEM = """You are an expert resume analyst.
Return ONLY valid JSON with exactly these keys (no extra text):
technologies: string[]
experience: {years_overall: int|null, areas: string[]}
education: string[]
certifications: string[]
languages_spoken: string[]
languages_programming: string[]
location: string[]
work_authorization: string[]
clearance: string[]
responsibilities: string[]
achievements: string[]
industry_tags: string[]
seniority_level: one of ["Intern","Junior","Mid","Senior","Lead","Staff","Principal","Manager","Director","VP","C-level","Multiple/Unclear"]
latest_experience: {
  title: string|null, organization: string|null, start_date: string|null, end_date: string|null,
  is_current: boolean|null, months: int|null, location: string[], technologies: string[], responsibilities: string[]
}
primary_families_ranked: string[]
seniority_level_est: one of ["Intern","Junior","Mid","Senior","Lead","Staff","Principal","Manager","Director","VP","C-level","Multiple/Unclear"]
countries: string[]
"""

def _coerce_resume_ie(data: Dict[str, Any]) -> Dict[str, Any]:
    empty = {
        "technologies": [],
        "experience": {"years_overall": None, "areas": []},
        "education": [],
        "certifications": [],
        "languages_spoken": [],
        "languages_programming": [],
        "location": [],
        "work_authorization": [],
        "clearance": [],
        "responsibilities": [],
        "achievements": [],
        "industry_tags": [],
        "seniority_level": "Multiple/Unclear",
        "latest_experience": {
            "title": None, "organization": None, "start_date": None, "end_date": None,
            "is_current": None, "months": None, "location": [], "technologies": [], "responsibilities": []
        },
        "primary_families_ranked": [],
        "seniority_level_est": "Multiple/Unclear",
        "countries": []
    }
    out = deepcopy(empty)
    if not isinstance(data, dict): data = {}
    for k in empty:
        out[k] = data.get(k, empty[k])

    def _dedupe(arr: List[str]) -> List[str]:
        seen, res = set(), []
        for s in (arr or []):
            if not isinstance(s, str): continue
            s2 = " ".join(s.split())
            k = s2.lower()
            if s2 and k not in seen:
                res.append(s2); seen.add(k)
        return res

    out["technologies"] = _dedupe(out["technologies"])
    out["education"] = _dedupe(out["education"])
    out["responsibilities"] = _dedupe(out["responsibilities"])
    out["industry_tags"] = _dedupe(out["industry_tags"])
    out["primary_families_ranked"] = _dedupe(out["primary_families_ranked"])
    out["countries"] = _dedupe(out["countries"])

    # experience years normalize
    yrs = out["experience"].get("years_overall")
    if isinstance(yrs, float): yrs = int(yrs)
    if not isinstance(yrs, int): yrs = None
    out["experience"]["years_overall"] = yrs

    return out

def llm_enrich_resume(parsed: Dict[str, Any], model: str = IE_MODEL) -> Dict[str, Any]:
    client = openai_client()
    txt = (parsed.get("raw_text") or "")[:15000]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":_RESUME_SYSTEM},
                      {"role":"user","content":txt}],
            response_format={"type":"json_object"},
            temperature=0.0, top_p=1
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {}
    enriched = dict(parsed)
    enriched["resume_ie"] = _coerce_resume_ie(data)
    return enriched


# --- Compensation inference (Gemini API) ---
def extract_median_hourly_usd(resp):
    """
    Works with Google Generative AI Python SDK's GenerateContentResponse.
    Expects JSON (possibly fenced with ```json ... ```).
    Returns float or None.
    """
    if not resp or not getattr(resp, "candidates", None):
        return None

    # Concatenate all text parts from the top candidate
    parts = getattr(resp.candidates[0].content, "parts", []) or []
    text = "".join([getattr(p, "text", "") or "" for p in parts]).strip()
    print(text)
    if not text:
        return None

    # Remove ```json ... ``` fences if present
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)

    # Fallback: grab the first {...} block if anything extra leaked in
    m = re.search(r"\{.*\}", text, re.DOTALL)
    payload = m.group(0) if m else text

    try:
        data = json.loads(payload)
        val = data.get("median_salary_hourly_usd")
        return float(val) if isinstance(val, (int, float, str)) and str(val).strip() else None
    except Exception:
        return None
        
def infer_median_salary_hourly_usd_gemini(resume_ie: Dict[str, Any]) -> float | None:
    """
    Uses Gemini with google_search tool to infer the median hourly USD salary
    for the candidate's latest role (title + location).
    Returns float (hourly USD) or None.
    """

    latest = (resume_ie or {}).get("latest_experience") or {}
    title = latest.get("title") or ""
    locs = latest.get("location") or []
    location = locs[0] if locs else (", ".join(resume_ie.get("countries") or []))
    if not title or not location:
        return None

    # Build the prompt
    prompt = f"""
    Search the web for: median salary of {title} in {location}.
    Ground your results in reputable compensation and labor-market data sources 
    (e.g. Payscale, Glassdoor, Bureau of Labor Statistics).
    Return STRICT JSON with exactly this schema, no extra keys:
    
    {{
    "median_salary_hourly_usd": number
    }}
    
    Rules:
    - Convert annual salaries to hourly using 2080 hours/year.
    - Output must be numeric USD per hour.
    - If you cannot find reliable info, retry searches with slightly different query variations. If you still can't find reliable info, return {{ "median_salary_hourly_usd": null }}.
    """

    client = gemini_client()

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"tools": [{"google_search": {}}]}
        )

        val = extract_median_hourly_usd(response)
        if isinstance(val, (int, float)) and 5 <= val <= 1000:
            return float(val)
        return None
    except Exception as e:
        print("Gemini salary inference failed:", e)
        return None
