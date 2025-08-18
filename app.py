# app.py
import os
import pickle
import streamlit as st

from src.resume import (
    parse_resume_content,
    llm_enrich_resume,
    infer_median_salary_hourly_usd_gemini,
)
from src.comparison import rank_listings_for_resume, _why_row  

DATA_DIR = "data"
RESULT_PKL = os.path.join(DATA_DIR, "result.pkl")
PRE_PKL = os.path.join(DATA_DIR, "pre.pkl")

# --------- helpers to load data ---------
@st.cache_data(show_spinner=False)
def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _load_inputs():
    if not (os.path.exists(RESULT_PKL) and os.path.exists(PRE_PKL)):
        raise FileNotFoundError(
            f"Expected pickles not found in {DATA_DIR}/\n"
            f"- {RESULT_PKL}\n- {PRE_PKL}\n"
        )
    result = _load_pickle(RESULT_PKL)
    pre = _load_pickle(PRE_PKL)
    return result, pre

def _pay_in_range(target_hourly, low, high):
    try:
        if target_hourly is None or low is None or high is None:
            return None
        return float(low) <= float(target_hourly) <= float(high)
    except Exception:
        return None

# --------- UI ---------
st.set_page_config(page_title="Resume → Top Job Listings", page_icon="🔎", layout="wide")
st.title("🔎 Resume → Top Job Listing Matches")

# Load data
try:
    result, pre = _load_inputs()
    st.success(f"Loaded {len(result)} listings and {len(pre)} precomputed embeddings.")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# File uploader
uploaded = st.file_uploader(
    "Upload your resume (PDF or DOCX):",
    type=["pdf", "docx", "doc"],
    accept_multiple_files=False
)

col_a, col_b = st.columns([3, 1])
with col_b:
    top_k = st.number_input("Top K", min_value=3, max_value=50, value=10, step=1)

if uploaded:
    # Save uploaded resume to a temp path
    tmp_path = os.path.join("._tmp_upload_" + uploaded.name.replace("/", "_"))
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # 1) Parse + 2) Enrich
    with st.spinner("Parsing + enriching resume…"):
        parsed = parse_resume_content(tmp_path)
        resume_enriched = llm_enrich_resume(parsed)
        resume_ie = resume_enriched.get("resume_ie", {})

    # 2.5) Infer target hourly comp (USD) using Gemini (title + location from latest experience)
    with st.spinner("Inferring target hourly compensation (Gemini)…"):
        target_hourly = infer_median_salary_hourly_usd_gemini(resume_ie)

    # Show target comp + latest role context
    le = (resume_ie.get("latest_experience") or {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Target Hourly (USD)", f"${target_hourly:.2f}" if target_hourly else "Unknown")
    with c2:
        st.write("**Latest Title**:", le.get("title") or "—")
    with c3:
        locs = le.get("location") or []
        st.write("**Latest Location**:", ", ".join(locs) if locs else "—")

    # 3) Rank
    with st.spinner("Scoring against listings…"):
        ranked = rank_listings_for_resume(
            resume_enriched=resume_enriched,
            listings=result,
            precomputed=pre,
            top_k=top_k
        )

    # 4) Render results
    st.subheader("Top matches")
    if not ranked:
        st.info("No matches passed the filters. Try another resume.")
    else:
        for r in ranked:
            lid = r["listing_id"]
            job_pre = pre.get(lid, {})
            dbg = r.get("debug", {})
            title = r.get("title") or job_pre.get("title") or "Untitled"
            url = r.get("detail_url") or (result.get(lid, {}) or {}).get("detail_url")

            # Pull comp range from the original listing row in `result`
            listing_row = result.get(lid, {}) or {}
            comp_low = listing_row.get("comp_low")
            comp_high = listing_row.get("comp_high")
            in_range = _pay_in_range(target_hourly, comp_low, comp_high)

            score = r.get("score", 0)
            header = f"{title} — {score:.2f}"

            with st.container():
                st.markdown(f"### {header}")
                if url:
                    st.markdown(f"[Open listing]({url})")

                # metrics row
                cols = st.columns(5)
                cols[0].metric("Title similarity", f"{dbg.get('sim_title', 0.0):.2f}")
                cols[1].metric("Responsibilities similarity", f"{dbg.get('sim_responsibilities', 0.0):.2f}")
                cols[2].metric("Skills Overlap", f"{dbg.get('skills_overlap', 0.0):.2f}")
                cols[3].metric("Industry Overlap", f"{dbg.get('industry_sim', 0.0):.2f}")
                # comp
                if comp_low is not None and comp_high is not None:
                    comp_str = f"${comp_low:.0f}–${comp_high:.0f} / hr"
                else:
                    comp_str = "—"
                cols[4].metric("Listing Comp Range", comp_str)

                # Expected pay banner
                if in_range is True:
                    st.success("✅ Pay in expected range")
                elif in_range is False and target_hourly < comp_low:
                    st.warning("✅ Pay better than expected range")
                elif in_range is False:
                    st.warning("⚠️ Pay below expected range")                    
                else:
                    st.info("ℹ️ No target or comp available")

                # Deterministic explanations via your _why_row
                bullets = _why_row(job_pre, resume_enriched.get("resume_ie", {}), dbg)
                with st.expander("Why this match?"):
                    if bullets:
                        for b in bullets:
                            st.markdown(f"- {b}")
                    else:
                        st.markdown("- No specific highlights found.")

                # Filter reasons for transparency
                fr = (dbg.get("filter_reasons") or {})
                with st.expander("Filters & constraints"):
                    st.write({
                        "years_ok": fr.get("years_ok"),
                        "years_need": fr.get("years_need"),
                        "years_have": fr.get("years_have"),
                        "education_ok": fr.get("education_ok"),
                        "required_education": fr.get("required_education_evaluated"),
                        "location_ok": fr.get("location_ok"),
                        "required_regions": fr.get("required_regions"),
                        "resume_countries": fr.get("resume_countries"),
                        "penalties": dbg.get("penalties"),
                    })

                st.divider()

    # Cleanup temp file
    try:
        os.remove(tmp_path)
    except Exception:
        pass
else:
    st.info("Upload a resume to get started.")
