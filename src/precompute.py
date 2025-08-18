# Scrape → LLM IE → Precompute embeddings → Save artifacts (JSON + PKL)

import os, json, argparse
from src.scraper import scrape_mercor_with_next
from src.listings import llm_enrich_jobs, precompute_listings, serialize_precomputed, normalize_comp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tabs", type=int, default=10, help="Max 'Next' tabs to click on Explore")
    ap.add_argument("--headless", action="store_true", help="Run Chrome headless")
    ap.add_argument("--listings-json", default="data/result.json", help="Path to save listings+IE JSON")
    ap.add_argument("--precomputed-pkl", default="data/pre.pkl", help="Path to save precomputed PKL")
    args = ap.parse_args()

    # 1) Scrape Mercor listings
    print("[STEP 1] Scraping Mercor…")
    listings_raw = scrape_mercor_with_next(headless=args.headless, max_tabs=args.max_tabs)

    # 2) LLM IE (extract hard filters, responsibilities, family tags, etc.)
    print("[STEP 2] Enriching listings with LLM IE…")
    listings_with_ie = llm_enrich_jobs(listings_raw)
    listings_with_ie = normalize_comp(listings_with_ie)

    # 3) Precompute (title/text embeddings + inferred constraints)
    print("[STEP 3] Precomputing embeddings…")
    pre = precompute_listings(listings_with_ie)

    # 4) Save artifacts
    print(f"[SAVE] {args.listings_json}")
    with open(args.listings_json, "w", encoding="utf-8") as f:
        json.dump(listings_with_ie, f, ensure_ascii=False)

    print(f"[SAVE] {args.precomputed_pkl}")
    serialize_precomputed(pre, args.precomputed_pkl)

    print("\n✅ Done. You can now launch the Streamlit app:\n"
          "   streamlit run app.py\n"
          "and point it to the generated `precomputed_listings.pkl` (sidebar).\n")

if __name__ == "__main__":
    main()
