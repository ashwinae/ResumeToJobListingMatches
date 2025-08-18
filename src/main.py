# main.py
# Example end-to-end CLI: scrape -> IE -> precompute -> run one resume

from src.scraper import scrape_mercor_with_next
from src.listings import llm_enrich_jobs, precompute_listings, serialize_precomputed
from src.resume import parse_resume_content, llm_enrich_resume
from src.comparison import rank_listings_for_resume

def run_pipeline():
    # 1) Scrape
    listings_raw = scrape_mercor_with_next(headless=True, max_tabs=10)

    # 2) LLM IE for jobs
    listings_with_ie = llm_enrich_jobs(listings_raw)

    # 3) Precompute job embeddings
    pre = precompute_listings(listings_with_ie)
    serialize_precomputed(pre, "precomputed_listings.pkl")

    # 4) One resume example
    parsed = parse_resume_content("example_resume.pdf")
    resume_enriched = llm_enrich_resume(parsed)

    # 5) Rank
    ranked = rank_listings_for_resume(
        resume_enriched=resume_enriched,
        listings=listings_with_ie,
        precomputed=pre,
        top_k=10
    )
    for r in ranked:
        print(f"{r['score']:6.2f}  {r['title']}  ->  {r.get('detail_url')}")
        print("  why:", r["debug"])

if __name__ == "__main__":
    run_pipeline()
