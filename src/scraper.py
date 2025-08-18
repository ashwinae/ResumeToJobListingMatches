# scraper.py
# Mercor scraper using Selenium; returns {listing_id: {...}} with title, raw_text, detail_url, etc.

import re, time, hashlib
from typing import Dict, Optional, Set
from urllib.parse import urlsplit, urlunsplit, parse_qs
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions

BASE = "https://work.mercor.com"
EXPLORE = f"{BASE}/explore"
UA = "Mozilla/5.0 (compatible; MercorMatching/1.0; +https://example.com)"
LISTING_ID_RX = re.compile(r"list_[A-Za-z0-9_-]+")
STOP_MARKER_RX = re.compile(r"\bEarn\s*\$?", re.I)

JS_FIND_ALL_DEEP = """
const selector = arguments[0]; const max = arguments[1] || 9999;
const out = []; const seen = new Set();
function visit(root){
  if(!root) return;
  try { root.querySelectorAll(selector).forEach(el => { if(!seen.has(el)){ out.push(el); seen.add(el); } }); } catch(e){}
  const w = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
  let node = w.currentNode;
  while(node){ if(node.shadowRoot){ visit(node.shadowRoot); } node = w.nextNode(); }
}
visit(document);
return out.slice(0, max);
"""

JS_INNER_TEXT_DEEP = """
function innerTextDeep(node){
  let txt = "";
  function walk(n){
    if(n.nodeType === Node.ELEMENT_NODE){
      const el = n;
      if(el.shadowRoot){ walk(el.shadowRoot); }
      try { txt += (el.innerText || "") + "\\n"; } catch(e){}
      for(const c of el.children){ walk(c); }
    }
  }
  walk(node);
  return txt;
}
return innerTextDeep(document.body);
"""

def _norm_url(u: str) -> str:
    parts = list(urlsplit(u)); parts[3] = ""; parts[4] = ""; return urlunsplit(parts)

def _setup_driver(headless: bool = True, viewport=(1440, 2200)):
    opts = ChromeOptions()
    if headless: opts.add_argument("--headless=new")
    opts.add_argument(f"--user-agent={UA}")
    opts.add_argument("--disable-gpu"); opts.add_argument("--no-sandbox"); opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(f"--window-size={viewport[0]},{viewport[1]}")
    d = webdriver.Chrome(options=opts)  # Selenium Manager handles driver
    d.set_page_load_timeout(60)
    return d

def _scroll_to_bottom(driver, passes=10, pause=0.5):
    last_h = driver.execute_script("return document.documentElement.scrollHeight || document.body.scrollHeight;")
    for _ in range(passes):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_h = driver.execute_script("return document.documentElement.scrollHeight || document.body.scrollHeight;")
        if new_h == last_h: break
        last_h = new_h

def _harvest_listing_ids(driver) -> Set[str]:
    ids = set()
    try:
        anchors = driver.execute_script(JS_FIND_ALL_DEEP, "a[href*='listingId=list_']", 4000)
        for a in anchors:
            href = a.get_attribute("href") or ""
            q = parse_qs(urlsplit(href).query)
            if "listingId" in q: ids.add(q["listingId"][0])
    except Exception:
        pass
    try:
        deep_text = driver.execute_script(JS_INNER_TEXT_DEEP)
        for m in LISTING_ID_RX.finditer(deep_text): ids.add(m.group(0))
    except Exception:
        pass
    return ids

def _find_next_button(driver):
    try:
        els = driver.execute_script(JS_FIND_ALL_DEEP, "button[title='Next']", 20)
        if els: return els[0]
    except Exception:
        pass
    try:
        buttons = driver.execute_script(JS_FIND_ALL_DEEP, "button", 300)
        for b in buttons:
            t = (b.text or "").strip()
            ttl = (b.get_attribute("title") or "").strip().lower()
            if ttl == "next" or t in {"›", "»", "Next"}:
                return b
    except Exception:
        pass
    return None

def _click_js(driver, el) -> bool:
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        time.sleep(0.08)
        driver.execute_script("arguments[0].click();", el)
        return True
    except Exception:
        return False

def _tab_signature(driver) -> str:
    try:
        ids = sorted(list(_harvest_listing_ids(driver)))
        digest = hashlib.sha1(("|".join(ids)).encode()).hexdigest()
        return digest
    except Exception:
        return str(time.time())

def _get_title_h1_featured(driver):
    try:
        t = driver.execute_script("""
            const el = document.querySelector('h1#featured-post');
            return el ? (el.innerText || el.textContent || '').trim() : null;
        """)
        return t or None
    except Exception:
        return None

def _clip_text_single_marker(text: str) -> str:
    if not text: return text
    m = STOP_MARKER_RX.search(text)
    return text[:m.start()].strip() if m else text.strip()

def _normalize_text_light(s: str) -> str:
    s = re.sub(r"^\s*(Sign in\s*)?←\s*View all opportunities\s*", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()

def extract_meta_compact(raw_text: str) -> dict:
    # keep this compact: we only need title/work mode/pay hints if present
    t = (raw_text or "").replace("\u00a0"," ").strip()
    return {"title": None}  # downstream title comes from <h1>, so we keep minimal

def _parse_detail_from_text(url: str, rendered_text: str, driver) -> Dict:
    clipped = _clip_text_single_marker(rendered_text)
    txt = _normalize_text_light(clipped)
    title_h1 = _get_title_h1_featured(driver)
    meta = extract_meta_compact(txt)
    title = title_h1 or meta.get("title")
    lid = LISTING_ID_RX.search(url)
    listing_id = lid.group(0) if lid else None
    return {
        "listing_id": listing_id,
        "detail_url": _norm_url(url),
        "title": title,
        "raw_text": txt,
    }

def scrape_mercor_with_next(headless=True, max_tabs=25, scroll_passes=12, pause=0.45) -> Dict[str, Dict]:
    d = _setup_driver(headless=headless)
    try:
        d.get(EXPLORE); time.sleep(1.2)
        page_idx = 1
        all_ids = set()
        while page_idx <= max_tabs:
            _scroll_to_bottom(d, passes=scroll_passes, pause=pause)
            ids = _harvest_listing_ids(d)
            all_ids |= ids
            nxt = _find_next_button(d)
            if not nxt: break
            before_sig = _tab_signature(d)
            if not _click_js(d, nxt): break
            time.sleep(0.8)
            tries, changed = 0, False
            while tries < 8:
                time.sleep(0.3)
                after_sig = _tab_signature(d)
                if after_sig != before_sig:
                    changed = True; break
                tries += 1
            if not changed: break
            page_idx += 1

        out: Dict[str, Dict] = {}
        for lid in list(all_ids):
            url = f"{BASE}/jobs/{lid}"
            try:
                d.get(url); time.sleep(1.1)
                rendered_text = d.execute_script(JS_INNER_TEXT_DEEP)
            except Exception:
                continue
            row = _parse_detail_from_text(url, rendered_text, d)
            if row["listing_id"]:
                out[row["listing_id"]] = row
        return out
    finally:
        d.quit()
