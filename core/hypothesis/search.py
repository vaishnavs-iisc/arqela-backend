"""
Academic search strategies for the Hypothesis Tester.

Priority order per query:
  1. PubMed (biomedical domains)
  2. Crossref (all academic fields)
  3. DuckDuckGo HTML scraping (fallback)
  4. Synthetic placeholders (last resort — clearly marked)
"""
import json
import logging
import time
import urllib.parse
import urllib.request
from functools import wraps
from typing import Optional

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

logger = logging.getLogger("HypothesisSearch")

# ---------------------------------------------------------------------------
# Domain → preferred academic sites mapping
# ---------------------------------------------------------------------------
DOMAIN_ACADEMIC_SITES: dict[str, str] = {
    "biology": "site:pubmed.ncbi.nlm.nih.gov OR site:biorxiv.org OR site:nature.com OR site:scholar.google.com",
    "medicine": "site:pubmed.ncbi.nlm.nih.gov OR site:clinicaltrials.gov OR site:thelancet.com OR site:nejm.org OR site:scholar.google.com",
    "neuroscience": "site:pubmed.ncbi.nlm.nih.gov OR site:biorxiv.org OR site:nature.com OR site:scholar.google.com",
    "physics": "site:arxiv.org OR site:aps.org OR site:nature.com OR site:scholar.google.com",
    "astrophysics": "site:arxiv.org OR site:adsabs.harvard.edu OR site:nature.com OR site:scholar.google.com",
    "chemistry": "site:pubs.acs.org OR site:rsc.org OR site:nature.com OR site:scholar.google.com",
    "materials science": "site:pubs.acs.org OR site:nature.com OR site:scholar.google.com OR site:sciencedirect.com",
    "environmental science": "site:sciencedirect.com OR site:nature.com OR site:scholar.google.com OR site:springer.com",
    "economics": "site:nber.org OR site:jstor.org OR site:repec.org OR site:scholar.google.com",
    "software engineering": "site:ieeexplore.ieee.org OR site:dl.acm.org OR site:github.com OR site:scholar.google.com",
    "psychology": "site:pubmed.ncbi.nlm.nih.gov OR site:apa.org OR site:jstor.org OR site:scholar.google.com",
    "sociology": "site:jstor.org OR site:sagepub.com OR site:scholar.google.com OR site:springer.com",
    "management": "site:jstor.org OR site:ssrn.com OR site:scholar.google.com OR site:springer.com",
    "organizational behavior": "site:apa.org OR site:jstor.org OR site:scholar.google.com OR site:sciencedirect.com",
    "strategy & innovation": "site:jstor.org OR site:ssrn.com OR site:scholar.google.com OR site:sciencedirect.com",
}

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_on_exception(retries: int = 3, backoff: float = 2.0):
    """Retry a function with exponential backoff on any exception."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = backoff
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        logger.error(f"'{func.__name__}' failed permanently after {retries} attempts: {e}")
                        raise
                    logger.warning(
                        f"'{func.__name__}' attempt {attempt + 1}/{retries} failed — "
                        f"retrying in {current_delay}s. Error: {e}"
                    )
                    time.sleep(current_delay)
                    current_delay *= 2
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Individual search backends
# ---------------------------------------------------------------------------

@retry_on_exception(retries=3, backoff=2.0)
def ddg_html_search(query: str, max_results: int = 3) -> list:
    """DuckDuckGo HTML interface scraper (fallback when APIs return nothing)."""
    logger.info(f"DDG HTML search: '{query}'")
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            soup = BeautifulSoup(response.read(), "html.parser")
            snippets = soup.find_all("a", class_="result__snippet")
            results, seen = [], set()
            for r in snippets:
                parent = r.find_parent("div", class_="result__body")
                if not parent:
                    continue
                title_a = parent.find("a", class_="result__url")
                if not title_a:
                    continue
                raw_link = title_a["href"]
                real_link = raw_link
                if "uddg=" in raw_link:
                    params = urllib.parse.parse_qs(urllib.parse.urlparse(raw_link).query)
                    extracted = params.get("uddg", [None])[0]
                    if extracted:
                        real_link = extracted
                if real_link.startswith("//"):
                    real_link = "https:" + real_link
                if real_link not in seen:
                    seen.add(real_link)
                    results.append({"title": title_a.text.strip(), "href": real_link, "body": r.text.strip()})
                if len(results) >= max_results:
                    break
            return results
    except Exception as e:
        logger.warning(f"DDG HTML search error for '{query}': {e}")
        return []


def query_pubmed(query: str, max_results: int = 3) -> list:
    """Query PubMed E-utilities API for biomedical literature."""
    logger.info(f"PubMed API search: '{query}'")
    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={urllib.parse.quote(query)}&retmode=json&retmax={max_results}"
    )
    req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []
            ids_str = ",".join(id_list)
            summary_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=pubmed&id={ids_str}&retmode=json"
            )
            with urllib.request.urlopen(
                urllib.request.Request(summary_url, headers={"User-Agent": "Mozilla/5.0"}), timeout=8
            ) as sum_resp:
                sum_data = json.loads(sum_resp.read().decode())
                results = []
                for uid in id_list:
                    doc = sum_data.get("result", {}).get(uid, {})
                    title = doc.get("title", "No Title")
                    href = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
                    source = doc.get("source", "")
                    pubdate = doc.get("pubdate", "")
                    results.append({"title": title, "href": href, "body": f"Journal: {source} ({pubdate})"})
                return results
    except Exception as e:
        logger.warning(f"PubMed API error: {e}")
        return []


def query_crossref(query: str, max_results: int = 3) -> list:
    """Query Crossref REST API for cross-disciplinary academic papers."""
    logger.info(f"Crossref API search: '{query}'")
    url = f"https://api.crossref.org/works?query={urllib.parse.quote(query)}&rows={max_results}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "mailto:research@example.com (Scientific Platform Evaluation)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            items = data.get("message", {}).get("items", [])
            results = []
            for item in items:
                title = item.get("title", ["No Title"])[0]
                doi = item.get("DOI", "")
                href = f"https://doi.org/{doi}" if doi else ""
                container = item.get("container-title", [""])[0]
                pub_date = "Unknown"
                date_parts = item.get("published-print", {}).get("date-parts", [])
                if date_parts:
                    pub_date = date_parts[0][0]
                elif item.get("published-online", {}).get("date-parts", []):
                    pub_date = item.get("published-online", {}).get("date-parts", [])[0][0]
                results.append({"title": title, "href": href, "body": f"Journal: {container} ({pub_date})"})
            return results
    except Exception as e:
        logger.warning(f"Crossref API error: {e}")
        return []


# ---------------------------------------------------------------------------
# Composite academic search
# ---------------------------------------------------------------------------

@retry_on_exception(retries=3, backoff=2.0)
def academic_search(query: str, domain: Optional[str] = None, max_results: int = 3) -> str:
    """
    Prioritised academic search:
      1. PubMed (biomedical domains)
      2. Crossref (all fields)
      3. DDG HTML scraping
    Returns a formatted string ready to inject into an LLM prompt.
    """
    logger.info(f"Academic search: '{query}' (domain='{domain}')")
    results_list: list = []
    seen_links: set = set()

    domain_lower = domain.lower() if domain else ""
    is_biomedical = any(d in domain_lower for d in ["biology", "medicine", "neuroscience", "psychology"])

    if is_biomedical:
        for r in query_pubmed(query, max_results=max_results):
            if r["href"] not in seen_links:
                seen_links.add(r["href"])
                results_list.append(r)

    if len(results_list) < max_results:
        for r in query_crossref(query, max_results=max_results):
            if r["href"] not in seen_links:
                seen_links.add(r["href"])
                results_list.append(r)

    if len(results_list) < max_results:
        logger.info("APIs returned insufficient results — falling back to DDG HTML scraping.")
        for r in ddg_html_search(query, max_results=max_results):
            if r["href"] not in seen_links:
                seen_links.add(r["href"])
                results_list.append(r)

    formatted = [
        f"Title: {r.get('title')}\nLink: {r.get('href')}\nContent: {r.get('body')}\n"
        for r in results_list[:max_results + 1]
    ]
    return "\n---\n".join(formatted) if formatted else f"No search results found for query: '{query}'."


@retry_on_exception(retries=3, backoff=2.0)
def web_search(query: str, max_results: int = 3) -> str:
    """General web search via DuckDuckGo with Crossref fallback."""
    logger.info(f"Web search: '{query}'")
    results = ddg_html_search(query, max_results=max_results)
    if not results:
        results = query_crossref(query, max_results=max_results)
    if not results:
        return f"No search results found for query: '{query}'."
    formatted = [
        f"Title: {r.get('title')}\nLink: {r.get('href')}\nContent: {r.get('body')}\n"
        for r in results
    ]
    return "\n---\n".join(formatted)
