import json
import re

import requests
from bs4 import BeautifulSoup


def _basic_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }


def scrape_generic_profile(url, timeout=15):
    try:
        r = requests.get(url, headers=_basic_headers(), timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text_content = soup.get_text(separator="\n", strip=True)
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, text_content)
        phone_pattern = r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]"
        phones = re.findall(phone_pattern, text_content)
        name = ""
        title_tag = soup.find("title")
        if title_tag:
            name = title_tag.text.split("|")[0].split("-")[0].strip()
        h1 = soup.find("h1")
        if h1 and len(h1.text.strip()) < 60:
            name = h1.text.strip()
        meta_desc = soup.find("meta", {"name": "description"})
        summary = meta_desc.get("content", "") if meta_desc else ""
        info = {
            "name": name,
            "email": emails[0] if emails else "",
            "phone": phones[0] if phones else "",
            "headline": summary[:200],
            "summary": text_content[:500],
            "content": text_content[:2000],
        }
        return info, "✅ Successfully scraped profile information"
    except requests.exceptions.RequestException as e:
        return None, f"❌ Network error: {e}"
    except Exception as e:
        return None, f"❌ Error scraping profile: {e}"


def scrape_linkedin(url, timeout=15):
    """Attempt to gather public metadata from LinkedIn pages. LinkedIn actively blocks scraping —
    this function is best-effort and returns a friendly error when blocked."""
    try:
        r = requests.get(url, headers=_basic_headers(), timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        info = {
            "name": "",
            "headline": "",
            "summary": "",
            "experience": "",
            "education": "",
            "skills": "",
        }
        title = soup.find("title")
        if title and title.text:
            info["name"] = title.text.split("-")[0].strip()
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc and meta_desc.get("content") is not None:
            info["headline"] = str(meta_desc.get("content"))
        # JSON-LD
        json_ld = soup.find_all("script", type="application/ld+json")
        for s in json_ld:
            try:
                if s.string is not None:
                    data = json.loads(s.string)
                    if isinstance(data, dict) and data.get("@type") == "Person":
                        info["name"] = data.get("name", info["name"])
                        info["headline"] = data.get("jobTitle", info["headline"])
            except Exception:
                pass
        # Basic heuristic search for education keywords
        body_text = soup.get_text("\n", strip=True)
        if "university" in body_text.lower() or "college" in body_text.lower():
            for line in body_text.split("\n")[:200]:
                if any(
                    k in line.lower()
                    for k in ("university", "college", "bachelor", "master", "phd")
                ):
                    info["education"] += line.strip() + "\n"
        if not info["name"] and not info["headline"]:
            return None, "⚠️ LinkedIn scraping blocked or profile not public."
        return info, "✅ Successfully scraped LinkedIn profile (limited)"
    except requests.exceptions.RequestException as e:
        return None, f"❌ Error: Could not access LinkedIn. Details: {e}"
    except Exception as e:
        return None, f"❌ Error scraping LinkedIn: {e}"
