import argparse
import csv
import html
import json
import os
import re
import time

import requests

API = "https://commons.wikimedia.org/w/api.php"
UA = "ArchForge-DatasetCollector/0.1 (research dataset build; contact marwantosolve@gmail.com)"

CATEGORIES = [
    "Category:Islamic Cairo",
    "Category:Mamluk architecture",
    "Category:Mamluk architecture in Cairo",
    "Category:Historic Cairo",
    "Category:Architecture of Cairo",
    "Category:Mosques in Cairo",
    "Category:Madrasas in Cairo",
    "Category:Muqarnas",
    "Category:Mashrabiya",
    "Category:Minarets in Cairo",
    "Category:Domes in Cairo",
    "Category:Doors in Cairo",
    "Category:Mosque of Sultan Hassan",
    "Category:Qalawun complex",
    "Category:Bab Zuweila",
    "Category:Khan el-Khalili",
    "Category:City of the Dead (Cairo)",
    "Category:Al-Azhar Mosque",
    "Category:Ibn Tulun Mosque",
    "Category:Islamic geometric patterns",
]

MIN_EDGE = 800
THUMB_WIDTH = 1280


def api_get(params, retries=5):
    query = dict(params)
    query["format"] = "json"
    query["formatversion"] = 2
    last = None
    for attempt in range(retries):
        try:
            resp = requests.get(API, params=query, headers={"User-Agent": UA}, timeout=30)
        except requests.RequestException as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
            continue
        if resp.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        if resp.status_code >= 400:
            last = RuntimeError(f"HTTP {resp.status_code}")
            time.sleep(2 * (attempt + 1))
            continue
        return resp.json()
    raise RuntimeError(f"API request failed after {retries} attempts: {last}")


def clean(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("value", "")
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def category_files(category, limit):
    data = api_get(
        {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": THUMB_WIDTH,
        }
    )
    return data.get("query", {}).get("pages", [])


def parse_page(page, category):
    info = (page.get("imageinfo") or [{}])[0]
    ext = info.get("extmetadata") or {}
    width = info.get("width") or 0
    height = info.get("height") or 0
    if not width or not height:
        return None
    if min(width, height) < MIN_EDGE:
        return None
    if info.get("mime") not in ("image/jpeg", "image/png"):
        return None
    url = info.get("thumburl") or info.get("url")
    if not url:
        return None
    return {
        "pageid": page.get("pageid"),
        "title": page.get("title", ""),
        "category": category,
        "download_url": url,
        "descriptionsource": info.get("descriptionurl", ""),
        "width": width,
        "height": height,
        "mime": info.get("mime", ""),
        "license": clean(ext.get("LicenseShortName")) or clean(ext.get("License")) or "UNKNOWN",
        "license_url": clean(ext.get("LicenseUrl")),
        "artist": clean(ext.get("Artist")),
        "credit": clean(ext.get("Credit")),
        "description": clean(ext.get("ImageDescription"))[:600],
        "date": clean(ext.get("DateTimeOriginal")),
        "usage_terms": clean(ext.get("UsageTerms")),
    }


def download(url, dest):
    resp = requests.get(url, headers={"User-Agent": UA}, timeout=120, stream=True)
    resp.raise_for_status()
    tmp = dest + ".part"
    with open(tmp, "wb") as handle:
        for chunk in resp.iter_content(65536):
            handle.write(chunk)
    size = os.path.getsize(tmp)
    os.replace(tmp, dest)
    return size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--per-category", type=int, default=6)
    parser.add_argument("--target", type=int, default=48)
    parser.add_argument("--min-candidates", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    candidates = []
    seen = set()
    for category in CATEGORIES:
        try:
            pages = category_files(category, 60)
        except Exception as exc:
            print(f"[warn] {category}: {exc}")
            continue
        kept = 0
        for page in pages:
            record = parse_page(page, category)
            if not record:
                continue
            if record["pageid"] in seen:
                continue
            seen.add(record["pageid"])
            candidates.append(record)
            kept += 1
            if kept >= args.per_category:
                break
        print(f"[info] {category}: {kept} candidates")
        time.sleep(0.3)

    print(f"[info] total unique candidates: {len(candidates)}")
    if len(candidates) < args.min_candidates:
        raise SystemExit("too few candidates; aborting")

    rows = []
    for record in candidates[: args.target]:
        ext = ".png" if record["mime"] == "image/png" else ".jpg"
        name = f"{record['pageid']}{ext}"
        dest = os.path.join(args.out, name)
        if not os.path.exists(dest):
            try:
                size = download(record["download_url"], dest)
            except Exception as exc:
                print(f"[warn] download failed {record['title']}: {exc}")
                continue
        else:
            size = os.path.getsize(dest)
        if size < 20_000:
            os.remove(dest)
            continue
        record["image_path"] = f"{args.out}/{name}"
        record["bytes"] = size
        rows.append(record)
        print(f"[ok] {name} {size // 1024}KB {record['license']}")

    if not rows:
        raise SystemExit("no images downloaded")

    fields = [
        "image_path",
        "pageid",
        "title",
        "category",
        "license",
        "license_url",
        "artist",
        "credit",
        "date",
        "descriptionsource",
        "download_url",
        "width",
        "height",
        "mime",
        "bytes",
        "usage_terms",
        "description",
    ]
    csv_path = os.path.join(args.out, "sources.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] {len(rows)} images -> {args.out}")
    print(f"[done] provenance -> {csv_path}")


if __name__ == "__main__":
    main()
