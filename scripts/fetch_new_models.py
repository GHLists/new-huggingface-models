#!/usr/bin/env python3
"""Fetch Hugging Face models created between the previous list and now.

New models are read from the Hugging Face Hub API sorted by creation date.
Pages are followed through the API's cursor links until the oldest model on a
page predates the requested window. The end of the last list is stored in the
manifest so the next run resumes where the previous one stopped.
"""

import argparse
import csv
import datetime as dt
import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MODELS_URL = "https://huggingface.co/api/models"
DEFAULT_USER_AGENT = (
    "new-huggingface-models/1.0 (https://github.com/GHLists/new-huggingface-models)"
)

PAGE_SIZE = 1000
MAX_PAGES = 50
TAG_LIMIT = 200
CSV_HEADER = ("created_at", "model", "author", "downloads", "likes", "tags")

TRANSIENT_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    json.JSONDecodeError,
    http.client.HTTPException,
    OSError,
)


def iso(moment):
    moment = moment.astimezone(dt.timezone.utc)
    if moment.microsecond:
        fraction = f"{moment.microsecond:06d}".rstrip("0")
        return moment.strftime("%Y-%m-%dT%H:%M:%S") + f".{fraction}Z"
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_timestamp(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = dt.datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc)


def timestamp_filename(moment):
    moment = moment.astimezone(dt.timezone.utc)
    stamp = moment.strftime("%Y-%m-%dT%H-%M-%S")
    if moment.microsecond:
        stamp += "-" + f"{moment.microsecond:06d}".rstrip("0")
    return stamp + "Z"


def fetch_json(url, user_agent, retries=3, backoff=5.0):
    """Return the parsed body and the response headers."""
    last_error = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response), response.headers
        except TRANSIENT_ERRORS as error:
            last_error = error
        if attempt < retries:
            print(f"attempt {attempt} failed ({last_error}), retrying", file=sys.stderr)
            time.sleep(backoff * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def next_link(headers):
    """Return the API's next-page URL from the Link header, if any."""
    link = headers.get("Link")
    if not link:
        return None
    match = re.search(r'<([^>]+)>\s*;\s*rel="next"', link)
    if not match:
        return None
    candidate = match.group(1)
    parts = urllib.parse.urlsplit(candidate)
    if parts.scheme != "https" or parts.netloc != "huggingface.co":
        raise RuntimeError(f"unexpected pagination link: {candidate}")
    return candidate


def fetch_new_models(since, user_agent, retries, max_pages):
    """Return the newest models, following pages until ``since`` is covered."""
    url = f"{MODELS_URL}?sort=createdAt&direction=-1&limit={PAGE_SIZE}"
    models = []
    for _ in range(max_pages):
        payload, headers = fetch_json(url, user_agent, retries=retries)
        if not isinstance(payload, list):
            raise RuntimeError("Hugging Face response is not a list")
        models.extend(payload)
        if not payload:
            return models, True
        try:
            oldest = min(parse_timestamp(model["createdAt"]) for model in payload)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Hugging Face response contains an invalid createdAt"
            ) from error
        if oldest <= since:
            return models, True
        url = next_link(headers)
        if url is None:
            return models, True
    return models, False


def clean_text(value, limit=TAG_LIMIT):
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "\u2026"
    return text


def build_row(model, created):
    model_id = model.get("id") or ""
    author = model.get("author")
    if not isinstance(author, str) or not author:
        author = model_id.split("/", 1)[0] if "/" in model_id else ""
    downloads = model.get("downloads")
    likes = model.get("likes")
    tags = model.get("tags")
    if isinstance(tags, list):
        tags_text = clean_text("; ".join(str(tag) for tag in tags))
    else:
        tags_text = ""
    return {
        "created_at": iso(created),
        "model": model_id,
        "author": author,
        "downloads": downloads if isinstance(downloads, int) else "",
        "likes": likes if isinstance(likes, int) else "",
        "tags": tags_text,
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_manifest_text(path):
    """Read the manifest from disk, or fall back to the committed copy.

    The workflow checks out only ``scripts`` from the repository, so the
    manifest can be missing from the working tree even though it is committed.
    """
    manifest_path = Path(path)
    try:
        return manifest_path.read_text(encoding="utf-8")
    except OSError:
        pass
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{manifest_path.as_posix()}"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def load_manifest(path):
    text = read_manifest_text(path)
    if text is None:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"manifest {path} is not valid JSON") from error
    if not isinstance(data, dict):
        raise RuntimeError(f"manifest {path} must contain a JSON object")
    version = data.get("state_version", 1)
    if version != 1:
        raise RuntimeError(f"manifest {path} has an unsupported state version")
    return data


def save_manifest(path, manifest):
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, manifest_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        help="UTC start timestamp as ISO 8601 (default: end of the last list)",
    )
    parser.add_argument(
        "--until",
        help="UTC end timestamp as ISO 8601 (default: now)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=MAX_PAGES,
        help=f"maximum API pages to walk (default: {MAX_PAGES})",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--manifest", default="latest.json")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=1.0,
        help="window length when no previous list exists (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc)
    until = parse_timestamp(args.until) if args.until else now
    manifest = load_manifest(args.manifest)

    if args.since:
        since = parse_timestamp(args.since)
        if "window" in manifest:
            stored_window = parse_timestamp(manifest["window"])
            if since < stored_window:
                raise RuntimeError(
                    "backfill would move the window backwards; "
                    f"the manifest window is {iso(stored_window)}"
                )
    elif "window" in manifest:
        since = parse_timestamp(manifest["window"])
    else:
        since = until - dt.timedelta(hours=args.lookback_hours)

    if since >= until:
        print(f"nothing to do ({iso(since)} >= {iso(until)})", file=sys.stderr)
        return 0

    models, exhausted = fetch_new_models(
        since, args.user_agent, args.retries, max(1, args.max_pages)
    )

    rows = []
    seen = set()
    skipped = 0
    for model in models:
        if not isinstance(model, dict):
            skipped += 1
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            skipped += 1
            continue
        try:
            created = parse_timestamp(model["createdAt"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if created <= since or created > until:
            continue
        if model_id in seen:
            continue
        seen.add(model_id)
        rows.append(build_row(model, created))
    rows.sort(key=lambda row: row["created_at"])
    if skipped:
        print(f"skipped {skipped} malformed models", file=sys.stderr)

    manifest["window"] = iso(until)
    manifest["source_truncated"] = not exhausted
    if rows:
        output = Path(args.output_dir) / f"new-models-{timestamp_filename(until)}.csv"
        write_csv(output, rows)
        manifest["list"] = {
            "path": output.as_posix(),
            "from": iso(since),
            "to": iso(until),
            "count": len(rows),
        }
        print(
            f"wrote {len(rows)} models created between {iso(since)} "
            f"and {iso(until)} to {output}"
        )
    else:
        print(f"no new models between {iso(since)} and {iso(until)}")
    if not exhausted:
        print(
            "Hugging Face page limit reached; the window may be incomplete",
            file=sys.stderr,
        )
    save_manifest(args.manifest, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
