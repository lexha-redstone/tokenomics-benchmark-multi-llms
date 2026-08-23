#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Fetch the nltk data ClassEval needs, without using nltk's own downloader.

Why this exists
---------------
`nltk.download()` refuses to run behind a proxy. Recent nltk versions validate
the resolved IP of the host they fetch from, and a proxy performs the egress on
their behalf, so the pin cannot be enforced:

    Security Violation [pathsec.urlopen]: refusing a proxied fetch of
    '.../nltk_data/gh-pages/index.xml'. A configured proxy performs the egress,
    so NLTK cannot pin the validated IP and SSRF protection cannot be enforced
    (CWE-918).

That is nltk protecting itself against a class of attack, and switching it off
with NLTK_ALLOW_PROXIED_URLOPEN=1 means asserting your proxy is SSRF-safe --
somebody else's call to make, not this script's.

The data itself is plain zip files in a public GitHub repository. Fetching them
over ordinary HTTPS, which honours HTTPS_PROXY like any other client, needs none
of that machinery: no index.xml, no IP pinning, no opt-out of a security
control.

**Which resources** is not hardcoded. nltk renamed them between releases --
`pos_tag` wanted `averaged_perceptron_tagger` before 3.9 and
`averaged_perceptron_tagger_eng` after, and `word_tokenize` moved from `punkt`
to `punkt_tab` -- so a fixed list is wrong on some installed version whichever
list you pick. The installed nltk is asked what it wants, by running the
operations ClassEval performs and reading the resource out of the LookupError.

Usage:
    python3 tools/fetch_nltk_data.py                 # report what is missing
    python3 tools/fetch_nltk_data.py --install       # fetch and unpack it
    python3 tools/fetch_nltk_data.py --install --dest ~/nltk_data
    python3 tools/fetch_nltk_data.py --install --resource taggers/punkt_tab
"""

import argparse
import io
import os
import ssl
import sys
import urllib.error
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.datasets import classeval_nltk_gaps
from src.paths import display as rel

BASE = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages"

# Where a resource lives when the LookupError did not say. Only used as a
# fallback; the error's "Attempted to load 'taggers/x/'" line is authoritative.
FALLBACK_COLLECTION = {
    "punkt": "tokenizers", "punkt_tab": "tokenizers",
    "averaged_perceptron_tagger": "taggers",
    "averaged_perceptron_tagger_eng": "taggers",
    "wordnet": "corpora", "omw-1.4": "corpora", "stopwords": "corpora",
}


def _ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def default_dest():
    """First writable directory nltk already searches.

    Preferring a directory nltk searches by default matters more than it looks:
    candidate programs run in a sandboxed child process, and one that has to be
    told where the data is via NLTK_DATA will not find data unpacked somewhere
    exotic.
    """
    try:
        import nltk
        candidates = list(nltk.data.path)
    except Exception:
        candidates = []
    candidates.append(os.path.join(os.path.expanduser("~"), "nltk_data"))
    for path in candidates:
        parent = os.path.dirname(os.path.abspath(path)) or os.sep
        if os.path.isdir(path) and os.access(path, os.W_OK):
            return path
        if os.path.isdir(parent) and os.access(parent, os.W_OK):
            return path
    return os.path.join(os.path.expanduser("~"), "nltk_data")


def proxy_note():
    seen = {k: os.environ[k] for k in
            ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "NO_PROXY")
            if os.environ.get(k)}
    if not seen:
        return "no proxy configured in the environment"
    return "using proxy from " + ", ".join(sorted(seen))


def fetch(collection, resource, dest, ctx):
    """Download one package zip and unpack it under ``dest/<collection>``."""
    url = f"{BASE}/{collection}/{resource}.zip"
    target = os.path.join(dest, collection)
    os.makedirs(target, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180, context=ctx) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(target)
    return url, len(blob), os.path.join(target, resource)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true",
                    help="actually download and unpack (default: report only)")
    ap.add_argument("--dest", default=None,
                    help="nltk_data directory (default: the first writable one "
                         "nltk already searches)")
    ap.add_argument("--resource", action="append", default=[],
                    help="explicit '<collection>/<id>', repeatable. Overrides "
                         "the probe -- use it to fetch something this dataset "
                         "does not itself require.")
    args = ap.parse_args()

    if args.resource:
        wanted = []
        for spec in args.resource:
            if "/" not in spec:
                coll = FALLBACK_COLLECTION.get(spec)
                if not coll:
                    print(f"cannot place '{spec}': give it as '<collection>/{spec}'")
                    return 1
                spec = f"{coll}/{spec}"
            coll, rid = spec.split("/", 1)
            wanted.append({"collection": coll, "resource": rid,
                           "needed_by": "requested"})
    else:
        try:
            import nltk  # noqa: F401
        except Exception:
            print("nltk is not installed. pip install -r "
                  f"{rel(os.path.join(ROOT, 'classeval', 'requirements.txt'))}")
            return 1
        wanted = classeval_nltk_gaps()

    if not wanted:
        print("nltk data: everything ClassEval needs is already present")
        return 0

    dest = os.path.abspath(os.path.expanduser(args.dest or default_dest()))
    print(f"nltk data: {len(wanted)} resource(s) missing")
    for w in wanted:
        coll = w.get("collection") or FALLBACK_COLLECTION.get(w["resource"], "corpora")
        w["collection"] = coll
        print(f"  {coll}/{w['resource']:<34} needed by {w['needed_by']}")
    print(f"  destination: {dest}")
    print(f"  {proxy_note()}")

    if not args.install:
        print("\nreport only. Re-run with --install to fetch them.")
        print("Alternative, if your proxy is known to be SSRF-safe, is to let "
              "nltk\nfetch them itself: NLTK_ALLOW_PROXIED_URLOPEN=1 "
              "python3 -c \"import nltk; nltk.download('...')\"")
        return 0

    failures = []
    for w in wanted:
        try:
            url, size, where = fetch(w["collection"], w["resource"], dest, _ssl_ctx())
            print(f"  fetched {size / 1e6:6.2f} MB -> {where}")
        except urllib.error.HTTPError as e:
            failures.append((w["resource"], f"HTTP {e.code} for {BASE}/"
                                            f"{w['collection']}/{w['resource']}.zip"))
        except Exception as e:
            failures.append((w["resource"], f"{type(e).__name__}: {e}"))

    if failures:
        print("\nfailed:")
        for rid, why in failures:
            print(f"  {rid}: {why}")
        print("\nIf the proxy blocks raw.githubusercontent.com, download the zip "
              "on a\nmachine that can reach it and unpack it into "
              f"{dest}/<collection>/.")
        return 1

    # nltk caches the directory listing it searched; a fresh interpreter is the
    # honest check that the data is really findable now.
    print("\nverifying in a fresh interpreter ...")
    check = os.path.join(ROOT, "tools", "fetch_nltk_data.py")
    import subprocess
    r = subprocess.run([sys.executable, check], capture_output=True, text=True,
                       env={**os.environ, "NLTK_DATA": dest})
    print("  " + (r.stdout.strip().splitlines() or ["(no output)"])[0])
    return 0 if "already present" in r.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
