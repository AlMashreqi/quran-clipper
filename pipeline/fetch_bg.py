"""pipeline/fetch_bg.py -- download a background image from a URL into a source workspace.

Usage:
    python3 pipeline/fetch_bg.py <url> [sources/<id>]
    python3 pipeline/fetch_bg.py <url> --out sources/<id>/bg.jpg

Examples:
    python3 pipeline/fetch_bg.py "https://i.pinimg.com/736x/0c/be/67/0cbe6740119349801e938db96123f204.jpg" sources/bSK6owbCMFA
"""
import argparse
import os
import subprocess
import sys
from PIL import Image

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def fetch_image(url, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = ["curl", "-sL", "-A", USER_AGENT, "-o", out_path, url]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit("Error fetching image from %s: %s" % (url, e))

    try:
        with Image.open(out_path) as img:
            img.verify()
            w, h = img.size
        print("Fetched image -> %s (%dx%d)" % (out_path, w, h))
    except Exception as err:
        if os.path.exists(out_path):
            os.remove(out_path)
        sys.exit("Downloaded file from %s is not a valid image: %s" % (url, err))
    return out_path


def main():
    ap = argparse.ArgumentParser(description="fetch a background image from URL")
    ap.add_argument("url", help="Image URL (e.g. Pinterest, direct link)")
    ap.add_argument("source", nargs="?", default=None, help="sources/<id> directory or file path")
    ap.add_argument("-o", "--out", help="explicit output file path")
    args = ap.parse_args()

    out_path = args.out
    if not out_path:
        target_dir = args.source or "."
        if os.path.isfile(target_dir):
            target_dir = os.path.dirname(target_dir)
        ext = ".png" if ".png" in args.url.lower() else ".jpg"
        out_path = os.path.join(target_dir, "bg" + ext)

    fetch_image(args.url, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
