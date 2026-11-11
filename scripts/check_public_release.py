"""Read-only public release scan; never prints matched secret values."""
import argparse, json, re, subprocess
from pathlib import Path
from urllib.parse import urlsplit, unquote
ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"portfolio.html", "portpolio.html", "architecture.html", "tech.html", "notion-daily-system.md", "outputs-all", "모델_스크립트", "코드기반_분석_MD", "credentials", "secrets", "models", "data", "recordings", ".aws", ".codex", ".agents"}
PATTERNS = {
    "jwt": r"eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}",
    "google_api_key": r"AIza[\w-]{30,}",
    "private_key": r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----",
    "supabase_secret": r"sb_secret_[A-Za-z0-9_-]{15,}",
}
PLACEHOLDERS = {"YOUR_DB_PASSWORD", "YOUR_PASSWORD", "password", "example", "REPLACE_ME"}
def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)
def secret_kinds(data):
    if b"\0" in data: return set()
    text = data.decode("utf-8", errors="replace")
    kinds = {name for name, pattern in PATTERNS.items() if re.search(pattern, text)}
    for match in re.finditer(r"postgres(?:ql)?(?:\+\w+)?://[^\s\x22\x27<>]+", text):
        try: password = unquote(urlsplit(match.group().replace("postgresql+psycopg:", "postgresql:")).password or "")
        except ValueError:
            kinds.add("malformed_database_url")
            continue
        if password and password not in PLACEHOLDERS and not password.startswith(("${", "<", "[")): kinds.add("database_password")
    return kinds
def excluded(path):
    p = Path(path)
    return (p.parts[0] in EXCLUDED or (p.name.startswith(".env") and p.name != ".env.example") or p.suffix.lower() in {".pem", ".key"} or (p.name.startswith("service-account") and p.suffix == ".json") or p.name.endswith("-service-account.json"))
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Inspect all blobs reachable from local refs.")
    args = parser.parse_args()
    findings = []
    paths = [p.decode("utf-8") for p in git("ls-files", "-z").split(b"\0") if p]
    for path in paths:
        if excluded(path): findings.append({"scope": "current", "path": path, "kind": "excluded_path_tracked"})
        file = ROOT / path
        if file.is_file():
            for kind in sorted(secret_kinds(file.read_bytes())): findings.append({"scope": "current", "path": path, "kind": kind})
    count = 0
    if args.history:
        for row in git("rev-list", "--objects", "--all").splitlines():
            oid, _, raw_path = row.partition(b" ")
            oid = oid.decode("ascii")
            if git("cat-file", "-t", oid).strip() != b"blob": continue
            count += 1
            for kind in sorted(secret_kinds(git("cat-file", "blob", oid))):
                findings.append({"scope": "history", "object": oid, "path": raw_path.decode("utf-8", errors="replace"), "kind": kind})
    print(json.dumps({"tracked_files_checked": len(paths), "history_blobs_checked": count, "findings": findings, "result": "BLOCKED" if findings else "NO_PATTERN_FINDINGS", "limits": "Pattern scan only. Current scan reads working copies of tracked files, not a pending commit."}, ensure_ascii=False, indent=2))
    return 1 if findings else 0
if __name__ == "__main__":
    raise SystemExit(main())
