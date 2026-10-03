"""PreToolUse hook: keep Claude from reading .env-style secret files.

Reads the hook payload on stdin. To block a call it prints a JSON "deny"
decision; to allow it, it prints nothing. Exit code is always 0.

Covered:
  * file tools (Read, Edit, Write, NotebookEdit, Grep, Artifact, MCP tools):
    any path-like argument that names or resolves to an env file
  * shell tools (Bash, PowerShell): file arguments, quote/escape/concat
    tricks, globs and brace expansions that expand to an env file, symlinks,
    junctions, 8.3 short names, hard links, recursive searches (grep -r,
    rg --hidden, findstr /s, gci -Recurse | sls, find -exec), dotenv/direnv
    loaders, and git commands that would print a tracked/committed env file.

Not covered (no hook can fully stop these): a program that loads .env
internally (e.g. `python app.py` calling load_dotenv), and deliberately
obfuscated commands (base64-decoded paths, etc.).
"""

import fnmatch
import glob
import json
import os
import re
import subprocess
import sys
import time

SAFE_SUFFIXES = {"example", "examples", "sample", "template", "tmpl", "dist", "schema"}
# `process.env`, `os.env`, `import.meta.env` show up in code, not as files.
CODE_STEMS = {"process", "os", "import.meta", "self", "this"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
             ".ruff_cache", ".pytest_cache", ".tox"}
GLOB_CHARS = set("*?[{")

CWD = os.getcwd()


# ---------- names and paths ----------

def _is_env_core(n):
    n = n.strip().lower().rstrip(". ")  # Windows ignores trailing dots/spaces
    if not n:
        return False
    if n in (".env", ".envrc"):
        return True
    m = re.fullmatch(r"\.env[.\-_](.+)", n)
    if m:
        return re.split(r"[.\-_]", m.group(1))[-1] not in SAFE_SUFFIXES
    if n.endswith(".env"):
        stem = n[:-4]
        return bool(stem) and stem not in CODE_STEMS and stem not in SAFE_SUFFIXES
    return False


def is_env_name(name):
    # ':' covers NTFS streams (.env::$DATA) and git object refs (HEAD:.env).
    return any(_is_env_core(part) for part in name.split(":"))


def _abs(p):
    p = os.path.expanduser(p)
    if os.name == "nt":  # Git Bash / Cygwin style: /c/Users/... or /cygdrive/c/...
        m = re.match(r"^(?:/cygdrive)?/([a-zA-Z])(?=/|$)", p.replace("\\", "/"))
        if m:
            p = m.group(1) + ":" + (p.replace("\\", "/")[m.end():] or "/")
    return p if os.path.isabs(p) else os.path.join(CWD, p)


def _same_file_as_env_nearby(full):
    try:
        st = os.stat(full)
    except OSError:
        return False
    if st.st_nlink < 2:
        return False
    for d in {os.path.dirname(os.path.abspath(full)), CWD}:
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if is_env_name(name):
                try:
                    s2 = os.stat(os.path.join(d, name))
                except OSError:
                    continue
                if (s2.st_ino, s2.st_dev) == (st.st_ino, st.st_dev):
                    return True
    return False


def path_is_secret(p):
    p = p.strip().strip("\"'")
    if not p:
        return False
    base = p.replace("\\", "/").rstrip("/").split("/")[-1]
    full = _abs(p)
    try:
        if os.path.isdir(full):  # e.g. a virtualenv named .env
            return False
    except (OSError, ValueError):
        pass
    if is_env_name(base):
        return True
    try:
        if os.path.exists(full):
            # realpath resolves symlinks, junctions and 8.3 short names.
            if is_env_name(os.path.basename(os.path.realpath(full))):
                return True
            if _same_file_as_env_nearby(full):
                return True
    except (OSError, ValueError):
        pass
    return False


def env_files_under(root, limit=20000, budget=1.5):
    """Env files under root. Returns (files, truncated)."""
    root = _abs(root)
    if os.path.isfile(root):
        return ([root] if path_is_secret(root) else []), False
    found, seen, t0 = [], 0, time.time()
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d.lower() not in SKIP_DIRS
                  and not os.path.exists(os.path.join(dp, d, "pyvenv.cfg"))]
        found += [os.path.join(dp, f) for f in fns if is_env_name(f)]
        seen += len(fns) + len(dns)
        if seen > limit or time.time() - t0 > budget:
            return found, True
    return found, False


def env_files_upward(start):
    """Env files in start and its parents (where dotenv loaders look)."""
    d = _abs(start)
    while True:
        try:
            if any(is_env_name(n) and os.path.isfile(os.path.join(d, n)) for n in os.listdir(d)):
                return True
        except OSError:
            pass
        parent = os.path.dirname(d)
        if parent == d:
            return False
        d = parent


def expand_braces(s, depth=0):
    m = re.search(r"\{([^{}]*,[^{}]*)\}", s)
    if not m or depth > 6:
        return [s]
    out = []
    for alt in m.group(1).split(","):
        out += expand_braces(s[:m.start()] + alt + s[m.end():], depth + 1)
    return out[:200]


def glob_hits_secret(token):
    for pat in expand_braces(token):
        if not set(pat) & set("*?["):
            if path_is_secret(pat):
                return True
            continue
        try:
            hits = glob.glob(_abs(pat), recursive=True, include_hidden=True)
        except (OSError, ValueError, re.error):
            continue
        if any(path_is_secret(h) for h in hits[:5000]):
            return True
    return False


# ---------- shell analysis ----------

EXCLUDE_ARG = re.compile(
    r"(--exclude(?:-dir)?|--glob|--iglob|-g|-Exclude|-iglob)(?:\s*=\s*|\s+)(\S+)",
    re.IGNORECASE)
SEARCHERS = {"grep", "egrep", "fgrep", "rg", "ag", "ack", "findstr", "select-string",
             "sls", "gci", "get-childitem", "ls", "dir", "find", "tar", "zip", "rsync", "git"}
LISTERS = {"gci", "get-childitem", "ls", "dir", "find"}
READERS = {"select-string", "sls", "gc", "get-content", "cat", "type", "findstr", "xargs",
           "foreach-object", "%", "more", "grep", "rg", "head", "tail", "less"}


def shell_tokens(text):
    variants = [
        text.replace("\\", "/"),
        re.sub(r"[\"'`^+]", "", text).replace("\\", "/"),
        re.sub(r"[\"'`^+\\]", "", text),  # bash escapes like .e\nv
    ]
    out = set()
    # Split with and without commas: commas separate code args but also live
    # inside brace expansions like .{e,x}nv.
    pieces = [t for v in variants for sep in (r"[\s;|&<>(),=\"'`]+", r"[\s;|&<>()=\"'`]+")
              for t in re.split(sep, v)]
    for t in pieces:
        t = t.lstrip("@")
        if t.startswith("-") and ":" in t:  # PowerShell -Path:.env
            t = t.split(":", 1)[1]
        if t:
            out.add(t)
    return out


def words(segment):
    return [w for w in re.split(r"[\s\"'`]+", segment) if w]


def cmd_name(w):
    w = w.lstrip("({!").strip()  # subshells/groups: (grep ...), { cat ...; }
    return re.sub(r"\.(exe|cmd|bat|ps1)$", "", w.replace("\\", "/").split("/")[-1].lower())


def recursive_read(segment, pipeline):
    w = words(segment)
    if not w:
        return False
    c, flags = cmd_name(w[0]), [x.lower() for x in w[1:]]
    if c in ("grep", "egrep", "fgrep"):
        return any(re.fullmatch(r"-[a-z]*r[a-z]*", f, re.I) or f in ("--recursive", "--dereference-recursive")
                   for f in w[1:])
    if c in ("rg", "ripgrep"):
        u = sum(f.count("u") for f in flags if re.fullmatch(r"-u+", f))
        return "--hidden" in flags or "-." in flags or "--no-ignore-dot" in flags or u >= 2
    if c == "ag":
        return "--hidden" in flags or "-u" in flags or "--unrestricted" in flags
    if c == "ack":
        return True
    if c == "findstr":
        return any(f in ("/s", "-s") or re.fullmatch(r"/[a-z]*s[a-z]*", f) for f in flags)
    if c == "find":
        return "-exec" in flags or "-execdir" in flags or "-ok" in flags
    if c == "git" and "grep" in flags:
        return "--untracked" in flags or "--no-index" in flags
    if c in LISTERS:
        recurse = any(f.startswith("-rec") or f in ("-r", "/s", "-s") for f in flags)
        if recurse:
            i = pipeline.index(segment) if segment in pipeline else -1
            later = pipeline[i + 1:] if i >= 0 else []
            return any(words(s) and cmd_name(words(s)[0]) in READERS for s in later)
    return False


def segment_roots(segment):
    roots = []
    for w in words(segment)[1:]:
        if w.startswith("-") or (w.startswith("/") and len(w) <= 3):
            continue
        try:
            if os.path.isdir(_abs(w)):
                roots.append(w)
        except (OSError, ValueError):
            pass
    return roots or ["."]


# Code or CLIs that load an env file -- not package names like python-dotenv.
DOTENV_LOADER = re.compile(
    r"load_dotenv|dotenv_values|find_dotenv|dotenv/config|dotenv\.config|"
    r"\bdotenv\s+(?:run|get|list|set|-e|-f|-c|--)|\bdirenv\s+(?:exec|export|allow|reload)",
    re.IGNORECASE)

GIT_CONTENT_CMDS = {"diff", "show", "log", "stash", "blame", "grep", "cat-file", "archive",
                    "format-patch", "whatchanged", "annotate"}


def git_history_has_env():
    try:
        r = subprocess.run(
            ["git", "-C", CWD, "log", "--all", "--pretty=format:", "--name-only", "--",
             ":(glob)**/.env*", ":(glob)**/*.env", ":(glob)**/.envrc"],
            capture_output=True, text=True, timeout=4)
    except (OSError, subprocess.SubprocessError):
        return False
    return any(is_env_name(line.strip().split("/")[-1]) for line in r.stdout.splitlines() if line.strip())


def git_risky(segment):
    w = words(segment)
    if not w or cmd_name(w[0]) != "git":
        return False
    i = 1
    while i < len(w) and w[i].startswith("-"):
        i += 2 if w[i] in ("-C", "-c", "--git-dir", "--work-tree") else 1
    if i >= len(w):
        return False
    sub, rest = w[i].lower(), w[i + 1:]
    if sub not in GIT_CONTENT_CMDS:
        return False
    if sub == "log" and not any(f in ("-p", "--patch", "-u") or f.startswith("--patch") for f in rest):
        return False
    if sub == "stash" and not any(f in ("-p", "--patch") for f in rest):
        return False
    if "--" in rest:
        specs = rest[rest.index("--") + 1:]
        if specs and not any(set(s) & GLOB_CHARS or s in (".", "/", ":/") for s in specs):
            return False  # explicit pathspecs; token checks already vetted them
    else:
        args = [a for a in rest if not a.startswith("-")]
        if sub in ("show", "cat-file") and any(":" in a for a in args):
            return False  # explicit HEAD:file objects, vetted by token checks
        try:
            if any(os.path.isfile(_abs(a)) for a in args):
                return False
        except (OSError, ValueError):
            pass
    return git_history_has_env()


def check_shell(cmd):
    pipelines = re.split(r"\n|;|&&|\|\||(?<![|>])&(?!&)", cmd)
    for pipe_text in pipelines:
        segs = [s.strip() for s in pipe_text.split("|") if s.strip()]
        for seg in segs:
            excluded_env = False
            first = cmd_name(words(seg)[0]) if words(seg) else ""
            text = seg
            if first in SEARCHERS:
                def _drop(m):
                    nonlocal excluded_env
                    if "env" in m.group(2).lower():
                        excluded_env = True
                    return " "
                text = EXCLUDE_ARG.sub(_drop, seg)

            for t in shell_tokens(text):
                if set(t) & GLOB_CHARS:
                    if first in LISTERS and all(words(s) and cmd_name(words(s)[0]) in LISTERS for s in segs):
                        continue  # names only
                    if glob_hits_secret(t):
                        return f"`{t}` expands to an env file"
                elif path_is_secret(t):
                    return f"`{t}` is an env file"

            if recursive_read(seg, segs) and not excluded_env:
                for root in segment_roots(seg):
                    files, truncated = env_files_under(root)
                    if files or truncated:
                        return ("recursive read over a tree that contains env files; "
                                "add an exclusion such as --exclude='.env*' / -g '!.env*' / -Exclude .env*")

            if DOTENV_LOADER.search(seg) and env_files_upward("."):
                return "dotenv/direnv loader would read an env file"

            if git_risky(seg):
                return "git command would print an env file that is tracked or in history; pass explicit paths after --"
    return None


# ---------- tool inputs ----------

PATH_KEY = re.compile(r"(path|file|dir|root|from|source|target)", re.I)


def path_values(obj, key_ok=False):
    if isinstance(obj, str):
        if key_ok:
            yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from path_values(v, key_ok or bool(PATH_KEY.search(str(k))))
    elif isinstance(obj, list):
        for v in obj:
            yield from path_values(v, key_ok)


def check_grep(ti):
    path = ti.get("path") or "."
    if path_is_secret(path):
        return f"`{path}` is an env file"
    g, typ = ti.get("glob"), ti.get("type")
    if ti.get("output_mode", "files_with_matches") in ("files_with_matches", "count"):
        return None  # file names / counts only
    if typ and typ.lower() not in ("env", "dotenv", "sh", "bash"):
        return None
    files, truncated = env_files_under(path)
    if g:
        pats = [p.lower() for p in expand_braces(g) if p]
        if any(p.startswith("!") and "env" in p for p in pats):
            return None
        pos = [p.lstrip("/") for p in pats if not p.startswith("!")]
        if pos:
            def matches(f, p):
                rel = os.path.relpath(f, _abs(path)).replace("\\", "/").lower()
                return fnmatch.fnmatch(os.path.basename(f).lower(), p.split("/")[-1]) or fnmatch.fnmatch(rel, p)
            hit = any(matches(f, p) for f in files for p in pos)
            if truncated:
                hit = hit or any(fnmatch.fnmatch(".env", p.split("/")[-1]) for p in pos)
            return ("content search would include env files; set glob to '!**/.env*' or narrow the path/type"
                    if hit else None)
    if files or truncated:
        return "content search would include env files; set glob to '!**/.env*' or narrow the path/type"
    return None


def decide(payload):
    global CWD
    CWD = payload.get("cwd") or CWD
    tool = payload.get("tool_name", "")
    ti = payload.get("tool_input") or {}

    if tool in ("Bash", "PowerShell"):
        return check_shell(ti.get("command", ""))
    if tool == "Grep":
        return check_grep(ti)
    for p in path_values(ti):
        if p and len(p) < 4096 and (path_is_secret(p) or (set(p) & set("*?[") and glob_hits_secret(p))):
            return f"`{p}` is an env file"
    return None


def main():
    raw = sys.stdin.buffer.read().decode("utf-8", "replace")
    try:
        reason = decide(json.loads(raw))
    except Exception as e:  # fail closed only when env files could be involved
        reason = f"env guard errored ({type(e).__name__}); failing closed" if "env" in raw.lower() else None
    if reason:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"Blocked by the user's env-file guard: {reason}. The user does not want "
                "Claude to read .env files. Do not work around this; ask the user for any value you need."),
        }}))


if __name__ == "__main__":
    main()
