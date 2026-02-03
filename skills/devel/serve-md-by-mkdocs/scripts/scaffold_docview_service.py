from __future__ import annotations

import argparse
import re
import subprocess
import textwrap
from pathlib import Path

DEFAULT_IMAGE_EXTENSIONS = (
    ".avif",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".tif",
    ".tiff",
    ".webp",
)


def _yaml_single_quote(value: str) -> str:
    # YAML single-quoted scalars escape a single quote by doubling it.
    return "'" + value.replace("'", "''") + "'"


def _default_dirs_env_var(service_dir: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", Path(service_dir).name).strip("_").upper()
    if not slug:
        slug = "DOCVIEW"
    if slug[0].isdigit():
        slug = f"DOCVIEW_{slug}"
    return f"{slug}_DIRS"

def _default_repo_root_env_var(dirs_env_var: str) -> str:
    if dirs_env_var.endswith("_DIRS"):
        return f"{dirs_env_var[:-5]}_REPO_ROOT"
    return f"{dirs_env_var}_REPO_ROOT"


def _detect_repo_root(start_dir: Path) -> Path:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(start_dir), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return start_dir.resolve()
    if not output:
        return start_dir.resolve()
    return Path(output).resolve()


def _write_file(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _service_gitignore(service_name: str, staging_dir: str) -> str:
    return textwrap.dedent(
        f"""\
        # Generated staged docs tree (symlinks/copies)
        {staging_dir}/

        # MkDocs build output
        site/

        # Generated MkDocs config (created by refresh-docs-tree.sh)
        mkdocs.yml

        # Cached repo root (created at scaffold time; override via env var); usually machine-specific
        repo-root.txt

        # Background serve helpers/logs (local only)
        {service_name}-serve.pid
        {service_name}-serve.log
        """
    )


def _docview_manifest_yaml(
    *,
    site_name: str,
    dev_addr: str | None,
    staging_dir: str,
    default_targets: list[str],
) -> str:
    dev_addr_yaml = _yaml_single_quote(dev_addr) if dev_addr is not None else "null"
    targets = default_targets or ["."]

    include_globs: list[str] = []
    for t in targets:
        t = t.strip().rstrip("/")
        if not t:
            continue
        if t in (".", "./"):
            include_globs.append("*.md")
            include_globs.append("**/*.md")
            continue
        if t.lower().endswith(".md"):
            include_globs.append(t)
            continue
        include_globs.append(f"{t}/**/*.md")

    if not include_globs:
        include_globs = ["**/*.md"]

    lines: list[str] = []
    lines.append("# DocView manifest (v1)")
    lines.append("#")
    lines.append("# This file captures your “what should be staged” intent.")
    lines.append("# - scan.include_globs: glob patterns for Markdown discovery (defaults)")
    lines.append("# - scan.force_globs: glob patterns scanned even if gitignored (explicit user intent)")
    lines.append("# - scan.exclude_globs: glob patterns excluded from Markdown discovery")
    lines.append("#")
    lines.append("# Notes:")
    lines.append("# - By default, discovery respects .gitignore (gitignored dirs/files are not scanned for Markdown).")
    lines.append("# - Referenced image assets from selected Markdown are still staged even if gitignored.")
    lines.append("")
    lines.append("version: 1")
    lines.append("")
    lines.append("staging:")
    lines.append(f"  dir: {_yaml_single_quote(staging_dir)}")
    lines.append("")
    lines.append("mkdocs:")
    lines.append(f"  site_name: {_yaml_single_quote(site_name)}")
    lines.append(f"  dev_addr: {dev_addr_yaml}")
    lines.append("")
    lines.append("scan:")
    lines.append("  respect_gitignore: true")
    lines.append("  include_hidden: false")
    lines.append("  auto_exclude_generated: true")
    lines.append("  include_globs:")
    for g in include_globs:
        lines.append(f"    - {g}")
    lines.append("  exclude_globs: []")
    lines.append("  force_globs: []")
    lines.append("")
    lines.append("assets:")
    lines.append("  include_images: true")
    lines.append("  image_extensions:")
    for ext in DEFAULT_IMAGE_EXTENSIONS:
        lines.append(f"    - {ext}")
    lines.append("")
    return "\n".join(lines)

def _scan_files_script() -> str:
    # Written into the work dir so the service is self-contained and easy to copy/move.
    #
    # Scope: discover Markdown files under user-selected roots, then discover local image assets
    # referenced by those Markdown files, and output repo-relative file paths (null-delimited by default).
    #
    # This intentionally uses only stdlib.
    return textwrap.dedent(
        r"""\
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
import posixpath
import subprocess


DEFAULT_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
}


def is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _is_git_repo(repo_root: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _git_ls_files(repo_root: Path, args: list[str]) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    out = proc.stdout
    if not out:
        return []
    return [p for p in out.split("\0") if p]


def _is_hidden_rel_posix(rel_posix: str) -> bool:
    parts = [p for p in rel_posix.split("/") if p]
    return any(p.startswith(".") for p in parts)


def _under_excludes(path_abs: Path, exclude_abs: set[Path]) -> bool:
    return any(path_abs == ex or ex in path_abs.parents for ex in exclude_abs)


def _looks_like_generated_docs_view_workdir(dir_path: Path) -> bool:
    # Heuristic for excluding DocView/MDView workdirs (and their generated trees) from discovery.
    #
    # New-style DocView workdirs always include:
    # - docview.yml
    # - refresh-docs-tree.sh
    # - scan-files-to-stage.py
    #
    # Legacy-style viewers often include:
    # - refresh-docs-tree.sh
    # - mkdocs.yml (generated by refresh)
    #
    # We intentionally keep this conservative: "if it looks like a viewer service directory,
    # exclude the entire directory tree by default".
    if not dir_path.is_dir():
        return False
    try:
        dir_path = dir_path.resolve()
    except Exception:
        return False

    if (dir_path / "docview.yml").is_file() and (dir_path / "refresh-docs-tree.sh").is_file():
        if (dir_path / "scan-files-to-stage.py").is_file():
            return True

    if (dir_path / "refresh-docs-tree.sh").is_file() and (dir_path / "mkdocs.yml").is_file():
        # Extra hint: these workdirs usually create one of these subdirs.
        if (dir_path / "site").exists() or (dir_path / "docs").exists() or (dir_path / "_staged").exists():
            return True
        gi = dir_path / ".gitignore"
        if gi.is_file():
            try:
                txt = gi.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                txt = ""
            if "site/" in txt and ("docs/" in txt or "_staged/" in txt):
                return True

    return False


def discover_generated_docs_view_workdirs(repo_root: Path, *, max_depth: int = 4) -> set[Path]:
    # Find DocView/MDView workdirs inside repo_root to avoid indexing:
    # - generated staged trees (symlinks)
    # - other viewer configs/logs
    #
    # This is a best-effort filesystem walk (not git-based) so it also catches untracked workdirs.
    repo_root = repo_root.resolve()
    if not repo_root.is_dir():
        return set()

    def is_prunable_dir(name: str) -> bool:
        return name.startswith(".") or name in {
            "__pycache__",
            ".git",
            ".pixi",
            ".mypy_cache",
            ".ruff_cache",
            "_staged",
            "site",
            "node_modules",
        }

    out: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(repo_root):
        try:
            rel = Path(dirpath).resolve().relative_to(repo_root)
        except Exception:
            continue
        if len(rel.parts) > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not is_prunable_dir(d)]

        if "refresh-docs-tree.sh" not in filenames:
            continue
        d = Path(dirpath)
        if _looks_like_generated_docs_view_workdir(d):
            out.add(d.resolve())
            # Avoid descending into this workdir (it may contain a large staged tree).
            dirnames[:] = []

    return out


def iter_markdown_files_walk(
    repo_root: Path, targets: list[str], *, exclude_abs: set[Path], include_hidden: bool
) -> list[Path]:
    markdown_paths: list[Path] = []

    for target in targets:
        src = (repo_root / target).resolve()
        if not src.exists():
            continue

        if any(src == ex or ex in src.parents for ex in exclude_abs):
            continue

        if src.is_file():
            if src.suffix.lower() == ".md":
                markdown_paths.append(src)
            continue

        for root, dirs, files in os.walk(src):
            root_path = Path(root)
            # prune dirs
            pruned: list[str] = []
            for d in dirs:
                dp = root_path / d
                try:
                    rel = dp.relative_to(repo_root)
                except Exception:
                    continue
                if not include_hidden and is_hidden_path(rel):
                    continue
                if any(dp == ex or ex in dp.parents for ex in exclude_abs):
                    continue
                pruned.append(d)
            dirs[:] = pruned

            for fname in files:
                if not fname.lower().endswith(".md"):
                    continue
                fpath = (root_path / fname).resolve()
                if any(fpath == ex or ex in fpath.parents for ex in exclude_abs):
                    continue
                markdown_paths.append(fpath)

    return sorted(set(markdown_paths))


def iter_markdown_files_git(
    repo_root: Path,
    targets: list[str],
    *,
    forced_targets: list[str],
    exclude_abs: set[Path],
    include_hidden: bool,
) -> list[Path]:
    # Respect gitignore by using git's file listing.
    if not targets:
        return []

    rels: set[str] = set()
    rels |= set(_git_ls_files(repo_root, ["--", *targets]))  # tracked
    rels |= set(_git_ls_files(repo_root, ["-o", "--exclude-standard", "--", *targets]))  # untracked, not ignored

    if forced_targets:
        # Ignored, untracked (only when user explicitly forces a target).
        rels |= set(_git_ls_files(repo_root, ["-o", "-i", "--exclude-standard", "--", *forced_targets]))

    md_paths: list[Path] = []
    for rel in sorted(rels):
        if not rel.lower().endswith(".md"):
            continue
        if not include_hidden and _is_hidden_rel_posix(rel):
            continue
        abs_path = (repo_root / rel).resolve()
        if _under_excludes(abs_path, exclude_abs):
            continue
        if abs_path.is_file():
            md_paths.append(abs_path)
    return md_paths


def _strip_angle(s: str) -> str:
    s = s.strip()
    if s.startswith("<") and s.endswith(">"):
        return s[1:-1].strip()
    return s


def _extract_md_destinations(markdown_text: str) -> list[str]:
    # Minimal, heuristic extraction of image paths:
    # - Markdown image: ![alt](path "title")
    # - HTML image: <img src="path">
    # - Reference definitions: [id]: path
    #
    # We only keep local-looking paths; URLs/data/mailto are ignored.
    candidates: list[str] = []

    for m in re.finditer(r"!\[[^\]]*]\(([^)]+)\)", markdown_text):
        raw = _strip_angle(m.group(1))
        raw = raw.strip()
        # drop title if present (best-effort)
        if " " in raw and not raw.startswith(("http://", "https://", "data:", "mailto:")):
            raw = raw.split(" ", 1)[0]
        candidates.append(raw)

    for m in re.finditer(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", markdown_text, flags=re.IGNORECASE):
        candidates.append(m.group(1).strip())

    for line in markdown_text.splitlines():
        m = re.match(r"^\s*\[[^\]]+]:\s*(\S+)\s*(?:\".*\")?\s*$", line)
        if m:
            candidates.append(_strip_angle(m.group(1)))

    return candidates


def _is_external_or_anchor(path: str) -> bool:
    p = path.strip()
    if not p:
        return True
    if p.startswith("#"):
        return True
    lower = p.lower()
    if lower.startswith(("http://", "https://", "data:", "mailto:")):
        return True
    return False


def _normalize_repo_relative(repo_rel: PurePosixPath) -> PurePosixPath | None:
    # Normalize while preventing escape from repo root.
    s = str(repo_rel)
    if repo_rel.is_absolute():
        s = s.lstrip("/")
    s = s.replace("\\\\", "/")
    norm = PurePosixPath(posixpath.normpath(s))
    # Reject paths that escape repo root (e.g. ../../x.png).
    if norm.parts and norm.parts[0] == "..":
        return None
    if str(norm) == ".":
        return None
    return norm


def iter_referenced_image_assets(
    repo_root: Path, markdown_files: list[Path], *, image_exts: set[str]
) -> tuple[set[Path], list[str]]:
    assets: set[Path] = set()
    warnings: list[str] = []

    for md_abs in markdown_files:
        try:
            text = md_abs.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            warnings.append(f"scan: failed to read markdown: {md_abs}: {e}")
            continue

        try:
            md_rel = PurePosixPath(md_abs.relative_to(repo_root).as_posix())
        except Exception:
            # not under repo root
            continue

        md_dir = md_rel.parent
        for raw in _extract_md_destinations(text):
            if _is_external_or_anchor(raw):
                continue

            ref = raw.split("#", 1)[0].split("?", 1)[0].strip()
            if not ref:
                continue

            if ref.startswith("/"):
                candidate_rel = PurePosixPath(ref.lstrip("/"))
            else:
                candidate_rel = md_dir / PurePosixPath(ref)

            candidate_rel = _normalize_repo_relative(candidate_rel)
            if candidate_rel is None:
                continue

            if PurePosixPath(candidate_rel).suffix.lower() not in image_exts:
                continue

            candidate_abs = (repo_root / Path(str(candidate_rel))).resolve()
            if not candidate_abs.exists():
                warnings.append(f"scan: missing asset referenced by {md_rel}: {candidate_rel}")
                continue
            if not candidate_abs.is_file():
                continue
            assets.add(candidate_abs)

    return assets, sorted(set(warnings))


class GitIgnore:
    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root

    def ignored(self, repo_relative_posix_paths: list[str]) -> set[str]:
        if not repo_relative_posix_paths:
            return set()
        try:
            proc = subprocess.run(
                ["git", "-C", str(self._repo_root), "check-ignore", "-z", "--stdin"],
                input="".join(f"{p}\n" for p in repo_relative_posix_paths),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except Exception:
            return set()
        if proc.returncode not in (0, 1):
            return set()
        out = proc.stdout
        if not out:
            return set()
        parts = [p for p in out.split("\0") if p]
        return set(parts)


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # 1) Prefer PyYAML when available (MkDocs environments usually have it).
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    def parse_scalar(s: str):
        v = s.strip()
        if not v:
            return ""
        if v in ("null", "Null", "NULL", "~"):
            return None
        if v in ("true", "True", "TRUE"):
            return True
        if v in ("false", "False", "FALSE"):
            return False
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        # int
        if re.fullmatch(r"-?\d+", v):
            try:
                return int(v)
            except Exception:
                return v
        return v

    def parse_inline_list(s: str) -> list:
        inner = s.strip()[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        return [parse_scalar(p) for p in parts if p]

    def parse_simple_yaml(text: str) -> dict:
        # Very small YAML subset (sufficient for docview.yml):
        # - mappings: key: value
        # - nested mappings/lists via indentation
        # - sequences: key:\n  - item
        # - inline lists: key: [a, b]
        #
        # Not supported: anchors, multi-line scalars, complex types, list-of-maps, quoted escaping, etc.
        root: dict = {}
        stack: list[tuple[int, object]] = [(0, root)]

        lines = text.splitlines()
        i = 0
        while i < len(lines):
            raw_line = lines[i].rstrip()
            i += 1
            if not raw_line.strip():
                continue
            if raw_line.lstrip().startswith("#"):
                continue

            indent = len(raw_line) - len(raw_line.lstrip(" "))
            content = raw_line.lstrip(" ")

            while len(stack) > 1 and indent < stack[-1][0]:
                stack.pop()
            container = stack[-1][1]

            if content.startswith("- "):
                if not isinstance(container, list):
                    continue
                item = content[2:].strip()
                value = parse_inline_list(item) if item.startswith("[") and item.endswith("]") else parse_scalar(item)
                container.append(value)
                continue

            if ":" not in content:
                continue
            if isinstance(container, list):
                continue

            key, rest = content.split(":", 1)
            key = key.strip()
            rest = rest.strip()

            if rest == "":
                # Look ahead to decide whether this is a dict or a list.
                j = i
                next_line = ""
                next_indent = 0
                while j < len(lines):
                    cand = lines[j].rstrip()
                    if not cand.strip() or cand.lstrip().startswith("#"):
                        j += 1
                        continue
                    next_indent = len(cand) - len(cand.lstrip(" "))
                    next_line = cand.lstrip(" ")
                    break

                if next_line.startswith("- ") and next_indent > indent:
                    new_container: object = []
                    container[key] = new_container
                    stack.append((indent + 2, new_container))
                else:
                    new_container = {}
                    container[key] = new_container
                    stack.append((indent + 2, new_container))
                continue

            if rest.startswith("[") and rest.endswith("]"):
                container[key] = parse_inline_list(rest)
                continue

            container[key] = parse_scalar(rest)

        return root

    # 2) Try parsing a small YAML subset (enough for this manifest).
    try:
        data = parse_simple_yaml(raw)
        if isinstance(data, dict) and data:
            return data
    except Exception:
        pass

    # 3) Fallback: accept JSON subset (valid YAML 1.2).
    raw_strip = raw.lstrip()
    if raw_strip.startswith("{"):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _get_dict(d: dict, key: str) -> dict:
    v = d.get(key, {})
    return v if isinstance(v, dict) else {}


def _get_list(d: dict, key: str) -> list[str]:
    v = d.get(key)
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    return [str(v)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan repo roots for Markdown files and image assets referenced by those Markdown files."
    )
    parser.add_argument("--repo-root", required=True, help="Repo root (absolute path recommended).")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to YAML manifest. Default: <work_dir>/docview.yml (next to this script), if present.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Absolute path to exclude from scanning/linking (may be specified multiple times).",
    )
    parser.add_argument(
        "--respect-gitignore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Respect gitignore when discovering Markdown sources under targets (default: true). "
            "Referenced assets are still included even if gitignored."
        ),
    )
    parser.add_argument(
        "--print0",
        action="store_true",
        help="Print null-delimited repo-relative file paths (default: newline-delimited).",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Optional repo-relative paths or glob patterns. If provided, overrides scan.include_globs and is treated as force-scan intent.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    exclude_abs = {Path(p).resolve() for p in args.exclude if p}

    # Manifest (required)
    default_manifest = Path(__file__).resolve().parent / "docview.yml"
    manifest_path = Path(args.manifest).resolve() if args.manifest else default_manifest
    if not manifest_path.exists():
        print(f"scan: missing manifest: {manifest_path}", file=sys.stderr)
        return 2
    manifest = _load_manifest(manifest_path)
    scan_cfg = _get_dict(manifest, "scan")
    assets_cfg = _get_dict(manifest, "assets")

    include_globs = _get_list(scan_cfg, "include_globs")
    exclude_globs = _get_list(scan_cfg, "exclude_globs")
    force_globs = _get_list(scan_cfg, "force_globs")

    # Prevent self-indexing loops by excluding DocView/MDView workdirs (including this one).
    auto_exclude_generated = bool(scan_cfg.get("auto_exclude_generated", True))
    work_dir = Path(__file__).resolve().parent
    exclude_abs.add(work_dir.resolve())
    if auto_exclude_generated and repo_root.is_dir():
        for d in discover_generated_docs_view_workdirs(repo_root):
            exclude_abs.add(d)

    def looks_like_glob(s: str) -> bool:
        return any(ch in s for ch in ("*", "?", "[", "]")) or s.startswith(":(")

    def coerce_arg_to_glob(arg: str) -> str:
        a = str(arg).strip().replace("\\", "/")
        if a in (".", "./"):
            # Handled by the caller (expands to multiple patterns).
            return "."
        if a.startswith("./"):
            a = a[2:]
        # Repo-relative only. If the user passes an absolute path, treat it as repo-relative.
        while a.startswith("/"):
            a = a[1:]
        a = a.rstrip("/")
        if a in ("..",) or a.startswith("../"):
            return ""
        if not a:
            return ""
        if looks_like_glob(a):
            return a
        p = (repo_root / a).resolve()
        if p.is_dir():
            return f"{a}/**/*.md"
        return a

    explicit_inputs: list[str] = []
    for t in args.targets:
        raw = str(t).strip()
        if not raw:
            continue
        if raw in (".", "./"):
            explicit_inputs.extend(["*.md", "**/*.md"])
            continue
        g = coerce_arg_to_glob(raw)
        if g and g != ".":
            explicit_inputs.append(g)

    # If the user provided explicit scan inputs, treat that as strong intent:
    # - override include_globs
    # - also force-scan those globs even if gitignored
    include_globs_effective = explicit_inputs or include_globs
    force_globs_effective = sorted(set(force_globs + explicit_inputs))

    if not include_globs_effective:
        print(
            "scan: no scan patterns provided (set scan.include_globs in docview.yml or pass explicit patterns)",
            file=sys.stderr,
        )
        return 2

    # Apply extra absolute excludes from CLI only (manifest exclusions are globs handled below).

    include_hidden = bool(scan_cfg.get("include_hidden", False))
    respect_gitignore = bool(scan_cfg.get("respect_gitignore", args.respect_gitignore))

    def to_pathspec_glob(p: str) -> str:
        ps = p.strip()
        if not ps:
            return ""
        if ps.startswith(":("):
            return ps
        return f":(glob){ps}"

    def to_pathspec_exclude(p: str) -> str:
        ps = p.strip()
        if not ps:
            return ""
        if ps.startswith(":("):
            # If user supplies raw pathspec magic, accept it as-is.
            return ps
        return f":(glob,exclude){ps}"

    include_specs = [to_pathspec_glob(g) for g in include_globs_effective]
    exclude_specs = [to_pathspec_exclude(g) for g in exclude_globs]
    include_specs = [s for s in include_specs if s]
    exclude_specs = [s for s in exclude_specs if s]
    targets_specs = include_specs + exclude_specs
    forced_specs = [to_pathspec_glob(g) for g in force_globs_effective]
    forced_specs = [s for s in forced_specs if s] + exclude_specs

    # Markdown discovery: by default, respect gitignore by using git's file listing.
    if respect_gitignore and _is_git_repo(repo_root):
        markdown_files = iter_markdown_files_git(
            repo_root,
            targets_specs,
            forced_targets=forced_specs,
            exclude_abs=exclude_abs,
            include_hidden=include_hidden,
        )
    else:
        # Filesystem fallback: apply include/exclude globs under repo_root.
        markdown_files = []
        for pattern in include_globs_effective:
            if pattern.startswith(":("):
                continue
            for p in repo_root.glob(pattern):
                pp = p.resolve()
                if not pp.is_file():
                    continue
                if pp.suffix.lower() != ".md":
                    continue
                if _under_excludes(pp, exclude_abs):
                    continue
                try:
                    rel = pp.relative_to(repo_root).as_posix()
                except Exception:
                    continue
                if not include_hidden and _is_hidden_rel_posix(rel):
                    continue
                markdown_files.append(pp)
        # Exclude globs
        if exclude_globs:
            filtered = []
            for p in sorted(set(markdown_files)):
                try:
                    rel = p.relative_to(repo_root).as_posix()
                except Exception:
                    continue
                if any(PurePosixPath(rel).match(g) for g in exclude_globs if not g.startswith(":(")):
                    continue
                filtered.append(p)
            markdown_files = filtered

    # Defensive dedupe: include_globs may overlap.
    markdown_files = sorted(set(markdown_files))

    include_images = assets_cfg.get("include_images", True)
    image_exts: set[str] = set(DEFAULT_IMAGE_EXTS)
    ext_list = assets_cfg.get("image_extensions")
    if isinstance(ext_list, list) and ext_list:
        image_exts = {str(x).lower() for x in ext_list if str(x).strip()}

    assets: set[Path] = set()
    warnings: list[str] = []
    if include_images:
        assets, warnings = iter_referenced_image_assets(repo_root, markdown_files, image_exts=image_exts)

    all_files: set[Path] = set(markdown_files) | assets
    rels: list[str] = []
    for p in sorted(all_files):
        try:
            rel = p.relative_to(repo_root).as_posix()
        except Exception:
            continue
        rels.append(rel)

    sep = "\0" if args.print0 else "\n"
    sys.stdout.write(sep.join(rels))
    if args.print0 and rels:
        sys.stdout.write("\0")

    for w in warnings:
        print(w, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
    )


def _service_readme(
    *,
    service_name: str,
    site_name: str,
    dev_addr: str | None,
    dirs_env_var: str,
    repo_root_env_var: str,
    staging_dir: str,
    default_targets: list[str],
) -> str:
    default_list = "\n".join([f"- `{t}`" for t in default_targets]) if default_targets else "- (none)"
    bind_note = (
        f"Default bind address/port is `{dev_addr}`."
        if dev_addr
        else "Bind address/port uses MkDocs defaults unless overridden (typically `127.0.0.1:8000`)."
    )
    return textwrap.dedent(
        f"""\
        # {site_name}

        This is an auxiliary MkDocs configuration that serves Markdown files from selected repo paths while preserving directory structure.

        ## Select what to index

        Override scan inputs (repo-relative, space-separated; paths or globs):

        - Pass explicit paths:
          - `pixi run bash {service_name}/refresh-docs-tree.sh <scan_path_or_glob_1> <scan_path_or_glob_2>`
        - Or set `{dirs_env_var}`:
          - `{dirs_env_var}="<scan_path_or_glob_1> <scan_path_or_glob_2>" pixi run bash {service_name}/refresh-docs-tree.sh`
        - Or edit `{service_name}/docview.yml` to change the default scan include/exclude patterns.

        If this work dir is **not** inside the git repo, also set the repo root:

        - `{repo_root_env_var}="/abs/path/to/repo" pixi run bash {service_name}/refresh-docs-tree.sh <scan_path_or_glob_1> <scan_path_or_glob_2>`

        Defaults (when no args / env override):

        {default_list}

        ## Run

        From the repo root:

        - `pixi run bash {service_name}/refresh-docs-tree.sh`
        - `pixi run mkdocs serve -f {service_name}/mkdocs.yml`

        Notes:
        - Staged files live under `{service_name}/{staging_dir}/` and are generated as symlinks by `{service_name}/refresh-docs-tree.sh`.
        - `{service_name}/scan-files-to-stage.py` discovers Markdown files and image assets referenced by Markdown.
        - `{service_name}/docview.yml` stores the default scan patterns and gitignore policy (`scan.respect_gitignore`, `scan.auto_exclude_generated`, `scan.include_globs`, `scan.force_globs`, `scan.exclude_globs`).
        - `{service_name}/mkdocs.yml` is generated only if missing (existing configs are preserved).
        - {bind_note}
        - Mermaid and KaTeX are enabled by default via `pymdown-extensions` + `extra_javascript`/`extra_css` in the generated MkDocs config.
        - Mermaid code fences are supported via `pymdownx.superfences` with a `mermaid` custom fence.
        - Math is supported via `pymdownx.arithmatex` (generic mode) + KaTeX auto-render.
        - Search is enabled via the built-in MkDocs `search` plugin.
        """
    )


def _mermaid_init_js() -> str:
    # Works with MkDocs Material (instant navigation) by using document$ when present.
    return textwrap.dedent(
        """\
        /* global mermaid, document$ */

        function docviewRenderMermaid() {
          if (!window.mermaid) return;

          try {
            window.mermaid.initialize({ startOnLoad: false });
          } catch (_) {
            // ignore
          }

          try {
            const maybePromise = window.mermaid.run({ querySelector: ".mermaid" });
            if (maybePromise && typeof maybePromise.catch === "function") {
              maybePromise.catch(() => undefined);
            }
          } catch (_) {
            // ignore
          }
        }

        if (window.document$ && typeof window.document$.subscribe === "function") {
          window.document$.subscribe(docviewRenderMermaid);
        } else {
          document.addEventListener("DOMContentLoaded", docviewRenderMermaid);
        }
        """
    )


def _katex_init_js() -> str:
    # Uses KaTeX auto-render. Requires:
    # - katex.min.js
    # - contrib/auto-render.min.js
    return textwrap.dedent(
        """\
        /* global renderMathInElement */

        function docviewRenderKatex() {
          if (typeof window.renderMathInElement !== "function") return;
          try {
            window.renderMathInElement(document.body, {
              delimiters: [
                { left: "$$", right: "$$", display: true },
                { left: "\\\\[", right: "\\\\]", display: true },
                { left: "$", right: "$", display: false },
                { left: "\\\\(", right: "\\\\)", display: false },
              ],
              throwOnError: false,
            });
          } catch (_) {
            // ignore
          }
        }

        if (window.document$ && typeof window.document$.subscribe === "function") {
          window.document$.subscribe(docviewRenderKatex);
        } else {
          document.addEventListener("DOMContentLoaded", docviewRenderKatex);
        }
        """
    )


def _refresh_script(
    *,
    service_name: str,
    site_name: str,
    dev_addr: str | None,
    dirs_env_var: str,
    repo_root_env_var: str,
    staging_dir: str,
) -> str:
    dev_addr_yaml = f"dev_addr: {dev_addr}\n\n" if dev_addr else ""
    return textwrap.dedent(
        r"""\
#!/usr/bin/env bash
set -euo pipefail

# Build a staged tree made of symlinks to repo Markdown files (plus referenced image assets).
#
# Selection:
# - If arguments are provided, those paths/globs (relative to repo root) are indexed.
# - Otherwise, patterns from `docview.yml` are used.
# - You can also override defaults with `{dirs_env_var}` (space-separated list).
#
# Discovery:
# - By default, Markdown discovery respects .gitignore (configurable via `docview.yml`).
# - Referenced image assets from selected Markdown are staged even if gitignored.

work_dir="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"

repo_root="${{{repo_root_env_var}:-}}"
if [[ -z "${{repo_root}}" && -f "${{work_dir}}/repo-root.txt" ]]; then
  repo_root="$(cat -- "${{work_dir}}/repo-root.txt" 2>/dev/null || true)"
fi
if [[ -z "${{repo_root}}" ]]; then
  repo_root="$(git -C "${{work_dir}}" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "${{repo_root}}" ]]; then
  repo_root="$(cd -- "${{work_dir}}/.." && pwd)"
else
  repo_root="$(cd -- "${{repo_root}}" && pwd)"
fi

mkdocs_cfg="${{work_dir}}/mkdocs.yml"
site_dir="${{work_dir}}/site"
scan_py="${{work_dir}}/scan-files-to-stage.py"
manifest_yml="${{work_dir}}/docview.yml"
staging_dir="{staging_dir}"
out_dir="${{work_dir}}/${{staging_dir}}"

rm -rf -- "${{out_dir}}"
mkdir -p -- "${{out_dir}}"
rm -rf -- "${{site_dir}}"

if [[ ! -f "${{manifest_yml}}" ]]; then
  echo "{service_name}: missing manifest: ${{manifest_yml}}" >&2
  echo "{service_name}: run the scaffolder to create it (or re-run with --force)" >&2
  exit 1
fi

positional_targets=()
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'USAGE'
Usage: bash {service_name}/refresh-docs-tree.sh [paths_or_globs...]

Builds a staged symlink tree of Markdown files (plus referenced image assets) from repo paths.

Arguments:
  paths_or_globs...   Optional list of repo-root-relative paths or glob patterns to index. When omitted, `docview.yml` is used.
USAGE
      exit 0
      ;;
    *)
      positional_targets+=("$arg")
      ;;
  esac
done

targets=()
if [[ "${{#positional_targets[@]}}" -gt 0 ]]; then
  targets=("${{positional_targets[@]}}")
elif [[ -n "${{{dirs_env_var}:-}}" ]]; then
  # shellcheck disable=SC2206
  targets=(${{{dirs_env_var}}})
else
  # Let the manifest decide defaults.
  targets=()
fi

 # Note: targets may be globs; the scanner handles resolution and ignore rules.

# Link all required files (Markdown + referenced image assets).
#
# The scanner prints repo-relative paths. We convert them back to absolute paths under repo_root.
if [[ -f "${{scan_py}}" ]]; then
  run_in_repo_root=false
  py_cmd=()
  if command -v pixi >/dev/null 2>&1; then
    # Prefer running inside the project's Pixi environment.
    # Pixi resolves the project manifest from the current working directory.
    if (cd -- "${{repo_root}}" && pixi run python -c 'import sys' >/dev/null 2>&1); then
      py_cmd=(pixi run python)
      run_in_repo_root=true
    fi
  fi

  if [[ "${{#py_cmd[@]}}" -eq 0 ]]; then
    py_cmd=(python3)
    if ! command -v "${{py_cmd[0]}}" >/dev/null 2>&1; then
      py_cmd=(python)
    fi
  fi

  scan_args=("${{py_cmd[@]}}" "${{scan_py}}" --repo-root "${{repo_root}}" --exclude "${{work_dir}}" --print0 --manifest "${{manifest_yml}}")
  scan_args+=(-- "${{targets[@]}}")

  if [[ "${{run_in_repo_root}}" == true ]]; then
    (cd -- "${{repo_root}}" && "${{scan_args[@]}}") \
    | while IFS= read -r -d '' rel; do
        src="${{repo_root}}/${{rel}}"
        if [[ ! -e "${{src}}" ]]; then
          echo "{service_name}: missing staged file: ${{rel}}" >&2
          continue
        fi
        dest="${{out_dir}}/${{rel}}"
        mkdir -p -- "$(dirname -- "${{dest}}")"
        ln -sf -- "${{src}}" "${{dest}}"
      done
  else
    "${{scan_args[@]}}" \
    | while IFS= read -r -d '' rel; do
        src="${{repo_root}}/${{rel}}"
        if [[ ! -e "${{src}}" ]]; then
          echo "{service_name}: missing staged file: ${{rel}}" >&2
          continue
        fi
        dest="${{out_dir}}/${{rel}}"
        mkdir -p -- "$(dirname -- "${{dest}}")"
        ln -sf -- "${{src}}" "${{dest}}"
      done
  fi
else
  echo "{service_name}: missing scanner script: ${{scan_py}}" >&2
  exit 1
fi

# Provide a stable MkDocs homepage.
if [[ -f "${{repo_root}}/README.md" ]]; then
  ln -sf -- "${{repo_root}}/README.md" "${{out_dir}}/index.md"
fi

# Stage local JS helpers required by mkdocs.yml (Mermaid + KaTeX init).
#
# MkDocs expects extra_javascript paths to exist under docs_dir.
if [[ -d "${{work_dir}}/javascripts" ]]; then
  mkdir -p -- "${{out_dir}}/javascripts"
  if [[ -f "${{work_dir}}/javascripts/mermaid-init.js" ]]; then
    ln -sf -- "${{work_dir}}/javascripts/mermaid-init.js" "${{out_dir}}/javascripts/mermaid-init.js"
  fi
  if [[ -f "${{work_dir}}/javascripts/katex-init.js" ]]; then
    ln -sf -- "${{work_dir}}/javascripts/katex-init.js" "${{out_dir}}/javascripts/katex-init.js"
  fi
fi

# Verification: fail if any broken symlinks were created.
broken_links="$(find "${{out_dir}}" -xtype l -print || true)"
if [[ -n "${{broken_links}}" ]]; then
  echo "{service_name}: broken symlinks detected under ${{out_dir}}:" >&2
  echo "${{broken_links}}" >&2
  exit 1
fi

# Generate MkDocs config (kept out of git).
if [[ ! -f "${{mkdocs_cfg}}" ]]; then
cat > "${{mkdocs_cfg}}" <<'YAML'
site_name: {site_name_yaml}

docs_dir: {staging_dir}
site_dir: site
{dev_addr_yaml}theme:
  name: material
  features:
    - search.suggest
    - search.highlight
    - search.share

plugins:
  - search

markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format

extra_css:
  - https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css

extra_javascript:
  - https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js
  - https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js
  - javascripts/katex-init.js
  - https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js
  - javascripts/mermaid-init.js

use_directory_urls: true
YAML
fi

echo "{service_name}: linked staged files into ${{out_dir}}"
""".format(  # noqa: UP032
            service_name=service_name,
            dev_addr_yaml=dev_addr_yaml,
            dirs_env_var=dirs_env_var,
            repo_root_env_var=repo_root_env_var,
            staging_dir=staging_dir,
            site_name_yaml=_yaml_single_quote(site_name),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold an MkDocs docs viewer service for arbitrary Markdown roots."
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root to index from (default: auto-detect via git; fallback: current directory).",
    )
    work_dir_group = parser.add_mutually_exclusive_group(required=True)
    work_dir_group.add_argument(
        "--work-dir",
        help=(
            "Work directory to create/reuse (repo-relative unless absolute), "
            "e.g. docview-reports or tmp/docview-issues"
        ),
    )
    work_dir_group.add_argument(
        "--service-dir",
        help="Deprecated alias for --work-dir (kept for compatibility).",
    )
    parser.add_argument(
        "--site-name",
        required=True,
        help='MkDocs site title, e.g. "Reports View"',
    )
    dev_group = parser.add_mutually_exclusive_group(required=False)
    dev_group.add_argument(
        "--dev-addr",
        default=None,
        help='MkDocs dev_addr, e.g. "127.0.0.1:8000" or "0.0.0.0:<port>" (default: MkDocs built-in default if omitted).',
    )
    dev_group.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port shorthand (host decided by --host/--public; default host is 127.0.0.1).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help='Host for --port shorthand, e.g. "127.0.0.1" or "0.0.0.0". Ignored if --dev-addr is set.',
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Bind publicly on 0.0.0.0. If neither --dev-addr nor --port is set, uses 0.0.0.0:8000.",
    )
    parser.add_argument(
        "--staging-dir",
        default="_staged",
        help=(
            "Subdirectory under the work dir used to hold staged files (symlinks/copies) for MkDocs. "
            "Default: _staged"
        ),
    )
    parser.add_argument(
        "--dirs-env-var",
        default=None,
        help="Override env var name used to specify roots (default: derived from service dir).",
    )
    parser.add_argument(
        "--default-target",
        action="append",
        default=[],
        help=(
            "Repo-relative path included by default when no args/env override is provided. "
            "May be specified multiple times."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files inside the service directory.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _detect_repo_root(Path("."))

    work_dir_raw = args.work_dir or args.service_dir
    if not work_dir_raw:
        raise ValueError("Missing --work-dir/--service-dir.")

    work_dir = Path(work_dir_raw)
    service_root = (work_dir if work_dir.is_absolute() else (repo_root / work_dir)).resolve()
    service_name = service_root.name
    dirs_env_var = args.dirs_env_var or _default_dirs_env_var(service_name)
    repo_root_env_var = _default_repo_root_env_var(dirs_env_var)
    dev_addr = args.dev_addr
    if dev_addr is None:
        if args.port is not None:
            host = args.host or ("0.0.0.0" if args.public else "127.0.0.1")
            dev_addr = f"{host}:{args.port}"
        elif args.public:
            host = args.host or "0.0.0.0"
            dev_addr = f"{host}:8000"

    staging_dir = args.staging_dir.strip().rstrip("/").rstrip("\\")
    if not staging_dir:
        raise ValueError("--staging-dir must be a non-empty relative path.")
    if Path(staging_dir).is_absolute() or ".." in Path(staging_dir).parts:
        raise ValueError("--staging-dir must be a relative path that does not contain '..'.")

    default_targets = list(args.default_target)
    if not default_targets:
        # Default to scanning the whole repo while respecting .gitignore.
        default_targets = ["."]

    service_root.mkdir(parents=True, exist_ok=True)

    _write_file(
        service_root / ".gitignore",
        _service_gitignore(service_name, staging_dir),
        force=args.force,
    )
    _write_file(
        service_root / "README.md",
        _service_readme(
            service_name=service_name,
            site_name=args.site_name,
            dev_addr=dev_addr,
            dirs_env_var=dirs_env_var,
            repo_root_env_var=repo_root_env_var,
            staging_dir=staging_dir,
            default_targets=default_targets,
        ),
        force=args.force,
    )
    _write_file(
        service_root / "refresh-docs-tree.sh",
        _refresh_script(
            service_name=service_name,
            site_name=args.site_name,
            dev_addr=dev_addr,
            dirs_env_var=dirs_env_var,
            repo_root_env_var=repo_root_env_var,
            staging_dir=staging_dir,
        ),
        force=args.force,
    )
    _write_file(
        service_root / "scan-files-to-stage.py",
        _scan_files_script(),
        force=args.force,
    )
    _write_file(
        service_root / "javascripts" / "mermaid-init.js",
        _mermaid_init_js(),
        force=args.force,
    )
    _write_file(
        service_root / "javascripts" / "katex-init.js",
        _katex_init_js(),
        force=args.force,
    )
    _write_file(
        service_root / "docview.yml",
        _docview_manifest_yaml(
            site_name=args.site_name,
            dev_addr=dev_addr,
            staging_dir=staging_dir,
            default_targets=default_targets,
        ),
        force=args.force,
    )
    _write_file(service_root / "repo-root.txt", str(repo_root) + "\n", force=args.force)

    print(f"Created docview service: {service_root}")
    print(f"- Roots override env var: {dirs_env_var}")
    print(f"- Repo root override env var: {repo_root_env_var}")
    print(f"- Serve: pixi run bash {service_root}/refresh-docs-tree.sh && pixi run mkdocs serve -f {service_root}/mkdocs.yml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
