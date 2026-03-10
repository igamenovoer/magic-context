#!/usr/bin/env bash

set -euo pipefail

print_usage() {
    cat <<'EOF'
Usage: create_worktree.sh [--repo PATH] [--branch REF] [--path TARGET_PATH] [--link-dir NAME]

Create a clean git worktree from a repository and symlink selected untracked
local-state directories into it.

Options:
  --repo PATH        Repository path. Default: current working tree.
  --branch REF       Branch or ref to base the worktree on. Default: current branch.
  --path PATH        Target path. Default: <repo-root>/.shadow-repo/worktree-<ts>
  --link-dir NAME    Extra directory to symlink if it exists and is untracked.
  -h, --help         Show this help.
EOF
}

resolve_repo_root() {
    local repo_arg="$1"

    if [[ -n "$repo_arg" ]]; then
        git -C "$repo_arg" rev-parse --show-toplevel
    else
        git rev-parse --show-toplevel
    fi
}

resolve_target_path() {
    local repo_root="$1"
    local requested_path="$2"
    local ts=""

    if [[ -z "$requested_path" ]]; then
        ts="$(date -u +%Y%m%d-%H%M%S)"
        printf '%s/.shadow-repo/worktree-%s\n' "$repo_root" "$ts"
        return
    fi

    case "$requested_path" in
        /*)
            printf '%s\n' "$requested_path"
            ;;
        *)
            printf '%s/%s\n' "$repo_root" "$requested_path"
            ;;
    esac
}

is_local_branch() {
    local repo_root="$1"
    local ref_name="$2"

    git -C "$repo_root" show-ref --verify --quiet "refs/heads/$ref_name"
}

branch_checked_out() {
    local repo_root="$1"
    local branch_name="$2"

    git -C "$repo_root" worktree list --porcelain | awk -v want="refs/heads/$branch_name" '
        $1 == "branch" && $2 == want { found = 1 }
        END { exit found ? 0 : 1 }
    '
}

has_tracked_files() {
    local repo_root="$1"
    local rel_path="$2"

    git -C "$repo_root" ls-files -- "$rel_path" | grep -q .
}

is_pixi_project() {
    local repo_root="$1"
    local pyproject_path="$repo_root/pyproject.toml"

    if [[ -f "$repo_root/pixi.toml" || -f "$repo_root/pixi.lock" ]]; then
        return 0
    fi

    if [[ -f "$pyproject_path" ]] && grep -Eq '^\[tool\.pixi([.]|])?' "$pyproject_path"; then
        return 0
    fi

    return 1
}

add_unique_dir() {
    local dir_name="$1"

    if [[ -z "$dir_name" ]]; then
        return
    fi

    if [[ -z "${seen_link_dirs[$dir_name]+x}" ]]; then
        seen_link_dirs["$dir_name"]=1
        link_dirs+=("$dir_name")
    fi
}

symlink_dir_if_safe() {
    local repo_root="$1"
    local worktree_path="$2"
    local rel_path="$3"
    local source_path="$repo_root/$rel_path"
    local target_path="$worktree_path/$rel_path"

    if [[ ! -d "$source_path" && ! -L "$source_path" ]]; then
        skipped_missing_dirs+=("$rel_path")
        return
    fi

    if has_tracked_files "$repo_root" "$rel_path"; then
        skipped_tracked_dirs+=("$rel_path")
        return
    fi

    rm -rf "$target_path"
    ln -s "$source_path" "$target_path"
    linked_dirs+=("$rel_path")
}

repo_arg=""
source_ref=""
requested_path=""
declare -a extra_link_dirs=()

while (($# > 0)); do
    case "$1" in
        --repo)
            [[ $# -ge 2 ]] || { echo "error: --repo requires a value" >&2; exit 2; }
            repo_arg="$2"
            shift 2
            ;;
        --branch|--ref)
            [[ $# -ge 2 ]] || { echo "error: --branch requires a value" >&2; exit 2; }
            source_ref="$2"
            shift 2
            ;;
        --path)
            [[ $# -ge 2 ]] || { echo "error: --path requires a value" >&2; exit 2; }
            requested_path="$2"
            shift 2
            ;;
        --link-dir)
            [[ $# -ge 2 ]] || { echo "error: --link-dir requires a value" >&2; exit 2; }
            extra_link_dirs+=("$2")
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            print_usage >&2
            exit 2
            ;;
    esac
done

repo_root="$(resolve_repo_root "$repo_arg")"

if [[ -z "$source_ref" ]]; then
    source_ref="$(git -C "$repo_root" branch --show-current)"
fi

if [[ -z "$source_ref" ]]; then
    echo "error: no current branch detected; pass --branch <ref> explicitly" >&2
    exit 1
fi

source_commit="$(git -C "$repo_root" rev-parse --verify "$source_ref^{commit}")"
worktree_path="$(resolve_target_path "$repo_root" "$requested_path")"

mkdir -p "$(dirname "$worktree_path")"

checkout_mode="detached"
if is_local_branch "$repo_root" "$source_ref" && ! branch_checked_out "$repo_root" "$source_ref"; then
    git -C "$repo_root" worktree add "$worktree_path" "$source_ref"
    checkout_mode="branch"
else
    git -C "$repo_root" worktree add --detach "$worktree_path" "$source_commit"
fi

declare -A seen_link_dirs=()
declare -a link_dirs=()
declare -a linked_dirs=()
declare -a skipped_tracked_dirs=()
declare -a skipped_missing_dirs=()

for dir_name in \
    .claude \
    .codex \
    .gemini \
    .github \
    .aider \
    .cursor \
    .continue \
    .windsurf \
    .kiro
do
    add_unique_dir "$dir_name"
done

for dir_name in "${extra_link_dirs[@]}"; do
    add_unique_dir "$dir_name"
done

if is_pixi_project "$repo_root" && [[ -e "$repo_root/.pixi" || -L "$repo_root/.pixi" ]]; then
    add_unique_dir ".pixi"
fi

for dir_name in "${link_dirs[@]}"; do
    symlink_dir_if_safe "$repo_root" "$worktree_path" "$dir_name"
done

echo "WORKTREE=$worktree_path"
echo "SOURCE_REF=$source_ref"
echo "CHECKOUT_MODE=$checkout_mode"
echo "COMMIT=$(git -C "$worktree_path" rev-parse HEAD)"
echo "BRANCH=$(git -C "$worktree_path" branch --show-current)"

for dir_name in "${linked_dirs[@]}"; do
    echo "LINKED=$dir_name"
done

for dir_name in "${skipped_tracked_dirs[@]}"; do
    echo "SKIPPED_TRACKED=$dir_name"
done

for dir_name in "${skipped_missing_dirs[@]}"; do
    echo "SKIPPED_MISSING=$dir_name"
done
