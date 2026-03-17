#!/usr/bin/env bash

set -euo pipefail

automation_root_rel=".agent-automation/hacktest"

print_usage() {
    cat <<'EOF'
Usage: create_snapshot_worktree.sh [--repo PATH] [--topic TOPIC_SLUG] [--branch NAME] [--htt-home PATH] [--path WORKTREE_PATH]

Create a throwaway snapshot branch from the current repository state, including
untracked files, without switching the active checkout. Then create a separate
worktree for that branch and create the helper-managed log root, issue-note directory,
and runs root.

Options:
  --repo PATH        Repository path. Default: current working tree.
  --topic SLUG       Topic slug used for branch and log naming.
  --branch NAME      Throwaway branch name. Default: hacktest/<topic-slug>
  --htt-home PATH    HTT home path. Default: <repo-root>/.agent-automation/hacktest/<topic-slug>
  --path PATH        Worktree path. Default: <repo-root>/.agent-automation/hacktest/<topic-slug>/repo
  -h, --help         Show this help.
EOF
}

gitignore_has_hacktest_entry() {
    local gitignore_path="$1"

    [[ -f "$gitignore_path" ]] || return 1
    grep -Eq '^[[:space:]]*/?\.agent-automation/hacktest/?[[:space:]]*$' "$gitignore_path"
}

gitignore_has_hacktest_comment() {
    local gitignore_path="$1"

    [[ -f "$gitignore_path" ]] || return 1
    grep -Eq '^[[:space:]]*#.*\.agent-automation/hacktest/?' "$gitignore_path"
}

ensure_hacktest_gitignored() {
    local repo_root="$1"
    local gitignore_path="$repo_root/.gitignore"
    local last_char=""

    if gitignore_has_hacktest_entry "$gitignore_path"; then
        return 0
    fi

    if gitignore_has_hacktest_comment "$gitignore_path"; then
        return 0
    fi

    if [[ -f "$gitignore_path" && -s "$gitignore_path" ]]; then
        last_char="$(tail -c 1 "$gitignore_path" 2>/dev/null || true)"
        if [[ "$last_char" != $'\n' ]]; then
            printf '\n' >> "$gitignore_path"
        fi
    fi

    printf '.agent-automation/hacktest/\n' >> "$gitignore_path"
}

normalize_topic_slug() {
    printf '%s' "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-{2,}/-/g'
}

validate_topic_slug() {
    [[ "$1" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]
}

resolve_repo_root() {
    local repo_arg="$1"

    if [[ -n "$repo_arg" ]]; then
        git -C "$repo_arg" rev-parse --show-toplevel
    else
        git rev-parse --show-toplevel
    fi
}

resolve_worktree_path() {
    local repo_root="$1"
    local htt_home="$2"
    local requested_path="$3"

    if [[ -z "$requested_path" ]]; then
        printf '%s/repo\n' "$htt_home"
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

resolve_htt_home() {
    local repo_root="$1"
    local topic_slug="$2"
    local requested_htt_home="$3"

    if [[ -z "$requested_htt_home" ]]; then
        printf '%s/%s/%s\n' "$repo_root" "$automation_root_rel" "$topic_slug"
        return
    fi

    case "$requested_htt_home" in
        /*)
            printf '%s\n' "$requested_htt_home"
            ;;
        *)
            printf '%s/%s\n' "$repo_root" "$requested_htt_home"
            ;;
    esac
}

repo_arg=""
topic_slug=""
snapshot_branch=""
requested_htt_home=""
requested_path=""

while (($# > 0)); do
    case "$1" in
        --repo)
            [[ $# -ge 2 ]] || { echo "error: --repo requires a value" >&2; exit 2; }
            repo_arg="$2"
            shift 2
            ;;
        --topic)
            [[ $# -ge 2 ]] || { echo "error: --topic requires a value" >&2; exit 2; }
            topic_slug="$2"
            shift 2
            ;;
        --branch)
            [[ $# -ge 2 ]] || { echo "error: --branch requires a value" >&2; exit 2; }
            snapshot_branch="$2"
            shift 2
            ;;
        --htt-home)
            [[ $# -ge 2 ]] || { echo "error: --htt-home requires a value" >&2; exit 2; }
            requested_htt_home="$2"
            shift 2
            ;;
        --path)
            [[ $# -ge 2 ]] || { echo "error: --path requires a value" >&2; exit 2; }
            requested_path="$2"
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
timestamp="$(date -u +%Y%m%d-%H%M%S)"
hacktest_root_path="$repo_root/$automation_root_rel"
hacktest_root_was_present=0

if [[ -e "$hacktest_root_path" ]]; then
    hacktest_root_was_present=1
fi

if [[ -n "$topic_slug" ]]; then
    topic_slug="$(normalize_topic_slug "$topic_slug")"
    if ! validate_topic_slug "$topic_slug"; then
        echo "error: topic slug must normalize to hyphen-case like foo-bar" >&2
        exit 2
    fi
fi

if [[ -z "$snapshot_branch" ]]; then
    if [[ -z "$topic_slug" ]]; then
        echo "error: pass --topic or --branch" >&2
        exit 2
    fi
    snapshot_branch="hacktest/$topic_slug"
fi

if [[ -z "$topic_slug" ]]; then
    case "$snapshot_branch" in
        hacktest/*)
            topic_slug="${snapshot_branch#hacktest/}"
            ;;
        *)
            echo "error: pass --topic when branch is not of the form hacktest/<topic-slug>" >&2
            exit 2
            ;;
    esac
fi

if ! validate_topic_slug "$topic_slug"; then
    echo "error: topic slug must be hyphen-case and branch must look like hacktest/<topic-slug>" >&2
    exit 2
fi

if git -C "$repo_root" show-ref --verify --quiet "refs/heads/$snapshot_branch"; then
    echo "error: branch already exists: $snapshot_branch" >&2
    exit 1
fi

htt_home="$(resolve_htt_home "$repo_root" "$topic_slug" "$requested_htt_home")"
worktree_path="$(resolve_worktree_path "$repo_root" "$htt_home" "$requested_path")"

if [[ -e "$worktree_path" ]]; then
    echo "error: worktree path already exists: $worktree_path" >&2
    exit 1
fi

parent_commit="$(git -C "$repo_root" rev-parse HEAD)"
tmp_index_dir="$(mktemp -d "${TMPDIR:-/tmp}/hack-through-index.XXXXXX")"
tmp_index="$tmp_index_dir/index"
log_root_path="$htt_home/logs"
issues_dir_path="$log_root_path/issues"
runs_root_path="$htt_home/runs"

cleanup() {
    rm -rf "$tmp_index_dir"
}

trap cleanup EXIT

GIT_INDEX_FILE="$tmp_index" git -C "$repo_root" add -A
tree_id="$(GIT_INDEX_FILE="$tmp_index" git -C "$repo_root" write-tree)"
snapshot_commit="$(
    git -C "$repo_root" commit-tree \
        "$tree_id" \
        -p "$parent_commit" \
        -m "hacktest snapshot $timestamp"
)"

git -C "$repo_root" branch "$snapshot_branch" "$snapshot_commit"
mkdir -p "$(dirname "$worktree_path")"
git -C "$repo_root" worktree add "$worktree_path" "$snapshot_branch" >/dev/null
mkdir -p "$log_root_path"
mkdir -p "$issues_dir_path"
mkdir -p "$runs_root_path"

if [[ "$hacktest_root_was_present" -eq 0 && -d "$hacktest_root_path" ]]; then
    ensure_hacktest_gitignored "$repo_root"
fi

echo "REPO_ROOT=$repo_root"
echo "SOURCE_HEAD=$parent_commit"
echo "TOPIC_SLUG=$topic_slug"
echo "HTT_BRANCH=$snapshot_branch"
echo "SNAPSHOT_COMMIT=$snapshot_commit"
echo "HTT_HOME=$htt_home"
echo "WORKTREE=$worktree_path"
echo "LOG_ROOT=$log_root_path"
echo "ISSUES_DIR=$issues_dir_path"
echo "RUNS_ROOT=$runs_root_path"
