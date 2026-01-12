#!/bin/bash
set -e

# init-speckit-here.sh
# Initializes speckit for the current directory using available AI tools.

# Argument parsing
USAGE="Usage: $0 [--agent {codex|gemini|claude|copilot|all}]"
TARGET_AGENT="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)
            if [[ -z "$2" ]]; then echo "Error: Missing argument for --agent"; echo "$USAGE"; exit 1; fi
            TARGET_AGENT="$2"
            shift 2
            ;;
        -h|--help)
            echo "$USAGE"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "$USAGE"
            exit 1
            ;;
    esac
done

# Check for gh (GitHub CLI) and export token if logged in
if command -v gh >/dev/null 2>&1; then
    # Try to get the token. 
    # gh auth token returns 0 on success, non-zero if not authenticated.
    if _gh_token="$(gh auth token 2>/dev/null)"; then
        export GITHUB_TOKEN="$_gh_token"
        echo "Info: Exported GITHUB_TOKEN from gh CLI."
    else
        echo "Info: gh CLI installed but not authenticated or token unavailable. Skipping GITHUB_TOKEN export."
    fi
fi

# Detect AI tools
FOUND_AGENTS=()

check_agent_installed() {
    local agent=$1
    case $agent in
        codex) command -v codex >/dev/null 2>&1 ;;
        claude) command -v claude >/dev/null 2>&1 ;;
        gemini) command -v gemini >/dev/null 2>&1 ;;
        copilot) command -v gh >/dev/null 2>&1 && gh extension list 2>/dev/null | grep -q "copilot" ;;
        *) return 1 ;;
    esac
}

if [[ "$TARGET_AGENT" == "all" ]]; then
    check_agent_installed codex && FOUND_AGENTS+=("codex")
    check_agent_installed claude && FOUND_AGENTS+=("claude")
    check_agent_installed gemini && FOUND_AGENTS+=("gemini")
    check_agent_installed copilot && FOUND_AGENTS+=("copilot")
else
    # Validate requested agent
    if [[ ! "$TARGET_AGENT" =~ ^(codex|claude|gemini|copilot)$ ]]; then
        echo "Error: Invalid agent '$TARGET_AGENT'. Supported: codex, claude, gemini, copilot."
        exit 1
    fi

    if check_agent_installed "$TARGET_AGENT"; then
        FOUND_AGENTS+=("$TARGET_AGENT")
    else
        echo "Warning: Agent '$TARGET_AGENT' requested but not found/installed."
        FOUND_AGENTS+=("$TARGET_AGENT")
    fi
fi

if [[ ${#FOUND_AGENTS[@]} -gt 0 ]]; then
    for agent in "${FOUND_AGENTS[@]}"; do
        echo "Found AI tool: $agent"
        echo "Initializing specify for $agent (sh scripts)..."
        # Use --force to skip confirmation for non-empty directory
        # Use --script sh to generate POSIX shell scripts only
        specify init --here --ai "$agent" --force --script sh
    done
else
    echo "No supported AI tool detected (codex, claude, gemini, or gh copilot)."
    echo "Running specify init interactively..."
    specify init --here --script sh
fi

# Post-initialization: Add related directories to .gitignore
GITIGNORE=".gitignore"

add_to_gitignore() {
    local dir="$1"
    # Check if directory exists
    if [[ -d "$dir" ]]; then
        # Ensure .gitignore exists
        if [[ ! -f "$GITIGNORE" ]]; then
            touch "$GITIGNORE"
        fi

        # Check if already ignored (matching exactly 'dir' or 'dir/')
        if ! grep -Fxq "${dir}" "$GITIGNORE" && ! grep -Fxq "${dir}/" "$GITIGNORE"; then
            echo "${dir}/" >> "$GITIGNORE"
            echo "Added ${dir}/ to $GITIGNORE"
        fi
    fi
}

echo "Checking for created directories to ignore..."
# add_to_gitignore ".specify" # User requested to keep .specify/ in git tracking
add_to_gitignore ".codex"
add_to_gitignore ".claude"
add_to_gitignore ".gemini"
