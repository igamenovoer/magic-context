#!/bin/bash
# Configure Git Submodules: HTTPS Pull, SSH Push
#
# This script configures git submodules to use HTTPS for pull/fetch operations
# and SSH for push operations. This allows pulling without authentication while
# maintaining secure SSH-based authentication for pushes.
#
# Usage:
#   ./configure-submodule-https-ssh.sh [submodule_path]
#   ./configure-submodule-https-ssh.sh --all
#   ./configure-submodule-https-ssh.sh magic-context
#   ./configure-submodule-https-ssh.sh --dry-run magic-context
#
# Features:
# - Configure individual submodules or all submodules at once
# - Automatic detection of GitHub URLs and conversion to appropriate formats
# - Dry-run mode to preview changes without applying them
# - Global URL rewriting configuration for automatic HTTPS->SSH conversion
# - Validation of SSH key availability and GitHub connectivity

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script variables
SCRIPT_NAME="$(basename "$0")"
DRY_RUN=false
VERIFY_MODE=false
CONFIGURE_ALL=false
SKIP_SSH_CHECK=false
SUBMODULE_PATH=""

# Function to print colored output
print_status() {
    local message="$1"
    local status="${2:-info}"
    
    case "$status" in
        "error")   echo -e "${RED}${message}${NC}" ;;
        "success") echo -e "${GREEN}${message}${NC}" ;;
        "warning") echo -e "${YELLOW}${message}${NC}" ;;
        "info")    echo -e "${BLUE}${message}${NC}" ;;
        "header")  echo -e "${PURPLE}${message}${NC}" ;;
        *)         echo "$message" ;;
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS] [SUBMODULE_PATH]

Configure git submodules to use HTTPS for pull and SSH for push operations.

Options:
    --all               Configure all submodules in the repository
    --dry-run          Show what would be done without making changes
    --verify           Verify the current configuration of specified submodule
    --skip-ssh-check   Skip SSH connectivity check
    -h, --help         Show this help message

Arguments:
    SUBMODULE_PATH     Path to the submodule to configure

Examples:
    $SCRIPT_NAME magic-context              # Configure specific submodule
    $SCRIPT_NAME --all                      # Configure all submodules
    $SCRIPT_NAME --dry-run magic-context    # Preview changes without applying
    $SCRIPT_NAME --verify magic-context     # Verify current configuration

EOF
}

# Function to check if we're in a git repository
is_git_repo() {
    git rev-parse --git-dir >/dev/null 2>&1
}

# Function to check if SSH connection to GitHub works
check_ssh_connectivity() {
    print_status "Checking SSH connectivity to GitHub..." "info"
    
    if ssh -T -o ConnectTimeout=10 git@github.com 2>&1 | grep -q "successfully authenticated"; then
        print_status "✓ SSH connection to GitHub successful" "success"
        return 0
    else
        print_status "⚠ SSH connection to GitHub failed" "warning"
        print_status "  Make sure you have SSH keys set up for GitHub" "warning"
        return 1
    fi
}

# Function to convert HTTPS URL to SSH
convert_https_to_ssh() {
    local url="$1"
    echo "$url" | sed 's|https://github\.com/|git@github.com:|'
}

# Function to convert SSH URL to HTTPS
convert_ssh_to_https() {
    local url="$1"
    echo "$url" | sed 's|git@github\.com:|https://github.com/|'
}

# Function to check if URL is a GitHub URL
is_github_url() {
    local url="$1"
    [[ "$url" =~ ^https://github\.com/ ]] || [[ "$url" =~ ^git@github\.com: ]] || [[ "$url" =~ ^ssh://git@github\.com/ ]]
}

# Function to get submodule URL from .gitmodules
get_submodule_url() {
    local submodule_path="$1"
    git config --file .gitmodules "submodule.${submodule_path}.url" 2>/dev/null || echo ""
}

# Function to get all submodules
get_all_submodules() {
    git submodule status | awk '{print $2}' | grep -v '^$'
}

# Function to configure a single submodule
configure_submodule() {
    local submodule_path="$1"
    local dry_run_prefix=""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        dry_run_prefix="[DRY RUN] "
    fi
    
    print_status "\n${dry_run_prefix}Configuring submodule: $submodule_path" "header"
    
    # Check if submodule exists
    if [[ ! -d "$submodule_path" ]]; then
        print_status "✗ Submodule path does not exist: $submodule_path" "error"
        return 1
    fi
    
    # Get current submodule URL
    local current_url
    current_url=$(get_submodule_url "$submodule_path")
    if [[ -z "$current_url" ]]; then
        print_status "✗ Could not get URL for submodule: $submodule_path" "error"
        return 1
    fi
    
    print_status "Current URL: $current_url" "info"
    
    # Check if it's a GitHub URL
    if ! is_github_url "$current_url"; then
        print_status "⚠ Not a GitHub URL, skipping" "warning"
        return 0
    fi
    
    # Determine target URLs
    local https_url ssh_url
    if [[ "$current_url" =~ ^https:// ]]; then
        https_url="$current_url"
        ssh_url=$(convert_https_to_ssh "$current_url")
    else
        ssh_url="$current_url"
        https_url=$(convert_ssh_to_https "$current_url")
    fi
    
    print_status "Target fetch URL (HTTPS): $https_url" "info"
    print_status "Target push URL (SSH): $ssh_url" "info"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Update .gitmodules to use HTTPS
        if ! git config --file .gitmodules "submodule.${submodule_path}.url" "$https_url"; then
            print_status "✗ Failed to update .gitmodules" "error"
            return 1
        fi
        
        # Sync submodule configuration
        if ! git submodule sync "$submodule_path"; then
            print_status "✗ Failed to sync submodule" "error"
            return 1
        fi
        
        # Set push URL to SSH in the submodule
        if ! (cd "$submodule_path" && git remote set-url --push origin "$ssh_url"); then
            print_status "✗ Failed to set push URL" "error"
            return 1
        fi
    fi
    
    print_status "✓ Submodule $submodule_path configured successfully" "success"
    return 0
}

# Function to configure global URL rewriting
configure_url_rewriting() {
    local dry_run_prefix=""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        dry_run_prefix="[DRY RUN] "
    fi
    
    print_status "\n${dry_run_prefix}Configuring global URL rewriting..." "header"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        if ! git config url."git@github.com:".pushInsteadOf "https://github.com/"; then
            print_status "✗ Failed to configure URL rewriting" "error"
            return 1
        fi
    fi
    
    print_status "✓ URL rewriting configured: HTTPS URLs will use SSH for push operations" "success"
    return 0
}

# Function to verify configuration
verify_configuration() {
    local submodule_path="$1"
    
    print_status "\nVerifying configuration for: $submodule_path" "header"
    
    if [[ ! -d "$submodule_path" ]]; then
        print_status "✗ Submodule path does not exist: $submodule_path" "error"
        return 1
    fi
    
    # Check remote URLs
    local remote_output
    if ! remote_output=$(cd "$submodule_path" && git remote -v 2>&1); then
        print_status "✗ Failed to get remote URLs: $remote_output" "error"
        return 1
    fi
    
    print_status "Remote URLs:" "info"
    echo "$remote_output" | while read -r line; do
        [[ -n "$line" ]] && echo "  $line"
    done
    
    # Check for proper configuration
    local has_https_fetch=false
    local has_ssh_push=false
    
    if echo "$remote_output" | grep -q "https://github.com/.*fetch"; then
        has_https_fetch=true
    fi
    
    if echo "$remote_output" | grep -q "git@github.com:.*push"; then
        has_ssh_push=true
    fi
    
    if [[ "$has_https_fetch" == "true" && "$has_ssh_push" == "true" ]]; then
        print_status "✓ Configuration verified: HTTPS fetch, SSH push" "success"
        return 0
    else
        print_status "⚠ Configuration may not be correct" "warning"
        return 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            CONFIGURE_ALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verify)
            VERIFY_MODE=true
            shift
            ;;
        --skip-ssh-check)
            SKIP_SSH_CHECK=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            print_status "Unknown option: $1" "error"
            show_usage
            exit 1
            ;;
        *)
            if [[ -n "$SUBMODULE_PATH" ]]; then
                print_status "Multiple submodule paths specified" "error"
                show_usage
                exit 1
            fi
            SUBMODULE_PATH="$1"
            shift
            ;;
    esac
done

# Validate arguments
if [[ "$CONFIGURE_ALL" == "true" && -n "$SUBMODULE_PATH" ]]; then
    print_status "Cannot specify both --all and a specific submodule" "error"
    exit 1
fi

if [[ "$CONFIGURE_ALL" == "false" && -z "$SUBMODULE_PATH" ]]; then
    print_status "Must specify either a submodule path or --all" "error"
    show_usage
    exit 1
fi

# Check if we're in a git repository
if ! is_git_repo; then
    print_status "✗ Not in a git repository" "error"
    exit 1
fi

print_status "Git Submodule Configuration Tool" "header"
print_status "Configuring submodules for HTTPS pull and SSH push" "info"

# Check SSH connectivity (unless skipped or in dry-run/verify mode)
if [[ "$SKIP_SSH_CHECK" == "false" && "$DRY_RUN" == "false" && "$VERIFY_MODE" == "false" ]]; then
    if ! check_ssh_connectivity; then
        print_status "Consider setting up SSH keys or use --skip-ssh-check" "warning"
    fi
fi

success=true

if [[ "$VERIFY_MODE" == "true" ]]; then
    # Verify mode
    if [[ -n "$SUBMODULE_PATH" ]]; then
        if ! verify_configuration "$SUBMODULE_PATH"; then
            success=false
        fi
    else
        print_status "--verify requires a specific submodule path" "error"
        exit 1
    fi
elif [[ "$CONFIGURE_ALL" == "true" ]]; then
    # Configure all submodules
    mapfile -t submodules < <(get_all_submodules)
    
    if [[ ${#submodules[@]} -eq 0 ]]; then
        print_status "No submodules found" "warning"
        exit 0
    fi
    
    print_status "Found ${#submodules[@]} submodule(s)" "info"
    
    for submodule in "${submodules[@]}"; do
        if ! configure_submodule "$submodule"; then
            success=false
        fi
    done
    
    # Configure global URL rewriting
    if ! configure_url_rewriting; then
        success=false
    fi
else
    # Configure specific submodule
    if ! configure_submodule "$SUBMODULE_PATH"; then
        success=false
    fi
    
    # Configure global URL rewriting
    if ! configure_url_rewriting; then
        success=false
    fi
    
    # Verify configuration unless in dry-run mode
    if [[ "$DRY_RUN" == "false" ]]; then
        verify_configuration "$SUBMODULE_PATH"
    fi
fi

if [[ "$success" == "true" ]]; then
    if [[ "$DRY_RUN" == "false" ]]; then
        print_status "\n✓ Configuration completed successfully!" "success"
        print_status "You can now:" "info"
        print_status "  - Pull/fetch using HTTPS (no authentication required)" "info"
        print_status "  - Push using SSH (secure key-based authentication)" "info"
    else
        print_status "\n✓ Dry run completed successfully!" "success"
    fi
else
    print_status "\n✗ Some operations failed" "error"
    exit 1
fi