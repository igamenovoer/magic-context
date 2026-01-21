#!/bin/bash
# Configure Git Submodules: HTTPS Pull and Push
#
# This script configures git submodules to use HTTPS for both pull and push operations.
# It ensures that no SSH-based authentication is required or enforced for submodules.
#
# Usage:
#   ./configure-submodule-https.sh [submodule_path]
#   ./configure-submodule-https.sh --all
#
# Features:
# - Configure individual submodules or all submodules at once
# - Converts SSH URLs to HTTPS
# - Removes SSH-enforcing configurations

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script variables
SCRIPT_NAME="$(basename "$0")"
CONFIGURE_ALL=false
SUBMODULE_PATH=""

print_status() {
    local message="$1"
    local status="${2:-info}"
    case "$status" in
        "error")   echo -e "${RED}${message}${NC}" ;;
        "success") echo -e "${GREEN}${message}${NC}" ;;
        "info")    echo -e "${BLUE}${message}${NC}" ;;
        *)         echo "$message" ;;
    esac
}

show_usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS] [SUBMODULE_PATH]

Configure git submodules to use HTTPS for both pull and push operations.

Options:
    --all               Configure all submodules in the repository
    -h, --help         Show this help message

Arguments:
    SUBMODULE_PATH     Path to the submodule to configure
EOF
}

get_all_submodules() {
    git submodule status | awk '{print $2}' | grep -v '^$'
}

configure_submodule() {
    local submodule_path="$1"
    print_status "Configuring submodule: $submodule_path" "info"
    
    if [[ ! -d "$submodule_path" ]]; then
        print_status "✗ Submodule path does not exist: $submodule_path" "error"
        return 1
    fi
    
    # Get current URL
    local current_url
    current_url=$(git config --file .gitmodules "submodule.${submodule_path}.url" 2>/dev/null || echo "")
    
    # Convert to HTTPS if needed
    local https_url="$current_url"
    if [[ "$current_url" =~ ^git@github\.com: ]]; then
        https_url=$(echo "$current_url" | sed 's|git@github\.com:|https://github.com/|')
    elif [[ "$current_url" =~ ^ssh://git@github\.com/ ]]; then
         https_url=$(echo "$current_url" | sed 's|ssh://git@github\.com/|https://github.com/|')
    fi
    
    # Update .gitmodules
    git config --file .gitmodules "submodule.${submodule_path}.url" "$https_url"
    
    # Sync and Update
    git submodule sync "$submodule_path"
    
    # Explicitly set push URL to match fetch URL (HTTPS)
    (cd "$submodule_path" && git remote set-url --push origin "$https_url")
    
    print_status "✓ Submodule $submodule_path set to HTTPS" "success"
}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --all) CONFIGURE_ALL=true; shift ;;
        -h|--help) show_usage; exit 0 ;;
        -*) print_status "Unknown option: $1" "error"; show_usage; exit 1 ;;
        *) SUBMODULE_PATH="$1"; shift ;;
    esac
done

if [[ "$CONFIGURE_ALL" == "true" ]]; then
    mapfile -t submodules < <(get_all_submodules)
    if [[ ${#submodules[@]} -eq 0 ]]; then
        print_status "No submodules found." "info"
    else
        for s in "${submodules[@]}"; do configure_submodule "$s"; done
    fi
elif [[ -n "$SUBMODULE_PATH" ]]; then
    configure_submodule "$SUBMODULE_PATH"
else
    # If no args, maybe just clean up global repo config?
    print_status "No submodule specified. Use --all or provide a path." "error"
    exit 1
fi

# Clean up any potential pushInsteadOf global/local config that might force SSH
# We don't want to blindly delete, but we can warn or try to unset if it matches our pattern
if git config --get url."git@github.com:".pushInsteadOf > /dev/null; then
     print_status "Removing url.git@github.com:.pushInsteadOf config..." "info"
     git config --unset url."git@github.com:".pushInsteadOf
fi

print_status "Configuration complete." "success"
