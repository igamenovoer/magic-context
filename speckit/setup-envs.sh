#!/bin/bash

set -euo pipefail

usage() {
	cat <<'EOF'
Usage: setup-envs.sh [--proxy <proxy_addr|auto|none>]

Options:
	--proxy    Explicit proxy address, or one of:
						 auto (default) - detect proxy at http://127.0.0.1:7890
						 none           - do not configure any proxy

	Notes:
	- When running inside Docker, detection also probes host.docker.internal:7890
	  (HTTP and SOCKS5) in addition to 127.0.0.1.
	-h, --help Show this help message and exit
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if command -v realpath >/dev/null 2>&1; then
	CODEX_HOME=$(cd -- "$SCRIPT_DIR" && realpath ./.codex 2>/dev/null || printf '%s/.codex\n' "$SCRIPT_DIR")
else
	CODEX_HOME="${SCRIPT_DIR}/.codex"
fi
export CODEX_HOME

proxy_arg="auto"

while (($# > 0)); do
	case "$1" in
		--proxy)
			if (($# == 1)); then
				echo "Error: --proxy requires an argument." >&2
				usage
				exit 1
			fi
			shift
			proxy_arg="$1"
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Error: Unknown option: $1" >&2
			usage
			exit 1
			;;
	esac
	shift
done

clear_proxy() {
	unset HTTP_PROXY
	unset HTTPS_PROXY
	unset http_proxy
	unset https_proxy
}

set_proxy() {
	local proxy="$1"
	export HTTP_PROXY="$proxy"
	export HTTPS_PROXY="$proxy"
	export http_proxy="$proxy"
	export https_proxy="$proxy"
}

detect_local_proxy() {
	local debug="${DEBUG_PROXY:-false}"

	# Determine if we're inside a Docker/containerized environment
	local in_docker="false"
	if [[ -f "/.dockerenv" ]]; then
		in_docker="true"
	elif grep -qiE 'docker|containerd|kubepods' /proc/1/cgroup 2>/dev/null; then
		in_docker="true"
	elif [[ "${container:-}" == "docker" ]]; then
		in_docker="true"
	fi
	[[ "$debug" == "true" ]] && echo "Debug: in_docker=$in_docker" >&2

	if ! command -v curl >/dev/null 2>&1; then
		[[ "$debug" == "true" ]] && echo "Debug: curl not found" >&2
		return 1
	fi

	# Build candidate hosts list
	local candidate_hosts=("127.0.0.1")
	if [[ "$in_docker" == "true" ]]; then
		candidate_hosts+=("host.docker.internal")
	fi

	# Check if any candidate host's port is reachable first (if nc/netcat available)
	if command -v nc >/dev/null 2>&1; then
		local any_open="false"
		for h in "${candidate_hosts[@]}"; do
			if timeout 2 nc -z "$h" 7890 2>/dev/null || nc -z -w2 "$h" 7890 2>/dev/null; then
				[[ "$debug" == "true" ]] && echo "Debug: Port 7890 is listening on $h" >&2
				any_open="true"
				break
			else
				[[ "$debug" == "true" ]] && echo "Debug: Port 7890 not reachable on $h" >&2
			fi
		done
		# Do not early-return if closed; curl checks below may still succeed in some setups
	fi

	# Use fast, reliable test URLs that don't redirect
	local test_urls=(
		"http://www.google.com/generate_204"
		"http://captive.apple.com/hotspot-detect.txt"
		"http://connectivitycheck.gstatic.com/generate_204"
	)

	# Try HTTP proxy protocol for each candidate host
	for h in "${candidate_hosts[@]}"; do
		local http_proxy_candidate="http://$h:7890"
		for url in "${test_urls[@]}"; do
			[[ "$debug" == "true" ]] && echo "Debug: Testing HTTP proxy $http_proxy_candidate with $url" >&2
			if env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
				curl --silent --max-time 8 --output /dev/null --proxy "$http_proxy_candidate" "$url" 2>/dev/null; then
				[[ "$debug" == "true" ]] && echo "Debug: HTTP proxy successful via $http_proxy_candidate" >&2
				printf '%s\n' "$http_proxy_candidate"
				return 0
			fi
		done
	done

	# Try SOCKS5 proxy protocol as fallback (common for local proxies)
	for h in "${candidate_hosts[@]}"; do
		local socks_proxy_candidate="socks5://$h:7890"
		for url in "${test_urls[@]}"; do
			[[ "$debug" == "true" ]] && echo "Debug: Testing SOCKS5 proxy $socks_proxy_candidate with $url" >&2
			if env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
				curl --silent --max-time 8 --output /dev/null --socks5 "$h:7890" "$url" 2>/dev/null; then
				[[ "$debug" == "true" ]] && echo "Debug: SOCKS5 proxy successful via $socks_proxy_candidate" >&2
				printf '%s\n' "$socks_proxy_candidate"
				return 0
			fi
		done
	done

	[[ "$debug" == "true" ]] && echo "Debug: All proxy detection attempts failed" >&2
	return 1
}

proxy_status=""
case "$proxy_arg" in
	auto)
		# Respect existing proxy configuration
		if [[ -n "${HTTP_PROXY:-}" || -n "${HTTPS_PROXY:-}" || -n "${http_proxy:-}" || -n "${https_proxy:-}" ]]; then
			proxy_status="kept (pre-existing)"
		elif proxy_addr=$(detect_local_proxy); then
			set_proxy "$proxy_addr"
			proxy_status="detected and set to $proxy_addr"
		else
			clear_proxy
			proxy_status="not set (no proxy detected)"
		fi
		;;
	none)
		clear_proxy
		proxy_status="disabled (cleared)"
		;;
	*)
		set_proxy "$proxy_arg"
		proxy_status="explicitly set to $proxy_arg"
		;;
esac

# Print all environment variables set by this script
echo ""
echo "========================================="
echo "Environment variables configured:"
echo "========================================="
echo "CODEX_HOME = $CODEX_HOME"
echo ""
echo "Proxy status: $proxy_status"
if [[ -n "${HTTP_PROXY:-}" || -n "${HTTPS_PROXY:-}" || -n "${http_proxy:-}" || -n "${https_proxy:-}" ]]; then
	echo "  HTTP_PROXY  = ${HTTP_PROXY:-<not set>}"
	echo "  HTTPS_PROXY = ${HTTPS_PROXY:-<not set>}"
	echo "  http_proxy  = ${http_proxy:-<not set>}"
	echo "  https_proxy = ${https_proxy:-<not set>}"
else
	echo "  (no proxy variables set)"
fi
echo "========================================="
