#!/bin/sh

# Simple proxy setup helper.
# Features:
#   --port <port>        : proxy port (default: 7890)
#   --base-url <base>    : proxy base URL/host (default: http://127.0.0.1)
#   --no-check           : skip curl connectivity check
# Behavior:
#   - If no CLI options are given but http_proxy/HTTP_PROXY is set, reuse it as-is.
#   - When constructing a proxy from base/port inside Docker, replace
#     localhost/127.0.0.1 with host.docker.internal.

case "$0" in
  *setup-proxy.sh)
    echo "Warning: setup-proxy.sh is intended to be sourced (e.g. '. setup-proxy.sh'), not executed directly." >&2
    ;;
esac

DEFAULT_PORT=7890
DEFAULT_BASE_URL="http://127.0.0.1"
PORT=""
BASE_URL=""
NO_CHECK=0
BASE_URL_USER_SET=0
PORT_USER_SET=0

usage() {
  cat <<EOF
Usage: setup-proxy.sh [--base-url <url>] [--port <port>] [--no-check]

Options:
  --base-url <url>   Proxy base URL or host (default: http://127.0.0.1).
                     If URL has no scheme, "http://" is prepended.
  --port <port>      Proxy port (default: 7890).
  --no-check         Skip curl connectivity check after setting proxy.
EOF
}

in_docker() {
  if [ -f "/.dockerenv" ]; then
    return 0
  fi
  if [ -r /proc/1/cgroup ] && grep -q "docker" /proc/1/cgroup 2>/dev/null; then
    return 0
  fi
  return 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port)
      if [ "$#" -lt 2 ]; then
        echo "Error: --port requires a value" >&2
        usage
        exit 1
      fi
      PORT="$2"
      PORT_USER_SET=1
      shift 2
      ;;
    --base-url|--base_url)
      if [ "$#" -lt 2 ]; then
        echo "Error: --base-url requires a value" >&2
        usage
        exit 1
      fi
      BASE_URL="$2"
      BASE_URL_USER_SET=1
      shift 2
      ;;
    --no-check)
      NO_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# Case 1: no CLI overrides, but http_proxy/HTTP_PROXY already set -> reuse as-is.
USE_ENV_PROXY=0
ENV_HTTP_PROXY=""
ENV_HTTP_SOURCE=""
if [ -z "$BASE_URL" ] && [ -z "$PORT" ]; then
  if [ -n "${http_proxy:-}" ]; then
    ENV_HTTP_PROXY=$http_proxy
    ENV_HTTP_SOURCE="http_proxy"
    USE_ENV_PROXY=1
  elif [ -n "${HTTP_PROXY:-}" ]; then
    ENV_HTTP_PROXY=$HTTP_PROXY
    ENV_HTTP_SOURCE="HTTP_PROXY"
    USE_ENV_PROXY=1
  fi
fi

if [ "$USE_ENV_PROXY" -eq 1 ]; then
  PROXY="$ENV_HTTP_PROXY"
else
  # Build proxy from base_url/port
  [ -n "$BASE_URL" ] || BASE_URL="$DEFAULT_BASE_URL"
  [ -n "$PORT" ] || PORT="$DEFAULT_PORT"

  # Ensure scheme present; if missing, prepend http://
  case "$BASE_URL" in
    http://*|https://*)
      ;;
    *)
      BASE_URL="http://$BASE_URL"
      ;;
  esac

  # If inside Docker, translate localhost/127.0.0.1 to host.docker.internal
  if in_docker; then
    BASE_URL=$(printf '%s\n' "$BASE_URL" | sed 's#127\.0\.0\.1#host.docker.internal#g; s#localhost#host.docker.internal#g')
  fi

  # Strip trailing slash before appending port
  BASE_STRIPPED=${BASE_URL%/}
  PROXY="${BASE_STRIPPED}:$PORT"
fi

SOURCE_DESC="default"
if [ "$USE_ENV_PROXY" -eq 1 ]; then
  if [ -n "$ENV_HTTP_SOURCE" ]; then
    SOURCE_DESC="env:$ENV_HTTP_SOURCE"
  else
    SOURCE_DESC="env"
  fi
elif [ "$BASE_URL_USER_SET" -eq 1 ] || [ "$PORT_USER_SET" -eq 1 ]; then
  SOURCE_DESC="cli"
else
  SOURCE_DESC="default"
fi

export http_proxy="$PROXY"
export https_proxy="$PROXY"
export HTTP_PROXY="$PROXY"
export HTTPS_PROXY="$PROXY"

echo "Proxy set to: $PROXY (source: $SOURCE_DESC)"

if [ "$NO_CHECK" -eq 0 ]; then
  if command -v curl >/dev/null 2>&1; then
    # Use a simple HTTPS endpoint to verify outbound connectivity via proxy.
    if ! curl --max-time 5 --silent --head https://www.google.com >/dev/null 2>&1; then
      echo "Warning: proxy check via curl failed; proxy may not be running yet." >&2
    fi
  else
    echo "curl not found; skipping proxy connectivity check." >&2
  fi
fi
