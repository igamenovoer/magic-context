#!/bin/bash

# Timing / retry configuration defaults
KEEP_ALIVE=20
MAX_RETRIES=60
SSH_CONNECT_TIMEOUT=10
CHECK_CONNECT_TIMEOUT=10
PROBE_CONNECT_TIMEOUT=10
RETRY_DELAY=10
STATUS_TIMEOUT=10

# Runtime mode defaults
MODE="foreground"
TMUX_CHILD_MODE=false
BATCH_AUTH_AVAILABLE=false
ATTACH_MODE=false
REMOTE_BIND_ADDR=""

# Function to show usage
usage() {
    echo "Usage: $0 --remote-addr <target> --remote-port <port> --local-port <port> [--remote-bind-addr <localhost|ip-addr>] [--keep-alive <sec>] [--max-retries <count>] [--ssh-connect-timeout <sec>] [--check-connect-timeout <sec>] [--probe-connect-timeout <sec>] [--retry-delay <sec>] [--status-timeout <sec>] [--background|--tmux] [--attach]"
    echo ""
    echo "  --remote-addr   The remote target. Formats allowed:"
    echo "                  1. User & IP:   user@192.168.1.1"
    echo "                  2. Alias:       myserver (from ~/.ssh/config)"
    echo "  --remote-port   The port on the Remote Server to open"
    echo "  --local-port    The port on your Local Machine to expose"
    echo "  --remote-bind-addr Bind address for the remote listener: localhost or a literal IP address"
    echo "                  Omit this option to let the remote SSH server decide (GatewayPorts)"
    echo "  --keep-alive    ServerAliveInterval in seconds (default: 20)"
    echo "  --max-retries   ServerAliveCountMax for ssh keepalive failures (default: 60)"
    echo "  --ssh-connect-timeout SSH ConnectTimeout for the tunnel itself (default: 10)"
    echo "  --check-connect-timeout SSH ConnectTimeout for the preflight batch check (default: 10)"
    echo "  --probe-connect-timeout SSH ConnectTimeout for the remote port probe (default: 10)"
    echo "  --retry-delay   Seconds to wait before reconnecting after ssh exits (default: 10)"
    echo "  --status-timeout Seconds to wait for initial tunnel status (default: 10)"
    echo "  --background    Start a single background ssh process with nohup"
    echo "                  Requires passwordless SSH and no host-key prompt"
    echo "  --tmux          Run the reconnect loop inside tmux session ssh-tunnel-<remote-port>"
    echo "  --attach        Attach or switch into the tmux session after it is created"
    echo ""
    echo "Default mode: run the reconnect loop in the current terminal (foreground)."
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --remote-addr) REMOTE_TARGET="$2"; shift ;;
        --remote-port) REMOTE_PORT="$2"; shift ;;
        --local-port) LOCAL_PORT="$2"; shift ;;
        --remote-bind-addr) REMOTE_BIND_ADDR="$2"; shift ;;
        --keep-alive) KEEP_ALIVE="$2"; shift ;;
        --max-retries) MAX_RETRIES="$2"; shift ;;
        --ssh-connect-timeout) SSH_CONNECT_TIMEOUT="$2"; shift ;;
        --check-connect-timeout) CHECK_CONNECT_TIMEOUT="$2"; shift ;;
        --probe-connect-timeout) PROBE_CONNECT_TIMEOUT="$2"; shift ;;
        --retry-delay) RETRY_DELAY="$2"; shift ;;
        --status-timeout) STATUS_TIMEOUT="$2"; shift ;;
        --background) MODE="background" ;;
        --tmux) MODE="tmux" ;;
        --attach) ATTACH_MODE=true ;;
        --block) MODE="foreground" ;;
        --tmux-child) TMUX_CHILD_MODE=true ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [[ -z "$REMOTE_TARGET" || -z "$REMOTE_PORT" || -z "$LOCAL_PORT" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

if [ "$MODE" != "tmux" ] && [ "$ATTACH_MODE" = true ]; then
    echo "Error: --attach can only be used with --tmux."
    exit 1
fi

is_valid_ipv4_address() {
    local addr="$1"
    local IFS=.
    local octets=()

    [[ "$addr" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1

    read -r -a octets <<<"$addr"
    [ "${#octets[@]}" -eq 4 ] || return 1

    local octet
    for octet in "${octets[@]}"; do
        [ "$octet" -ge 0 ] && [ "$octet" -le 255 ] || return 1
    done
}

validate_remote_bind_addr() {
    local bind_addr="$1"

    if [[ -z "$bind_addr" || "$bind_addr" == "localhost" ]]; then
        return 0
    fi

    if is_valid_ipv4_address "$bind_addr"; then
        return 0
    fi

    if [[ "$bind_addr" =~ ^\[[0-9A-Fa-f:.]+\]$ ]]; then
        return 0
    fi

    if [[ "$bind_addr" == *:* && "$bind_addr" =~ ^[0-9A-Fa-f:.]+$ ]]; then
        return 0
    fi

    return 1
}

format_remote_bind_addr_for_ssh() {
    local bind_addr="$1"

    if [[ "$bind_addr" == \[*\] || "$bind_addr" != *:* ]]; then
        printf '%s' "$bind_addr"
    else
        printf '[%s]' "$bind_addr"
    fi
}

if ! validate_remote_bind_addr "$REMOTE_BIND_ADDR"; then
    echo "Error: --remote-bind-addr must be localhost or a literal IP address."
    exit 1
fi

# Unique signature to identify this tunnel process in 'pgrep'
REMOTE_FORWARD_TARGET="127.0.0.1:$LOCAL_PORT"
REMOTE_BIND_DISPLAY="${REMOTE_BIND_ADDR:-server default}"

if [[ -n "$REMOTE_BIND_ADDR" ]]; then
    REMOTE_FORWARD_SPEC="$(format_remote_bind_addr_for_ssh "$REMOTE_BIND_ADDR"):$REMOTE_PORT:$REMOTE_FORWARD_TARGET"
else
    REMOTE_FORWARD_SPEC="$REMOTE_PORT:$REMOTE_FORWARD_TARGET"
fi

TUNNEL_SIG="-R $REMOTE_FORWARD_SPEC $REMOTE_TARGET"

# Common SSH Options
# - StrictHostKeyChecking=no: Don't ask to confirm fingerprint (crucial for automation)
# - UserKnownHostsFile=/dev/null: Don't save the temporary key
# - BatchMode=yes: Fails immediately if password is required (used for the check)
SSH_OPTS="-o ServerAliveInterval=$KEEP_ALIVE -o ServerAliveCountMax=$MAX_RETRIES -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=$SSH_CONNECT_TIMEOUT"
SESSION_NAME="ssh-tunnel-$REMOTE_PORT"
SCRIPT_PATH="$(readlink -f "$0")"

echo "--- SSH Reverse Tunnel Setup ---"

# 1. Kill existing tunnel
EXISTING_PID=$(pgrep -f "ssh .*$TUNNEL_SIG")
if [[ -n "$EXISTING_PID" ]]; then
        echo "🔄  Found active tunnel (PID $EXISTING_PID). Killing it..."
    kill -9 "$EXISTING_PID" 2>/dev/null
fi

check_batch_connection() {
    echo "🔍  Checking connection to '$REMOTE_TARGET'..."

    # BatchMode=yes will fail immediately if password/passphrase/host confirmation is needed.
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout="$CHECK_CONNECT_TIMEOUT" "$REMOTE_TARGET" exit 2>/dev/null
    CHECK_EXIT=$?

    if [ $CHECK_EXIT -ne 0 ]; then
        echo "⚠️  AUTHENTICATION REQUIRED OR CONNECTION FAILED"
        echo "   (SSH exit code: $CHECK_EXIT)"
        return $CHECK_EXIT
    fi

    BATCH_AUTH_AVAILABLE=true
    echo "✅  Connection Verified (SSH keys configured, no interaction required)."
    return 0
}

get_tmux_pane_output() {
    tmux capture-pane -pt "$SESSION_NAME" -S -120 2>/dev/null
}

tmux_pane_waiting_for_interaction() {
    local pane_text="$1"
    grep -Eiq 'password:|enter passphrase|are you sure you want to continue connecting|continue connecting \(yes/no(/\[fingerprint\])?\)\?' <<<"$pane_text"
}

tmux_pane_shows_failure() {
    local pane_text="$1"
    grep -Eiq 'permission denied|connection refused|connection timed out|operation timed out|no route to host|could not resolve hostname|host key verification failed|remote host identification has changed|port forwarding failed|administratively prohibited|connection closed by remote host|kex_exchange_identification' <<<"$pane_text"
}

remote_port_is_listening() {
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout="$PROBE_CONNECT_TIMEOUT" "$REMOTE_TARGET" \
        "ss -ltn | awk '\$4 ~ /:$REMOTE_PORT$/ { found=1 } END { exit !found }'" >/dev/null 2>&1
}

report_initial_tmux_status() {
    local elapsed=0
    local pane_text=""

    while [ "$elapsed" -lt "$STATUS_TIMEOUT" ]; do
        pane_text="$(get_tmux_pane_output)"

        if tmux_pane_waiting_for_interaction "$pane_text"; then
            echo "⚠️  Tunnel is waiting for user interaction in tmux session '$SESSION_NAME'."
            echo "    Attach with: tmux attach -t $SESSION_NAME"
            return 0
        fi

        if tmux_pane_shows_failure "$pane_text"; then
            echo "⚠️  Tunnel session '$SESSION_NAME' is running, but ssh is currently failing."
            echo "    Attach with: tmux attach -t $SESSION_NAME"
            return 0
        fi

        if [ "$BATCH_AUTH_AVAILABLE" = true ] && remote_port_is_listening; then
            echo "✅  Tunnel is up."
            echo "    Remote Bind: $REMOTE_BIND_DISPLAY | Remote Port: $REMOTE_PORT -> Local: $LOCAL_PORT"
            return 0
        fi

        sleep 1
        elapsed=$((elapsed + 1))
    done

    pane_text="$(get_tmux_pane_output)"
    if tmux_pane_waiting_for_interaction "$pane_text"; then
        echo "⚠️  Tunnel is waiting for user interaction in tmux session '$SESSION_NAME'."
    elif tmux_pane_shows_failure "$pane_text"; then
        echo "⚠️  Tunnel session '$SESSION_NAME' is running, but ssh is currently failing."
    else
        echo "ℹ️  Tunnel session '$SESSION_NAME' started, but readiness is not confirmed yet."
    fi
    echo "    Attach with: tmux attach -t $SESSION_NAME"
}

run_tunnel_loop() {
    while true; do
        echo "[$(date -Is)] Starting tunnel to '$REMOTE_TARGET'..."
        echo "    Target: '$REMOTE_TARGET' | Remote Bind: $REMOTE_BIND_DISPLAY | Remote Port: $REMOTE_PORT -> Local: $LOCAL_PORT"
        echo "    Session: $SESSION_NAME"
        echo ""

        # We remove BatchMode here so that if a password is needed, it can be entered interactively.
        ssh -N $SSH_OPTS $TUNNEL_SIG

        EXIT_CODE=$?
        echo ""
        echo "[$(date -Is)] Tunnel exited with code $EXIT_CODE."
        echo "[$(date -Is)] Retrying in $RETRY_DELAY seconds..."
        echo ""
        sleep "$RETRY_DELAY"
    done
}

if [ "$TMUX_CHILD_MODE" = true ]; then
    run_tunnel_loop
elif [ "$MODE" = "foreground" ]; then
    echo "🚀  Starting tunnel to '$REMOTE_TARGET' in FOREGROUND mode..."
    echo "    (Enter password or confirm host key here if prompted)..."
    run_tunnel_loop
elif [ "$MODE" = "background" ]; then
    if ! check_batch_connection; then
        echo ""
        echo "❌  Background mode requires passwordless, non-interactive SSH setup."
        exit 1
    fi

    echo "🚀  Starting tunnel to '$REMOTE_TARGET' in BACKGROUND mode..."
    SSH_CMD="ssh -N $SSH_OPTS $TUNNEL_SIG"
    nohup $SSH_CMD > /dev/null 2>&1 &
    NEW_PID=$!

    sleep 2

    if ps -p "$NEW_PID" > /dev/null; then
        echo "✅  Tunnel is running (PID: $NEW_PID)."
        echo "    Target: '$REMOTE_TARGET' | Remote Bind: $REMOTE_BIND_DISPLAY | Remote Port: $REMOTE_PORT -> Local: $LOCAL_PORT"
    else
        echo "❌  Tunnel failed to start."
        exit 1
    fi
elif [ "$MODE" = "tmux" ]; then
    if ! check_batch_connection; then
        echo ""
        echo "ℹ️  tmux mode will still be created."
        echo "   If interaction is needed, attach to the tmux session and respond there."
    fi

    if ! command -v tmux >/dev/null 2>&1; then
        echo "❌  tmux is required for --tmux mode."
        exit 1
    fi

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "🔄  Found existing tmux session '$SESSION_NAME'. Recreating it..."
        tmux kill-session -t "$SESSION_NAME"
    fi

    echo "🚀  Starting tunnel in tmux session '$SESSION_NAME'..."
    TMUX_CMD="$SCRIPT_PATH --tmux-child --remote-addr $(printf '%q' "$REMOTE_TARGET") --remote-port $(printf '%q' "$REMOTE_PORT") --local-port $(printf '%q' "$LOCAL_PORT") --keep-alive $(printf '%q' "$KEEP_ALIVE") --max-retries $(printf '%q' "$MAX_RETRIES") --ssh-connect-timeout $(printf '%q' "$SSH_CONNECT_TIMEOUT") --check-connect-timeout $(printf '%q' "$CHECK_CONNECT_TIMEOUT") --probe-connect-timeout $(printf '%q' "$PROBE_CONNECT_TIMEOUT") --retry-delay $(printf '%q' "$RETRY_DELAY") --status-timeout $(printf '%q' "$STATUS_TIMEOUT")"
    if [[ -n "$REMOTE_BIND_ADDR" ]]; then
        TMUX_CMD="$TMUX_CMD --remote-bind-addr $(printf '%q' "$REMOTE_BIND_ADDR")"
    fi
    tmux new-session -d -s "$SESSION_NAME" "$TMUX_CMD"

    echo "✅  tmux session '$SESSION_NAME' created."
    report_initial_tmux_status
    echo ""

    if [ "$ATTACH_MODE" = true ]; then
        if [ ! -t 0 ] || [ ! -t 1 ]; then
            echo "ℹ️  --attach requested, but no interactive TTY is available. Leaving session detached."
        elif [ -n "$TMUX" ]; then
            tmux switch-client -t "$SESSION_NAME"
        else
            tmux attach-session -t "$SESSION_NAME"
        fi
    else
        echo "ℹ️  Session left detached. Reattach with: tmux attach -t $SESSION_NAME"
    fi
else
    echo "Error: Unknown mode '$MODE'."
    exit 1
fi
