#!/bin/bash

# Timing / retry configuration defaults
KEEP_ALIVE=20
MAX_RETRIES=60
SSH_CONNECT_TIMEOUT=10
CHECK_CONNECT_TIMEOUT=10
RETRY_DELAY=10
STATUS_TIMEOUT=10

# Runtime mode defaults
MODE="foreground"
TMUX_CHILD_MODE=false
BATCH_AUTH_AVAILABLE=false
ATTACH_MODE=false

usage() {
    echo "Usage: $0 --remote-addr <target> --remote-port <port> --local-port <port> [--keep-alive <sec>] [--max-retries <count>] [--ssh-connect-timeout <sec>] [--check-connect-timeout <sec>] [--retry-delay <sec>] [--status-timeout <sec>] [--background|--tmux] [--attach]"
    echo ""
    echo "  --remote-addr   SSH target. Formats: user@host or alias from ~/.ssh/config"
    echo "  --remote-port   Port on the remote host to forward locally"
    echo "  --local-port    Local port to bind (127.0.0.1:<local-port> -> remote:127.0.0.1:<remote-port>)"
    echo "  --keep-alive    ServerAliveInterval in seconds (default: 20)"
    echo "  --max-retries   ServerAliveCountMax (default: 60)"
    echo "  --ssh-connect-timeout SSH ConnectTimeout for the tunnel (default: 10)"
    echo "  --check-connect-timeout SSH ConnectTimeout for the preflight check (default: 10)"
    echo "  --retry-delay   Seconds to wait before reconnecting after ssh exits (default: 10)"
    echo "  --status-timeout Seconds to wait for initial tunnel status (default: 10)"
    echo "  --background    Start a single background ssh process with nohup"
    echo "  --tmux          Run the reconnect loop inside tmux session ssh-fwd-<local-port>"
    echo "  --attach        Attach or switch into the tmux session after it is created"
    echo ""
    echo "Default mode: run the reconnect loop in the current terminal (foreground)."
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --remote-addr) REMOTE_TARGET="$2"; shift ;;
        --remote-port) REMOTE_PORT="$2"; shift ;;
        --local-port) LOCAL_PORT="$2"; shift ;;
        --keep-alive) KEEP_ALIVE="$2"; shift ;;
        --max-retries) MAX_RETRIES="$2"; shift ;;
        --ssh-connect-timeout) SSH_CONNECT_TIMEOUT="$2"; shift ;;
        --check-connect-timeout) CHECK_CONNECT_TIMEOUT="$2"; shift ;;
        --retry-delay) RETRY_DELAY="$2"; shift ;;
        --status-timeout) STATUS_TIMEOUT="$2"; shift ;;
        --background) MODE="background" ;;
        --tmux) MODE="tmux" ;;
        --attach) ATTACH_MODE=true ;;
        --block) MODE="foreground" ;;
        --tmux-child) TMUX_CHILD_MODE=true ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
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

# Forward tunnel: bind local port -> remote localhost:remote-port
TUNNEL_SIG="-L $LOCAL_PORT:127.0.0.1:$REMOTE_PORT $REMOTE_TARGET"

SSH_OPTS="-o ServerAliveInterval=$KEEP_ALIVE -o ServerAliveCountMax=$MAX_RETRIES -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=$SSH_CONNECT_TIMEOUT"
SESSION_NAME="ssh-fwd-$LOCAL_PORT"
SCRIPT_PATH="$(readlink -f "$0")"

echo "--- SSH Forward Tunnel Setup ---"

# Kill any existing tunnel with the same signature
EXISTING_PID=$(pgrep -f "ssh .*$TUNNEL_SIG")
if [[ -n "$EXISTING_PID" ]]; then
    echo "🔄  Found active tunnel (PID $EXISTING_PID). Killing it..."
    kill -9 "$EXISTING_PID" 2>/dev/null
fi

check_batch_connection() {
    echo "🔍  Checking connection to '$REMOTE_TARGET'..."
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout="$CHECK_CONNECT_TIMEOUT" "$REMOTE_TARGET" exit 2>/dev/null
    CHECK_EXIT=$?
    if [ $CHECK_EXIT -ne 0 ]; then
        echo "⚠️  AUTHENTICATION REQUIRED OR CONNECTION FAILED (SSH exit code: $CHECK_EXIT)"
        return $CHECK_EXIT
    fi
    BATCH_AUTH_AVAILABLE=true
    echo "✅  Connection verified (passwordless SSH)."
    return 0
}

local_port_is_listening() {
    ss -ltn 2>/dev/null | awk "\$4 ~ /:$LOCAL_PORT\$/ { found=1 } END { exit !found }"
}

get_tmux_pane_output() {
    tmux capture-pane -pt "$SESSION_NAME" -S -120 2>/dev/null
}

tmux_pane_waiting_for_interaction() {
    grep -Eiq 'password:|enter passphrase|are you sure you want to continue connecting|continue connecting \(yes/no' <<<"$1"
}

tmux_pane_shows_failure() {
    grep -Eiq 'permission denied|connection refused|connection timed out|no route to host|could not resolve hostname|host key verification failed|port forwarding failed|bind: address already in use' <<<"$1"
}

report_initial_tmux_status() {
    local elapsed=0 pane_text=""
    while [ "$elapsed" -lt "$STATUS_TIMEOUT" ]; do
        pane_text="$(get_tmux_pane_output)"
        if tmux_pane_waiting_for_interaction "$pane_text"; then
            echo "⚠️  Tunnel waiting for interaction. Attach: tmux attach -t $SESSION_NAME"
            return 0
        fi
        if tmux_pane_shows_failure "$pane_text"; then
            echo "⚠️  Tunnel failing. Attach: tmux attach -t $SESSION_NAME"
            return 0
        fi
        if [ "$BATCH_AUTH_AVAILABLE" = true ] && local_port_is_listening; then
            echo "✅  Tunnel is up."
            echo "    localhost:$LOCAL_PORT -> $REMOTE_TARGET:127.0.0.1:$REMOTE_PORT"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    pane_text="$(get_tmux_pane_output)"
    if tmux_pane_waiting_for_interaction "$pane_text"; then
        echo "⚠️  Tunnel waiting for interaction."
    elif tmux_pane_shows_failure "$pane_text"; then
        echo "⚠️  Tunnel session running but ssh is failing."
    else
        echo "ℹ️  Tunnel session started; readiness not confirmed yet."
    fi
    echo "    Attach with: tmux attach -t $SESSION_NAME"
}

run_tunnel_loop() {
    while true; do
        echo "[$(date -Is)] Starting forward tunnel..."
        echo "    localhost:$LOCAL_PORT -> $REMOTE_TARGET:127.0.0.1:$REMOTE_PORT"
        echo ""
        ssh -N $SSH_OPTS $TUNNEL_SIG
        EXIT_CODE=$?
        echo ""
        echo "[$(date -Is)] Tunnel exited (code $EXIT_CODE). Retrying in ${RETRY_DELAY}s..."
        echo ""
        sleep "$RETRY_DELAY"
    done
}

if [ "$TMUX_CHILD_MODE" = true ]; then
    run_tunnel_loop
elif [ "$MODE" = "foreground" ]; then
    echo "🚀  Starting tunnel in FOREGROUND mode..."
    run_tunnel_loop
elif [ "$MODE" = "background" ]; then
    if ! check_batch_connection; then
        echo "❌  Background mode requires passwordless SSH."
        exit 1
    fi
    echo "🚀  Starting tunnel in BACKGROUND mode..."
    nohup ssh -N $SSH_OPTS $TUNNEL_SIG > /dev/null 2>&1 &
    NEW_PID=$!
    sleep 2
    if ps -p "$NEW_PID" > /dev/null; then
        echo "✅  Tunnel running (PID: $NEW_PID)."
        echo "    localhost:$LOCAL_PORT -> $REMOTE_TARGET:127.0.0.1:$REMOTE_PORT"
    else
        echo "❌  Tunnel failed to start."
        exit 1
    fi
elif [ "$MODE" = "tmux" ]; then
    if ! check_batch_connection; then
        echo "ℹ️  tmux session will be created; attach if interaction is needed."
    fi
    if ! command -v tmux >/dev/null 2>&1; then
        echo "❌  tmux is required for --tmux mode."
        exit 1
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "🔄  Recreating existing tmux session '$SESSION_NAME'..."
        tmux kill-session -t "$SESSION_NAME"
    fi
    echo "🚀  Starting tunnel in tmux session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME" \
        "$SCRIPT_PATH --tmux-child \
        --remote-addr $(printf '%q' "$REMOTE_TARGET") \
        --remote-port $(printf '%q' "$REMOTE_PORT") \
        --local-port $(printf '%q' "$LOCAL_PORT") \
        --keep-alive $(printf '%q' "$KEEP_ALIVE") \
        --max-retries $(printf '%q' "$MAX_RETRIES") \
        --ssh-connect-timeout $(printf '%q' "$SSH_CONNECT_TIMEOUT") \
        --check-connect-timeout $(printf '%q' "$CHECK_CONNECT_TIMEOUT") \
        --retry-delay $(printf '%q' "$RETRY_DELAY") \
        --status-timeout $(printf '%q' "$STATUS_TIMEOUT")"
    echo "✅  tmux session '$SESSION_NAME' created."
    report_initial_tmux_status
    echo ""
    if [ "$ATTACH_MODE" = true ]; then
        if [ ! -t 0 ] || [ ! -t 1 ]; then
            echo "ℹ️  No interactive TTY; leaving session detached."
        elif [ -n "$TMUX" ]; then
            tmux switch-client -t "$SESSION_NAME"
        else
            tmux attach-session -t "$SESSION_NAME"
        fi
    else
        echo "ℹ️  Reattach with: tmux attach -t $SESSION_NAME"
    fi
else
    echo "Error: Unknown mode '$MODE'."
    exit 1
fi
