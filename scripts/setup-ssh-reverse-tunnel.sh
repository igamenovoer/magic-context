#!/bin/bash

# Defaults
KEEP_ALIVE=60
BLOCK_MODE=false

# Function to show usage
usage() {
    echo "Usage: $0 --remote-addr <target> --remote-port <port> --local-port <port> [--keep-alive <sec>] [--block]"
    echo ""
    echo "  --remote-addr   The remote target. Formats allowed:"
    echo "                  1. User & IP:   user@192.168.1.1"
    echo "                  2. Alias:       myserver (from ~/.ssh/config)"
    echo "  --remote-port   The port on the Remote Server to open"
    echo "  --local-port    The port on your Local Machine to expose"
    echo "  --keep-alive    ServerAliveInterval in seconds (default: 60)"
    echo "  --block         Run in foreground (blocking) to enter password manually."
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --remote-addr) REMOTE_TARGET="$2"; shift ;;
        --remote-port) REMOTE_PORT="$2"; shift ;;
        --local-port) LOCAL_PORT="$2"; shift ;;
        --keep-alive) KEEP_ALIVE="$2"; shift ;;
        --block) BLOCK_MODE=true ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [[ -z "$REMOTE_TARGET" || -z "$REMOTE_PORT" || -z "$LOCAL_PORT" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# Unique signature to identify this tunnel process in 'pgrep'
TUNNEL_SIG="-R $REMOTE_PORT:127.0.0.1:$LOCAL_PORT $REMOTE_TARGET"

# Common SSH Options
# - StrictHostKeyChecking=no: Don't ask to confirm fingerprint (crucial for automation)
# - UserKnownHostsFile=/dev/null: Don't save the temporary key
# - BatchMode=yes: Fails immediately if password is required (used for the check)
SSH_OPTS="-o ServerAliveInterval=$KEEP_ALIVE -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"

echo "--- SSH Reverse Tunnel Setup ---"

# 1. Kill existing tunnel
EXISTING_PID=$(pgrep -f "ssh .*$TUNNEL_SIG")
if [[ -n "$EXISTING_PID" ]]; then
    echo "🔄  Found active tunnel (PID $EXISTING_PID). Killing it..."
    kill -9 "$EXISTING_PID" 2>/dev/null
fi

# 2. PRE-FLIGHT CHECK (Only if NOT in block mode)
# We test if the connection works WITHOUT a password.
if [ "$BLOCK_MODE" = false ]; then
    echo "🔍  Checking connection to '$REMOTE_TARGET'..."
    
    # Try to connect and exit immediately. 
    # BatchMode=yes will cause this to FAIL (exit 255) if a password is asked.
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 "$REMOTE_TARGET" exit 2>/dev/null
    
    CHECK_EXIT=$?
    
    if [ $CHECK_EXIT -ne 0 ]; then
        echo "⚠️  AUTHENTICATION REQUIRED OR CONNECTION FAILED"
        echo "   (SSH exit code: $CHECK_EXIT)"
        echo ""
        echo "   The script cannot run in background because '$REMOTE_TARGET'"
        echo "   requires a password or is unreachable."
        echo ""
        echo "👉  PLEASE RUN AGAIN WITH: --block"
        exit 1
    else
        echo "✅  Connection Verified (SSH Keys configured)."
    fi
fi

# 3. Start The Tunnel
# We remove BatchMode here so that if you use --block, you CAN type the password.
SSH_CMD="ssh -N $SSH_OPTS $TUNNEL_SIG"

if [ "$BLOCK_MODE" = true ]; then
    echo "🚀  Starting tunnel to '$REMOTE_TARGET' in BLOCKING mode..."
    echo "    (Enter password if prompted)..."
    $SSH_CMD
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌  Tunnel exited with error code $EXIT_CODE."
    fi
else
    echo "🚀  Starting tunnel to '$REMOTE_TARGET' in BACKGROUND mode..."
    nohup $SSH_CMD > /dev/null 2>&1 &
    NEW_PID=$!
    
    sleep 2
    
    if ps -p $NEW_PID > /dev/null; then
        echo "✅  Tunnel is running (PID: $NEW_PID)."
        echo "    Target: '$REMOTE_TARGET' | Remote Port: $REMOTE_PORT -> Local: $LOCAL_PORT"
        
        echo ""
        echo "========================================================"
        echo "   HOW TO ACCESS YOUR LOCAL SERVICE"
        echo "========================================================"
        echo "1. Public Access (if GatewayPorts yes): http://<remote-ip>:$REMOTE_PORT"
        echo "2. Private Access: SSH to remote, then curl http://localhost:$REMOTE_PORT"
        echo "========================================================"
    else
        echo "❌  Tunnel failed to start." 
        exit 1
    fi
fi