#!/bin/bash
# Start ssh-agent if not already running and add all SSH keys

# Check if ssh-agent is already running
if [ -z "$SSH_AUTH_SOCK" ] || ! ssh-add -l &>/dev/null; then
    echo "Starting ssh-agent..."
    eval "$(ssh-agent -s)"
else
    echo "ssh-agent is already running (PID: $SSH_AGENT_PID)"
fi

# Add all keys from ~/.ssh directory
echo "Adding SSH keys..."
for key in ~/.ssh/id_rsa ~/.ssh/id_ed25519 ~/.ssh/id_ecdsa ~/.ssh/id_dsa; do
    if [ -f "$key" ]; then
        echo "Adding key: $key"
        ssh-add "$key" 2>/dev/null || true
    fi
done

# List currently loaded keys
echo ""
echo "Currently loaded keys:"
ssh-add -l || echo "No SSH keys currently loaded"
