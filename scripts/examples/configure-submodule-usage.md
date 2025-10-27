# Example Usage of configure-submodule-https-ssh.sh

This directory contains examples of how to use the configure-submodule-https-ssh.sh script.

## Basic Usage

```bash
# Configure a specific submodule
./configure-submodule-https-ssh.sh magic-context

# Configure all submodules
./configure-submodule-https-ssh.sh --all

# Preview changes without applying them
./configure-submodule-https-ssh.sh --dry-run magic-context

# Verify current configuration
./configure-submodule-https-ssh.sh --verify magic-context
```

## Typical Workflow

1. **Check current configuration:**
   ```bash
   ./configure-submodule-https-ssh.sh --verify magic-context
   ```

2. **Preview changes (dry run):**
   ```bash
   ./configure-submodule-https-ssh.sh --dry-run magic-context
   ```

3. **Apply configuration:**
   ```bash
   ./configure-submodule-https-ssh.sh magic-context
   ```

4. **Verify the result:**
   ```bash
   ./configure-submodule-https-ssh.sh --verify magic-context
   ```

## What the Script Does

1. **Updates .gitmodules** to use HTTPS URLs for the submodule
2. **Syncs submodule configuration** with `git submodule sync`
3. **Sets push URL to SSH** in the submodule's remote configuration
4. **Configures global URL rewriting** so future GitHub repositories automatically use SSH for push
5. **Verifies SSH connectivity** to GitHub (can be skipped with `--skip-ssh-check`)

## Expected Result

After running the script, your submodule will have:
- **Fetch URL**: `https://github.com/user/repo.git` (no authentication needed)
- **Push URL**: `git@github.com:user/repo.git` (SSH key required)

This allows you to pull changes without authentication while maintaining secure SSH-based pushes.