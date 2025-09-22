# How to Configure Git Submodules: HTTPS Pull, SSH Push

This guide explains how to configure git repositories and submodules to use HTTPS for pull/fetch operations (no authentication required) while using SSH for push operations (secure key-based authentication).

## Problem Statement

When working with git submodules, you often want:
- **Pull/Fetch**: Use HTTPS (no authentication, works behind firewalls)
- **Push**: Use SSH (secure, key-based authentication)

## Solution: URL Rewriting with `pushInsteadOf`

The most elegant solution is to configure git to automatically rewrite HTTPS URLs to SSH URLs for push operations only.

### Step 1: Configure URL Rewriting

In your main repository, run:

```bash
git config url."git@github.com:".pushInsteadOf "https://github.com/"
```

This tells git to:
- Use HTTPS URLs as-is for fetch/pull operations
- Automatically rewrite `https://github.com/` to `git@github.com:` for push operations

### Step 2: Configure Submodules with HTTPS URLs

Ensure your `.gitmodules` file uses HTTPS URLs:

```ini
[submodule ".magic-context"]
    path = .magic-context
    url = https://github.com/username/repo.git
```

### Step 3: Sync Submodule Configuration

After updating `.gitmodules`, sync the configuration:

```bash
git submodule sync .magic-context
```

## Verification

### Check Current Configuration

```bash
# View the URL rewriting configuration
git config --get-regexp 'url\..*\.pushinsteadof'

# Check submodule remote URLs
cd .magic-context
git remote -v
```

### Test the Setup

```bash
# Pull will use HTTPS (no auth required)
cd .magic-context
git pull origin main

# Push will automatically use SSH (requires SSH key)
git push origin main
```

## How It Works

1. **Fetch/Pull Operations**: Git uses the HTTPS URL directly from `.gitmodules`
   ```
   https://github.com/username/repo.git
   ```

2. **Push Operations**: Git automatically rewrites the URL using `pushInsteadOf`
   ```
   https://github.com/username/repo.git → git@github.com:username/repo.git
   ```

## Advantages

- ✅ **No authentication** required for read operations
- ✅ **Secure SSH authentication** for write operations
- ✅ **Works behind firewalls** (HTTPS for pulls)
- ✅ **Automatic URL rewriting** - no manual intervention needed
- ✅ **Repository-scoped** - doesn't affect global git configuration
- ✅ **Submodule-friendly** - works seamlessly with git submodules

## Alternative Methods

### Method 2: Separate Push URL (Per Repository)

Set different URLs for fetch and push on individual repositories:

```bash
cd submodule-directory
git remote set-url --push origin git@github.com:username/repo.git
```

### Method 3: Global Configuration (Not Recommended)

Apply URL rewriting globally (affects all repositories):

```bash
git config --global url."git@github.com:".pushInsteadOf "https://github.com/"
```

## Troubleshooting

### SSH Key Issues

If push fails with SSH authentication errors:

1. Verify SSH key is added to ssh-agent:
   ```bash
   ssh-add -l
   ```

2. Test SSH connection to GitHub:
   ```bash
   ssh -T git@github.com
   ```

3. Check SSH key is added to your GitHub account

### Configuration Not Working

Verify the configuration is set correctly:

```bash
# Check if pushInsteadOf is configured
git config --list | grep pushinsteadof

# Verify submodule URL
git config --get submodule.SUBMODULE_NAME.url
```

## References

- [Git Documentation - git-config](https://git-scm.com/docs/git-config#Documentation/git-config.txt-urlltbasegtinsteadOf)
- [Git Submodules Documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [GitHub SSH Key Setup](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)