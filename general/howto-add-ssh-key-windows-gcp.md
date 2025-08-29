# How to Add SSH Key from Windows to GCP Instance

This guide explains how to add SSH keys from a Windows machine to Google Cloud Platform (GCP) Compute Engine instances for secure remote access.

## Prerequisites

- Google Cloud SDK installed on Windows
- Authenticated with `gcloud auth login`
- A GCP project set up with Compute Engine API enabled
- PowerShell or Command Prompt access

## Methods Overview

There are three main ways to add SSH keys to GCP instances:
1. **Project-wide metadata** (applies to all VMs in the project)
2. **Instance-specific metadata** (applies to a single VM)
3. **OS Login** (centralized key management via IAM)

## Method 1: Generate SSH Key Pair (if needed)

If you don't have an SSH key pair, generate one first:

```powershell
# Generate ED25519 key (recommended)
ssh-keygen -t ed25519 -f ~/.ssh/gcp_key -C "your_username"

# Or generate RSA key (alternative)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/gcp_key -C "your_username"
```

This creates two files:
- `~/.ssh/gcp_key` (private key - keep secure)
- `~/.ssh/gcp_key.pub` (public key - to be added to GCP)

## Method 2: Add SSH Key to Project-Wide Metadata

### Using gcloud CLI (Recommended)

**Option A: Simple addition (overwrites existing keys)**
```powershell
gcloud compute project-info add-metadata --metadata-from-file ssh-keys=~/.ssh/gcp_key.pub
```

**Option B: Preserve existing keys**
```powershell
gcloud compute project-info add-metadata \
--metadata ssh-keys="$(gcloud compute project-info describe \
--format="value(commonInstanceMetadata.items.filter(key:ssh-keys).firstof(value))")
your_username:$(Get-Content ~/.ssh/gcp_key.pub)"
```

### Using Google Cloud Console

1. Navigate to [Google Cloud Console](https://console.cloud.google.com)
2. Go to **Compute Engine** → **Metadata**
3. Click **SSH Keys** tab
4. Click **Edit** → **Add Item**
5. Paste your public key in format: `your_username:ssh-key-content`
6. Click **Save**

## Method 3: Add SSH Key to Specific Instance

### Format SSH Key Properly

Create a properly formatted SSH key file:

```powershell
# Read the public key and add username prefix
$sshKey = "your_username:" + (Get-Content ~/.ssh/gcp_key.pub)
$sshKey | Out-File -FilePath C:\temp\formatted_ssh_key.pub -Encoding utf8
```

### Add to Instance Metadata

```powershell
gcloud compute instances add-metadata INSTANCE_NAME \
--zone=ZONE_NAME \
--metadata-from-file ssh-keys=C:\temp\formatted_ssh_key.pub
```

### Example:
```powershell
gcloud compute instances add-metadata my-vm-instance \
--zone=us-central1-a \
--metadata-from-file ssh-keys=C:\temp\formatted_ssh_key.pub
```

## Method 4: Using OS Login (Enterprise/Organization)

### Enable OS Login
```powershell
# Enable at project level
gcloud compute project-info add-metadata --metadata enable-oslogin=TRUE

# Or enable for specific instance
gcloud compute instances add-metadata INSTANCE_NAME \
--zone=ZONE_NAME \
--metadata enable-oslogin=TRUE
```

### Add SSH Key to OS Login
```powershell
gcloud compute os-login ssh-keys add --key-file=~/.ssh/gcp_key.pub
```

### Grant IAM Permissions
```powershell
gcloud projects add-iam-policy-binding PROJECT_ID \
--member=user:YOUR_EMAIL \
--role=roles/compute.osLogin
```

## SSH Key Format Requirements

Your SSH key must be in this exact format:
```
username:ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... username@hostname
```

Key points:
- No newline characters
- All on one line
- Username prefix is crucial
- Space between key type and key content

## Configure SSH Client (Windows)

### Update SSH Config

Add an entry to your SSH config file (`~/.ssh/config`):

```ssh_config
Host gcp-vm-name
  HostName EXTERNAL_IP_ADDRESS
  User your_username
  IdentityFile ~/.ssh/gcp_key
  IdentitiesOnly yes
  ForwardX11 yes  # Optional: for GUI applications
```

### Example SSH Config Entry:
```ssh_config
Host my-gcp-server
  HostName 34.123.45.67
  User myuser
  IdentityFile ~/.ssh/gcp_key
  IdentitiesOnly yes
```

## Connect to Your Instance

### Using SSH Config Alias
```powershell
ssh gcp-vm-name
```

### Using gcloud (automatic key management)
```powershell
gcloud compute ssh INSTANCE_NAME --zone=ZONE_NAME
```

### Using Standard SSH
```powershell
ssh -i ~/.ssh/gcp_key your_username@EXTERNAL_IP_ADDRESS
```

## Verification Commands

### Check Project Metadata
```powershell
gcloud compute project-info describe \
--format="value(commonInstanceMetadata.items.filter(key:ssh-keys).firstof(value))"
```

### Check Instance Metadata
```powershell
gcloud compute instances describe INSTANCE_NAME \
--zone=ZONE_NAME \
--format="value(metadata.items.filter(key:ssh-keys).firstof(value))"
```

### Get Instance External IP
```powershell
gcloud compute instances describe INSTANCE_NAME \
--zone=ZONE_NAME \
--format="value(networkInterfaces[0].accessConfigs[0].natIP)"
```

## Troubleshooting

### Permission Denied Issues
1. Verify SSH key format is correct
2. Check that username matches between key and SSH command
3. Ensure private key has correct permissions (readable only by you)
4. Wait 1-2 minutes after adding keys for metadata to propagate

### Key Format Warnings
If you see "missing username" warnings:
```powershell
# Create properly formatted key file
"your_username:$(Get-Content ~/.ssh/gcp_key.pub)" | Out-File formatted_key.pub
gcloud compute project-info add-metadata --metadata-from-file ssh-keys=formatted_key.pub
```

### Firewall Issues
Ensure SSH firewall rule allows connections:
```powershell
gcloud compute firewall-rules create allow-ssh \
--allow tcp:22 \
--source-ranges 0.0.0.0/0 \
--description "Allow SSH access"
```

## Security Best Practices

1. **Use strong SSH keys**: Prefer ED25519 over RSA
2. **Protect private keys**: Never share or commit private keys
3. **Use IdentitiesOnly**: Prevents SSH from trying multiple keys
4. **Regular key rotation**: Rotate SSH keys periodically
5. **Use OS Login**: For better audit trails and centralized management
6. **Limit access**: Use instance-level keys for specific access requirements

## References

- [Official GCP SSH Keys Documentation](https://cloud.google.com/compute/docs/instances/adding-removing-ssh-keys)
- [Creating SSH Keys - Google Cloud](https://cloud.google.com/compute/docs/connect/create-ssh-keys)
- [OS Login Documentation](https://cloud.google.com/compute/docs/oslogin)
- [gcloud compute reference](https://cloud.google.com/sdk/gcloud/reference/compute)
