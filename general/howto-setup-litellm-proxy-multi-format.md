# How to Setup LiteLLM Proxy for Multi-Format Endpoints

This guide shows how to set up a LiteLLM proxy that exposes a remote OpenAI-compatible provider through multiple local endpoint formats (OpenAI, Gemini, Anthropic) without requiring a database.

## Overview

This setup allows you to:
- Access remote OpenAI-compatible APIs through a local proxy
- Expose the same models via multiple API formats (OpenAI, Gemini, Anthropic)
- Use environment variables for secure API key management
- Run without database dependencies for simpler deployment

## Prerequisites

1. **Install UV (if not already installed):**
   
   **Windows (PowerShell):**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   
   **Linux/macOS:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install LiteLLM with proxy support using UV:**
   ```bash
   uv tool install litellm[proxy]
   ```

3. **Verify installation:**
   ```bash
   litellm --help
   ```

## File Structure

Create the following directory structure:

```
litellm/
├── config.yaml           # LiteLLM configuration
├── .env.template         # Environment variable template
├── .env                  # Your actual environment variables (gitignored)
├── start-proxy.ps1       # PowerShell startup script
└── start-proxy.sh        # Bash startup script (optional)
```

## Configuration Files

### 1. Environment Variables Template (`.env.template`)

```env
# LiteLLM Proxy Environment Variables Template
# Copy this file to .env and fill in your actual values

# API Keys
OPENAI_API_KEY=your-api-key-here

# LiteLLM proxy settings
LITELLM_MASTER_KEY=sk-your-secure-master-key
LITELLM_PREFER_PORT=4444

# Optional: Additional API keys for other providers
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GEMINI_API_KEY=your-gemini-api-key

# Optional: Custom endpoint configurations
CUSTOM_API_BASE=https://your-custom-endpoint.com/v1
CUSTOM_API_KEY=your-custom-key
```

### 2. LiteLLM Configuration (`config.yaml`)

```yaml
model_list:
  # Main model configuration
  - model_name: gpt-5
    litellm_params:
      model: openai/gpt-5
      api_base: https://your-provider.ai/v1
      api_key: os.environ/OPENAI_API_KEY
      temperature: 1.0

  # Additional model variants (pointing to same backend)
  - model_name: gpt-5-2025-08-07
    litellm_params:
      model: openai/gpt-5-2025-08-07
      api_base: https://your-provider.ai/v1
      api_key: os.environ/OPENAI_API_KEY
      temperature: 1.0

  # Model aliases for different API formats
  - model_name: gemini-2.5-pro
    litellm_params:
      model: openai/gpt-5-2025-08-07
      api_base: https://your-provider.ai/v1
      api_key: os.environ/OPENAI_API_KEY
      temperature: 1.0

  - model_name: claude-opus-4-1-20250805
    litellm_params:
      model: openai/gpt-5-2025-08-07
      api_base: https://your-provider.ai/v1
      api_key: os.environ/OPENAI_API_KEY
      temperature: 1.0

# Proxy settings (disable UI to avoid database dependencies)
litellm_settings:
  # Disable admin UI -> avoids DB-backed pages / migrations
  ui: false
  # Format compatibility
  openai_compatible: true
  anthropic_compatible: true
  vertex_compatible: true
  drop_params: true  # Drop unknown params instead of erroring out

# General settings to disable database features
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY   # Provide master key here (not stored in DB)
  disable_spend_logs: true                    # Do not write spend logs to DB
  disable_error_logs: true                    # Do not write error logs to DB
  disable_adding_master_key_hash_to_db: true  # Do not store master key hash in DB
  allow_requests_on_db_unavailable: true      # Start/serve even if DB missing
  disable_reset_budget: true                  # Disable scheduled budget tasks (DB)
  # (No DATABASE_URL set; proxy runs statelessly.)
```

### 3. PowerShell Startup Script (`start-proxy.ps1`)

```powershell
<#
.SYNOPSIS
Starts LiteLLM proxy with configurable port and authentication.

.DESCRIPTION
This script starts a LiteLLM proxy server with configurable port and master key settings.
It automatically searches for .env and config.yaml files, supports environment variable
configuration, and provides multiple API format endpoints.

The script implements a priority system for configuration:
- Port: -Port parameter > LITELLM_PREFER_PORT env var > 4444 (default)
- Master Key: -MasterKey parameter > LITELLM_MASTER_KEY env var > "sk-none" (default)

File search order for .env and config.yaml:
1. Same directory as this script
2. Current working directory

.PARAMETER Port
Port number for the proxy server. If not specified, uses LITELLM_PREFER_PORT environment
variable or defaults to 4444.

.PARAMETER MasterKey
Master key for authentication. If not specified, uses LITELLM_MASTER_KEY environment
variable or defaults to "sk-none".

.PARAMETER EnvFile
Path to a custom .env file. If not specified, automatically searches for .env files
in the script directory first, then the current working directory.

.INPUTS
None. This script does not accept pipeline input.

.OUTPUTS
None. This script starts a background service and outputs status messages.

.EXAMPLE
.\start-proxy.ps1
Starts the proxy using environment variables or defaults for all settings.

.EXAMPLE
.\start-proxy.ps1 -Port 8080
Starts the proxy on port 8080, using environment variables or defaults for other settings.

.EXAMPLE
.\start-proxy.ps1 -MasterKey "sk-my-key"
Starts the proxy with a custom master key, using environment variables or defaults for other settings.

.EXAMPLE
.\start-proxy.ps1 -Port 8080 -MasterKey "sk-my-key"
Starts the proxy with both custom port and master key.

.EXAMPLE
.\start-proxy.ps1 -EnvFile "C:\path\to\.env"
Starts the proxy using a custom .env file path.

.EXAMPLE
.\start-proxy.ps1 -Port 8080 -EnvFile ".\custom.env"
Starts the proxy with custom port and .env file.

.NOTES
PREREQUISITES:
1. Install LiteLLM: uv tool install litellm[proxy]
2. Create .env file from .env.template with your API keys
3. Ensure config.yaml exists in script or current directory

ENVIRONMENT VARIABLES:
- LITELLM_PREFER_PORT: Preferred port number (used when -Port is not specified)
- LITELLM_MASTER_KEY: Master key for authentication (used when -MasterKey is not specified)

AVAILABLE ENDPOINTS (after starting):
- OpenAI format: http://localhost:PORT/v1/chat/completions
- Gemini format: http://localhost:PORT/gemini/v1beta/models/{model}:generateContent
- Anthropic format: http://localhost:PORT/anthropic/v1/messages
- Health check: http://localhost:PORT/health
- Models list: http://localhost:PORT/v1/models
- Admin UI (if enabled): http://localhost:PORT/ui

TESTING EXAMPLES:
# Test with curl (replace PORT and MASTER_KEY with actual values)
curl http://localhost:4444/v1/models -H "Authorization: Bearer your-master-key"

# Configure gemini-cli (replace PORT and MASTER_KEY with actual values)
$env:GOOGLE_GEMINI_BASE_URL = "http://localhost:4444"
$env:GEMINI_API_KEY = "your-master-key"

.LINK
https://github.com/BerriAI/litellm
#>

param(
    [int]$Port = 0,  # 0 means use environment variable or default
    [string]$EnvFile = "",
    [string]$MasterKey = ""  # Empty means use environment variable or default
)

# Determine the port to use: specified args > LITELLM_PREFER_PORT > 4444
if ($Port -eq 0) {
    $preferredPort = [Environment]::GetEnvironmentVariable("LITELLM_PREFER_PORT")
    if ($preferredPort -and $preferredPort -ne "" -and [int]::TryParse($preferredPort, [ref]$null)) {
        $Port = [int]$preferredPort
        Write-Host "Using port from LITELLM_PREFER_PORT environment variable: $Port" -ForegroundColor Cyan
    } else {
        $Port = 4444
        Write-Host "Using default port: $Port" -ForegroundColor Gray
    }
} else {
    Write-Host "Using port specified via argument: $Port" -ForegroundColor Cyan
}

# Determine the master key to use: specified args > LITELLM_MASTER_KEY > "sk-none"
if ($MasterKey -eq "") {
    $envMasterKey = [Environment]::GetEnvironmentVariable("LITELLM_MASTER_KEY")
    if ($envMasterKey -and $envMasterKey -ne "") {
        $MasterKey = $envMasterKey
        Write-Host "Using master key from LITELLM_MASTER_KEY environment variable" -ForegroundColor Cyan
    } else {
        $MasterKey = "sk-none"
        Write-Host "Warning: Using default master key 'sk-none' - set LITELLM_MASTER_KEY or use -MasterKey parameter" -ForegroundColor Yellow
    }
} else {
    Write-Host "Using master key specified via argument" -ForegroundColor Cyan
}

# Check if .env file exists - use custom path if provided, otherwise search in script dir then pwd
if ($EnvFile -eq "") {
    # Search order: 1) script directory, 2) current working directory
    $scriptDirEnv = Join-Path $PSScriptRoot ".env"
    $pwdEnv = Join-Path (Get-Location) ".env"
    
    if (Test-Path $scriptDirEnv) {
        $envFile = $scriptDirEnv
        Write-Host "Found .env file in script directory: $envFile" -ForegroundColor Cyan
    } elseif (Test-Path $pwdEnv) {
        $envFile = $pwdEnv
        Write-Host "Found .env file in current directory: $envFile" -ForegroundColor Cyan
    } else {
        $envFile = $pwdEnv  # Default path for error message
    }
} else {
    $envFile = $EnvFile
}

if (Test-Path $envFile) {
    Write-Host "Loading environment variables from: $envFile" -ForegroundColor Green
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([^#][^=]*)=(.*)$') {
            $name = $matches[1]
            $value = $matches[2]
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
} else {
    if ($EnvFile -eq "") {
        Write-Host "Warning: .env file not found in script directory ($PSScriptRoot) or current directory ($(Get-Location))" -ForegroundColor Yellow
        Write-Host "Please create .env file from .env.template in either location" -ForegroundColor Yellow
    } else {
        Write-Host "Warning: .env file not found at: $envFile" -ForegroundColor Yellow
    }
    Write-Host "Environment variables should be set manually or via system environment" -ForegroundColor Yellow
}

# Check if config.yaml exists - search in script dir first, then current directory
$scriptDirConfig = Join-Path $PSScriptRoot "config.yaml"
$pwdConfig = Join-Path (Get-Location) "config.yaml"

if (Test-Path $scriptDirConfig) {
    $configFile = $scriptDirConfig
    Write-Host "Found config.yaml in script directory: $configFile" -ForegroundColor Cyan
} elseif (Test-Path $pwdConfig) {
    $configFile = $pwdConfig
    Write-Host "Found config.yaml in current directory: $configFile" -ForegroundColor Cyan
} else {
    Write-Host "Error: config.yaml not found in script directory ($PSScriptRoot) or current directory ($(Get-Location))" -ForegroundColor Red
    exit 1
}

Write-Host "Starting LiteLLM proxy..." -ForegroundColor Green
Write-Host "Config file: $configFile" -ForegroundColor Gray
Write-Host "Proxy will be available at: http://localhost:$Port" -ForegroundColor Gray
Write-Host "Admin UI (if enabled): http://localhost:$Port/ui" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop the proxy" -ForegroundColor Yellow
Write-Host ""

# Start LiteLLM proxy
try {
    # Set the master key for the litellm process
    [Environment]::SetEnvironmentVariable("LITELLM_MASTER_KEY", $MasterKey, "Process")
    litellm --config $configFile --port $Port --host 0.0.0.0
}
catch {
    Write-Host "Error starting LiteLLM proxy: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Make sure LiteLLM is installed: uv tool install litellm[proxy]" -ForegroundColor Yellow
    exit 1
}
```

### 4. Bash Startup Script (`start-proxy.sh`) - Optional

```bash
#!/bin/bash
# LiteLLM Proxy Startup Script for Linux/macOS

# Default port
PORT=${1:-4444}

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Please create one from .env.template"
fi

# Check if config.yaml exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config.yaml not found in $SCRIPT_DIR"
    exit 1
fi

echo "Starting LiteLLM proxy..."
echo "Config file: $CONFIG_FILE"
echo "Proxy will be available at: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the proxy"
echo ""

# Start LiteLLM proxy
litellm --config "$CONFIG_FILE" --port $PORT --host 0.0.0.0
```

## Setup Steps

### 1. Create Environment File

```bash
# Copy the template
cp .env.template .env

# Edit .env with your actual values
# Replace 'your-api-key-here' with your actual API key
# Replace 'sk-your-secure-master-key' with a secure master key
```

### 2. Update Configuration

Edit `config.yaml` to:
- Replace `https://your-provider.ai/v1` with your actual API base URL
- Adjust model names to match your provider's offerings
- Modify temperature and other parameters as needed

### 3. Start the Proxy

**Windows (PowerShell):**
```powershell
# Get help for the script
Get-Help .\start-proxy.ps1
Get-Help .\start-proxy.ps1 -Examples

# Start with defaults (port from LITELLM_PREFER_PORT env var or 4444)
.\start-proxy.ps1

# Start with custom port
.\start-proxy.ps1 -Port 8080

# Start with custom master key
.\start-proxy.ps1 -MasterKey "sk-my-secure-key"

# Start with custom port and master key
.\start-proxy.ps1 -Port 8080 -MasterKey "sk-my-secure-key"

# Start with custom .env file
.\start-proxy.ps1 -EnvFile "C:\path\to\custom\.env"

# Start with custom port and .env file
.\start-proxy.ps1 -Port 8080 -EnvFile ".\custom.env"
```

**Linux/macOS (Bash):**
```bash
chmod +x start-proxy.sh

# Start with default port (4444)
./start-proxy.sh

# Start with custom port
./start-proxy.sh 8080
```

## Available Endpoints

Once running, the proxy exposes multiple API formats on `http://localhost:PORT` (default PORT is 4444):

### OpenAI Format (Default)
```bash
# List models (replace 4444 with your port if different)
curl http://localhost:4444/v1/models \
  -H "Authorization: Bearer your-litellm-master-key"

# Chat completion
curl http://localhost:4444/v1/chat/completions \
  -H "Authorization: Bearer your-litellm-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Gemini Format
```bash
# Configure gemini-cli (replace 4444 with your port if different)
export GOOGLE_GEMINI_BASE_URL="http://localhost:4444"
export GEMINI_API_KEY="your-litellm-master-key"

# Use gemini-cli
gemini chat -m gemini-2.5-pro "Hello"
```

### Anthropic Format
```bash
# Replace 4444 with your port if different
curl http://localhost:4444/anthropic/v1/messages \
  -H "Authorization: Bearer your-litellm-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4-1-20250805",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

## Python Client Examples

### OpenAI Format
```python
import openai

# Replace 4444 with your port if different
client = openai.OpenAI(
    api_key="your-litellm-master-key",
    base_url="http://localhost:4444"
)

response = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "Hello"}]
)

print(response.choices[0].message.content)
```

### Using Different Models Through Same Client
```python
# Same client, different models (all routed to same backend)
models = ["gpt-5", "gemini-2.5-pro", "claude-opus-4-1-20250805"]

for model in models:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"Hello from {model}"}]
    )
    print(f"{model}: {response.choices[0].message.content}")
```

## Key Features

### Database-Free Operation
- No PostgreSQL or database setup required
- All configuration via YAML and environment variables
- Stateless operation for simple deployment

### Multi-Format Support
- **OpenAI Format**: Standard `/v1/chat/completions` endpoint
- **Gemini Format**: Compatible with `gemini-cli` and Google AI SDKs
- **Anthropic Format**: Compatible with Anthropic SDKs

### Security
- API keys stored in environment variables, not in config files
- Master key authentication for all requests
- No sensitive data in version control

### Environment Variable Loading
- Automatic `.env` file loading via startup scripts
- Cross-platform support (PowerShell and Bash)
- Graceful fallback to system environment variables

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Use a different port
   .\start-proxy.ps1 -Port 4445
   # Or for bash
   ./start-proxy.sh 4445
   ```

2. **Environment variables not loaded:**
   - Ensure `.env` file exists and is properly formatted
   - Check that startup script has execution permissions

3. **API key errors:**
   - Verify API keys are valid and have proper permissions
   - Check that environment variables are set correctly

4. **Model not found:**
   - Ensure model names in requests match those in `config.yaml`
   - Check that the backend provider supports the requested model

### Health Check

```bash
# Basic health check (replace 4444 with your port if different)
curl http://localhost:4444/health

# Verify authentication
curl http://localhost:4444/v1/models \
  -H "Authorization: Bearer your-litellm-master-key"
```

## Security Notes

- Keep your `.env` file in `.gitignore` to prevent API key exposure
- Use strong, unique master keys for production deployments
- Consider using environment-specific configurations for different deployments
- Regularly rotate API keys and master keys

## Deployment Considerations

### Development
- Use the provided scripts for quick local setup
- Keep database features disabled for simplicity

### Production
- Consider using Docker containers for deployment
- Set up proper logging and monitoring
- Use secrets management instead of `.env` files
- Consider enabling database features for usage tracking

### Multiple Environments
- Create separate configuration files for different environments
- Use environment-specific API keys and endpoints
- Consider using configuration management tools for larger deployments

## References

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [LiteLLM Proxy Configuration](https://docs.litellm.ai/docs/proxy/quick_start)
- [Environment Variable Best Practices](https://12factor.net/config)
