# How to Use Yunwu API with Codex CLI

This guide shows how to configure Codex CLI to use the Yunwu API provider for AI-powered coding assistance.

## Prerequisites

- Codex CLI installed (`npm install -g @openai/codex`)
- Yunwu API key (obtain from https://yunwu.ai)

## Configuration

### 1. TOML Configuration

Create or edit your Codex configuration file at `~/.codex/config.toml`:

```toml
# Model Provider Configuration
[model_providers.yunwu]
name = "yunwu-api"
base_url = "https://yunwu.ai/v1"
env_key = "YUNWU_OPENAI_KEY"
wire_api = "chat"

# Profile Configuration
[profiles.yunwu]
model = "gpt-5"
model_provider = "yunwu"
model_reasoning_effort = "medium"  # Options: "minimal", "low", "medium", "high"
model_reasoning_summary = "auto"   # Options: "none", "concise", "detailed", "auto"

[profiles.default]
model = "gpt-5"
model_provider = "yunwu"
model_reasoning_effort = "medium"
model_reasoning_summary = "auto"
```

### 2. Environment Variable Setup

Set your Yunwu API key as an environment variable:

```bash
# Set the API key for current session
export YUNWU_OPENAI_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Make it persistent (add to your shell profile)
echo 'export YUNWU_OPENAI_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

**Note**: Replace `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` with your actual Yunwu API key.

## Usage Examples

### Basic Commands

```bash
# Use default profile with medium reasoning effort
codex "Explain the architecture of this codebase"

# Override reasoning effort for complex tasks
codex --profile yunwu "Analyze this complex algorithm and explain the mathematical foundations"

# Quick responses with minimal reasoning
codex --profile yunwu "What's the basic syntax for Python classes?"
```

### Advanced Reasoning Configuration

You can create specialized profiles for different reasoning needs:

```toml
# High-effort profile for complex analysis
[profiles.yunwu-deep]
model = "gpt-5"
model_provider = "yunwu"
model_reasoning_effort = "high"
model_reasoning_summary = "detailed"

# Fast profile for quick queries
[profiles.yunwu-fast]
model = "gpt-5"
model_provider = "yunwu"
model_reasoning_effort = "low"
model_reasoning_summary = "none"
```

Usage examples:

```bash
# Deep analysis for complex problems
codex --profile yunwu-deep "Analyze the design patterns used in this project and suggest improvements"

# Quick answers for simple questions
codex --profile yunwu-fast "How do I install pytest?"
```

### Development Tasks

```bash
# Code review
codex --profile yunwu --file src/main.py "Review this code for potential improvements"

# Generate tests
codex --profile yunwu --file src/utils.py "Generate unit tests for this module"

# Documentation help
codex --profile yunwu "How to run tests in this project?"

# Debug assistance
codex --profile yunwu "Help me understand this error: ModuleNotFoundError"
```

### Project-Specific Examples

```bash
# Architecture analysis
codex --profile yunwu "What are the main components of this project?"

# Development workflow
codex --profile yunwu "How to set up the development environment?"

# Testing guidance
codex --profile yunwu "Explain the testing structure in this repository"
```

## Configuration Options

### Model Provider Fields

- `name`: Human-readable name for the provider
- `base_url`: API endpoint URL
- `env_key`: Environment variable containing the API key
- `wire_api`: API protocol ("chat" for OpenAI-compatible APIs)

### Profile Fields

- `model`: Model name to use (e.g., "gpt-5", "gpt-4", "o3", "o4-mini")
- `model_provider`: Reference to the provider configuration
- `model_reasoning_effort`: Controls how much computational effort the model uses for reasoning
  - `"minimal"`: Fastest, least thorough reasoning (GPT-5 series only)
  - `"low"`: Quick responses with basic reasoning
  - `"medium"`: Balanced reasoning and speed (default)
  - `"high"`: More thorough reasoning, slower responses
- `model_reasoning_summary`: Controls reasoning output visibility
  - `"none"`: No reasoning summaries shown
  - `"concise"`: Brief reasoning explanations
  - `"detailed"`: Full reasoning breakdowns
  - `"auto"`: System decides based on context (default)

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   Error: Environment variable YUNWU_OPENAI_KEY not set
   ```
   Solution: Ensure the environment variable is properly set and exported.

2. **Invalid API Key**
   ```
   Error: 401 Unauthorized
   ```
   Solution: Verify your API key is correct and has sufficient credits.

3. **Network Issues**
   ```
   Error: Connection timeout
   ```
   Solution: Check your internet connection and the Yunwu API status.

### Verification

Test your configuration:

```bash
# Check if environment variable is set
echo $YUNWU_OPENAI_KEY

# Test with a simple query
codex --profile yunwu "Hello, are you working?"

# List available profiles
codex --list-profiles
```

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** instead of hardcoding keys
3. **Set proper file permissions** on config files:
   ```bash
   chmod 600 ~/.codex/config.toml
   ```
4. **Add API keys to .gitignore** if stored in project files

## Additional Resources

- [Codex CLI Documentation](https://github.com/openai/codex)
- [Yunwu API Documentation](https://yunwu.ai/docs)
- [OpenAI API Compatibility Guide](https://platform.openai.com/docs/api-reference)
