# Claude Code CLI

> **Note:** Before running the `.ps1` script, please run the `<workspace>/enable-ps1-permission.bat` script once to allow PowerShell script execution.

This component installs the Claude Code CLI used by Anthropic’s Claude Code tooling.

## Preferred installation (requires Node.js)

- Ensure Node.js is installed (prefer via winget, see `components/nodejs/README.md`).
- Global installation from npm:
  ```powershell
  npm install -g @anthropic-ai/claude-code
  ```

After installation, Claude Code CLI normally launches an onboarding/login flow the first time you run `claude`. The per-component installer for `claude-code-cli` (its `install-comp.ps1` inside this directory) is responsible for marking onboarding as complete by updating `%USERPROFILE%\.claude.json` (setting `hasCompletedOnboarding = true`), so you can start using the CLI immediately without going through the login wizard on this machine.

## China-friendly installation (npm mirrors)

- For Chinese networks, point npm to a faster mirror before installing:
  ```powershell
  npm config set registry https://registry.npmmirror.com
  npm config get registry
  ```
- Then install:
  ```powershell
  npm install -g @anthropic-ai/claude-code
  ```
- Our `install-comp` script will:
  - Prefer `npm` with `https://registry.npmmirror.com` (or another configured mirror) by default in China.
  - Respect `--proxy / -Proxy` and `--from-official`:
    - `--from-official` forces `https://registry.npmjs.org` as the registry.
   - As a post-install step (implemented inside this component’s own `install-comp.ps1`), it will:
     - Verify `claude` is available on `PATH`.
     - Create or update `%USERPROFILE%\.claude.json` with `hasCompletedOnboarding = true` (UTF-8 without BOM).
     - Ensure subsequent `claude` invocations skip the interactive login/onboarding flow on this host.

## Official installation

- Official docs and package:
  - npm: https://www.npmjs.com/package/@anthropic-ai/claude-code
  - Anthropic docs: https://docs.anthropic.com/
- When `--from-official` is used, the installer will:
  - Set `npm config set registry https://registry.npmjs.org` for the install step.

## Additional helpers in this directory

Besides `install-comp.ps1` / `install-comp.bat`, this component exposes several helper scripts:

- `config-skip-login.bat` / `.ps1`  
  - Marks onboarding as completed in `%USERPROFILE%\.claude.json` so `claude` 不再弹出首启 / 登录向导，适合在新机器上直接进入可用状态。  
- `config-custom-api-key.bat` / `.ps1`  
  - 在 PowerShell 用户配置中写入一个自定义函数（别名），启动前自动设置 `ANTHROPIC_BASE_URL`、`ANTHROPIC_API_KEY`（以及可选的主/副模型环境变量），然后执行 `claude --dangerously-skip-permissions`，方便对接 Kimi、SiliconFlow 等兼容 Anthropic 协议的服务。  
- `config-context7-mcp.bat` / `.ps1`  
  - 使用 npm / `npx` 安装并注册 Context7 MCP 服务器（`@upstash/context7-mcp`），在 Claude Code 的 MCP 配置中添加名为 `context7` 的服务器，让 Claude 可以按需查询最新、指定版本的库 / 框架文档。  
- `config-tavily-mcp.bat` / `.ps1`  
  - 安装 Tavily MCP、写入 Tavily API Key，并在 Claude Code 的 MCP 配置里注册 `tavily` 服务器，为 Claude 提供联网搜索 / 新闻检索等能力。  

通常的使用顺序是：先运行 `install-comp.bat` 安装 Claude Code CLI，再按需运行上述配置脚本。  

## Linux/macOS (POSIX) scripts

- Install:
  ```bash
  cd components/claude-code-cli
  sh ./install-comp.sh --dry-run
  sh ./install-comp.sh
  ```
- Helpers:
  ```bash
  sh ./config-skip-login.sh
  sh ./config-custom-api-key.sh --alias-name claude-kimi --base-url "https://example.com/anthropic/"
  sh ./config-context7-mcp.sh
  sh ./config-tavily-mcp.sh
  ```
- Notes:
  - `config-custom-api-key.sh` writes a launcher under `~/.local/bin/<alias>` and stores the key in plain text.
