# How to install Mermaid CLI (mmdc) and make it work on Linux

This guide shows a reliable way to install `@mermaid-js/mermaid-cli` and all required browser/runtime dependencies on Ubuntu/Debian servers and containers, plus a quick inline test to verify everything works.

## Prerequisites
- Node.js 18+ (Node 20/22 recommended)
- Bash shell
- Ubuntu/Debian (commands below use apt)

## Install mermaid-cli

You can use a global install or `npx` on demand. Either works.

- Global install (convenient if you’ll use it often):

```bash
npm install -g @mermaid-js/mermaid-cli
```

- One-off usage (no global install needed):

```bash
npx -y @mermaid-js/mermaid-cli --version
```

Both methods provide the `mmdc` command.

## Install a compatible headless Chrome for Puppeteer

Mermaid CLI uses `puppeteer-core` and does NOT bundle a browser. On first run, you’ll often see an error like:

```
Error: Could not find Chrome (ver. 131.0.6778.204)
```

Fix by installing that exact revision into Puppeteer’s cache:

```bash
# Replace 131.0.6778.204 with the version printed in your error message
npx -y puppeteer browsers install chrome-headless-shell@131.0.6778.204
```

Tips:
- The cache lives under `~/.cache/puppeteer` by default. You can change it with `PUPPETEER_CACHE_DIR`.
- If you prefer using a system Chromium/Chrome, set `PUPPETEER_EXECUTABLE_PATH` to the browser binary instead of installing the headless shell.

## Install Linux shared libraries Chrome needs (Ubuntu/Debian)

Headless Chrome requires several native libraries. If you see errors like `error while loading shared libraries: libatk-1.0.so.0`, install these:

```bash
sudo apt-get update && sudo apt-get install -y \
  libgtk-3-0 libgbm1 libnss3 libasound2 libxss1 \
  libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
  libxdamage1 libxfixes3 libglib2.0-0 libxshmfence1 \
  libxcomposite1 libxrandr2 libxrender1 libxkbcommon0 \
  libpango-1.0-0 libcairo2 fonts-liberation
```

## Avoid sandbox issues (containers/CI)

In some environments (Docker, CI, limited permissions) Chrome’s sandbox can fail. Use a Puppeteer config that passes `--no-sandbox` flags. This repo includes `tmp/puppeteer-config.json`:

```json
{
  "args": ["--no-sandbox", "--disable-setuid-sandbox"]
}
```

Use it via `-p`:

```bash
mmdc -i input.mmd -o output.svg -p ./tmp/puppeteer-config.json
```

You can also specify a system browser path in the same config:

```json
{
  "executablePath": "/usr/bin/chromium-browser",
  "args": ["--no-sandbox", "--disable-setuid-sandbox"]
}
```

## Quick verification (inline Mermaid via stdin)

You don’t need an input file to test. Pipe Mermaid code to stdin and generate an SVG:

```bash
# Creates ./tmp/inline.svg
mkdir -p ./tmp

echo 'graph TD; A-->B' | mmdc -i - -o ./tmp/inline.svg -p ./tmp/puppeteer-config.json
```

- If this succeeds, your installation is good.
- If you now see a Mermaid “Parse error,” it means Chrome launched fine; fix your diagram syntax instead.

Optional checks:

```bash
mmdc --version
ls -l ./tmp/inline.svg
```

## Troubleshooting

- Could not find Chrome (ver. X):
  - Install the exact headless shell revision with `npx puppeteer browsers install chrome-headless-shell@X`.
- error while loading shared libraries: libatk-1.0.so.0:
  - Install the apt packages listed above for Ubuntu/Debian.
- Timeout or sandbox errors in CI/Docker:
  - Use `--no-sandbox` flags via `-p ./tmp/puppeteer-config.json`.
- Use a system browser instead of downloading:
  - Set `PUPPETEER_EXECUTABLE_PATH` to the Chrome/Chromium binary or put `executablePath` in your puppeteer config.

## References
- Mermaid CLI: https://github.com/mermaid-js/mermaid-cli
- Puppeteer troubleshooting: https://pptr.dev/troubleshooting
- Local repo notes: `docs/mermaid-cli-setup.md`
