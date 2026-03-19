# Creation stage (persist a new Codex CLI session)

Goal: create a new Codex CLI session, capture its `thread_id`, and persist a user-facing session name for later deterministic resumes.

## Required input (don't invent)

- `session name`

If missing, ask the user for it.

## Optional input (only if user provides it)

- Role definition Markdown (`.md`) file path
  Codex CLI has no `--append-system-prompt`, so the helper prepends the file contents to the initial prompt text.

## Create a new session (headless)

Create a new session and print a JSON summary that includes the persisted mapping path:

```bash
python3 scripts/invoke_persist.py create-session --session-name "review-src" --print-mapping-json
```

If the user provides a role definition Markdown file, prepend it to the initial prompt:

```bash
python3 scripts/invoke_persist.py create-session --session-name "review-src" --role-definition-md "/abs/path/to/role.md" --print-mapping-json
```

Optional: if the user explicitly chooses a model, pass it at creation time so it becomes the stored default for later resumes:

```bash
python3 scripts/invoke_persist.py create-session --session-name "review-src" --model "MODEL" --print-mapping-json
```

Credentials: ensure the `codex` CLI is on PATH and authenticated. To load env vars from a file, pass `--env-file /path/to/vars.env` (KEY=VALUE lines).

If the user provides a codex-compatible executable wrapper command or script, pass it via `--codex-cmd` (or set `CODEX_CMD`):

```bash
python3 scripts/invoke_persist.py create-session --session-name "review-src" --print-mapping-json --codex-cmd "codex-wrapper"
```

If the user provides a POSIX alias/function wrapper, use shell-launch mode:

```bash
python3 scripts/invoke_persist.py create-session \
  --session-name "review-src" \
  --print-mapping-json \
  --codex-cmd 'codex-work' \
  --codex-shell posix \
  --codex-shell-init 'shopt -s expand_aliases; source ~/.bashrc' \
  --codex-shell-cmd 'bash -lc'
```

If the user provides a PowerShell function wrapper, use PowerShell shell-launch mode:

```bash
python3 scripts/invoke_persist.py create-session \
  --session-name "review-src" \
  --print-mapping-json \
  --codex-cmd 'codex-work' \
  --codex-shell powershell \
  --codex-shell-init '. $PROFILE' \
  --codex-shell-cmd 'pwsh -NoLogo -Command'
```

Do not assume wrapper names start with `codex-`. Use the exact wrapper name or shell snippet the user gave you.

Additional Codex exec flags supported by the helper at creation time:

```bash
python3 scripts/invoke_persist.py create-session \
  --session-name "review-src" \
  --config 'model_provider="oss"' \
  --enable my_feature \
  --image /abs/path/to/image.png \
  --full-auto
```

Do not set deadlines unless the user explicitly requested a time limit. If they did, pass `--deadline-seconds`:

```bash
python3 scripts/invoke_persist.py create-session --session-name "review-src" --print-mapping-json --deadline-seconds 30
```

Do not interrupt a running create/resume Codex call just because intermediate output appears to drift from the prompt. Let the process finish unless deadline, stall, or hard error criteria are met.

## Persist the alias mapping (system temp)

By default, this skill writes a workspace-scoped JSON file under system temp:

- Path: `<system-tmp>/agent-sessions/<workspace-basename>-<md5(abs-workspace-dir)>/codex-cli-alias-mapping.json`
- Shape: top-level `workspace_dir` plus `aliases` mapping session_name->{`thread_id`, `created_at`, optional `last_model`}
- Overwrite behavior: overwrite alias entries unconditionally

The default path depends on your current working directory. Run from the same workspace directory across create/resume, or pass `--mapping-file` explicitly.

If writing the mapping file fails, the script warns to stderr but still succeeds and prints the `thread_id`. Keep the alias and `thread_id` in the current chat context so you can resume within this conversation.

## Output contract (what to respond with)

When the creation stage runs, respond with:

- The created `thread_id` and the session name it was stored under.
- The mapping file path used and whether it was written successfully.
- The updated alias mapping JSON block from `--print-mapping-json`.
