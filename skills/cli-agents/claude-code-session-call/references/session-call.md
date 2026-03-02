# Claude Code session calling (resume)

Goal: run a prompt in an existing Claude Code conversation deterministically using `--resume` with a known `session_id`.

## Resolve session name/session alias to `session_id`

Sessions created via `$claude-code-create-session` can be referenced by a user-facing session name/session alias stored in a workspace-scoped mapping file:

- Default path: `<system-tmp>/agent-sessions/<basename-of-workspace>-<md5-hex-string-of-abs-path-of-workspace>/claude-code-alias-mapping.json`
- Shape: top-level `workspace_dir` plus `aliases` mapping alias->{`session_id`, `created_at`}
- Back-compat: older files may use a top-level alias map

## Resume (JSON output)

```bash
claude -p "Continue from where we left off" --resume "<session_id>" --output-format json
```

Prefer `--resume` over `--continue` in automation because it is explicit and deterministic.
