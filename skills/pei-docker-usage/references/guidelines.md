# General Guidelines

These guidelines ensure you use `pei-docker-cli` safely and effectively, avoiding common pitfalls like data loss or configuration drift.

## 1. Custom Logic Placement

**Rule:** Always place your custom scripts and assets in `installation/stage-1/custom/` or `installation/stage-2/custom/`.

*   **Avoid:** Modifying files in `installation/stage-X/system/`. These directories contain internal scripts managed by PeiDocker and may be overwritten or cause conflicts if the tool updates.
*   **Why:** This keeps your custom logic isolated from the framework's internal logic, making upgrades and maintenance significantly easier.

## 2. Configuration Persistence (`user_config.persist.yml`)

**Rule:** Create and maintain a `user_config.persist.yml` file in your project directory.

*   **Workflow:**
    1.  Create `user_config.persist.yml` for your permanent configuration.
    2.  Copy it to `user_config.yml` when you need to run `configure`:
        ```bash
        cp user_config.persist.yml user_config.yml
        pixi run pei-docker-cli configure
        ```
*   **Why:** The `pei-docker-cli create` subcommand (and potentially others) resets `user_config.yml` to a default template. If you only edit `user_config.yml`, you risk losing all your work if you accidentally re-run `create`. Treating `user_config.yml` as a transient build artifact prevents this.

## 3. Extending Beyond PeiDocker

**Rule:** If PeiDocker lacks a specific feature, modify the generated `docker-compose.yml` manually, but document your changes.

*   **Context:** `pei-docker-cli` covers common use cases (SSH, GPU, Volumes) but not every possible Docker feature.
*   **Workflow:**
    1.  Run `pixi run pei-docker-cli configure` to generate the base `docker-compose.yml`.
    2.  Manually edit `docker-compose.yml` to add advanced features (e.g., complex networks, specific logging drivers).
    3.  **Crucial:** Create a `compose-diff.md` (or similar) to record exactly what you changed.
*   **Why:** Re-running `configure` will overwrite `docker-compose.yml`. Your notes in `compose-diff.md` will allow you to quickly re-apply your manual changes after a configuration update.

## 4. Storage Strategy: Storage vs. Mount

PeiDocker offers two ways to manage persistent data in Stage-2: the `storage` system and direct `mount`s.

### The `storage` System (Recommended for Core Data)
Maps three standardized paths (`/soft/app`, `/soft/data`, `/soft/workspace`) to flexible backends.

*   **Use when:** You want the flexibility to switch between host binds (development), Docker volumes (production/local), or in-image storage (distribution) without changing your application code or paths.
*   **Advantage:** Abstraction. Your app always writes to `/soft/data`, regardless of where that actually lives physically.

### Direct `mount` (Recommended for Specific Integrations)
Maps arbitrary host paths or volumes to arbitrary container paths.

**Warning:** Do not name your mount `workspace`. This key is reserved for the `storage` system.

*   **Use when:**
    *   You need to mount configuration files to specific system locations (e.g., `/etc/myapp/config.toml`).
    *   You are mounting user credentials (e.g., `~/.ssh`, `~/.aws`).
    *   You have legacy paths that cannot be moved to `/soft/`.
*   **Advantage:** Precision. You control exactly what goes where.

### Decision Heuristic
**If unsure:** Ask yourself, "Might I want to bake this content into the final Docker image later?"
*   **Yes:** Use `storage`. It allows you to switch `type: host` to `type: image` seamlessly.
*   **No:** Use direct `mount`.
*   **Still unsure:** Use direct `mount`. It's simpler and more explicit.

## 5. Proxy Handling: Avoid Baking in Images

**Rule:** Avoid baking proxy settings into the Docker image unless absolutely necessary for the runtime environment in a specific network.

*   **Use Case:** You need a proxy to *build* the image (e.g., download packages), but the container shouldn't depend on that proxy when running on another machine.
*   **Configuration:**
    ```yaml
    proxy:
      address: host.docker.internal
      port: 7890
      enable_globally: false      # Don't set global env vars automatically
      remove_after_build: true    # CRITICAL: Clean up proxy vars after build
    apt:
      use_proxy: true             # Use proxy ONLY for apt during build
      keep_proxy_after_build: false # Clean up apt proxy config
    ```
*   **Why:** Hardcoded proxy settings make images non-portable. If you share the image with someone on a different network, it will fail to connect.

## 6. Development Persistence: Mount User Home

**Rule:** For development images, mount a volume or host directory to the user's home directory (e.g., `/home/developer`) in `stage_2`.

*   **Configuration:**
    ```yaml
    stage_2:
      mount:
        home_developer:
          type: auto-volume  # Or 'host' for direct access
          dst_path: /home/developer
    ```
*   **Why:** This persists user-specific settings (like `.bashrc`, `.gitconfig`, shell history, VS Code server data) independently of the container lifecycle. You avoid the need to "commit" the image just to save a shell alias or git setting.
