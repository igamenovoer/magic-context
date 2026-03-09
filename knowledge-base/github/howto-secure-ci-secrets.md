# How to Configure Secret-Backed CI with Permission Control in GitHub Actions

When maintaining a repository where contributors have "Write" access or can submit Pull Requests from forks, you must carefully manage secrets. Normal repository secrets can be exfiltrated if a user modifies the CI pipeline to print or send the secret over the network. To protect owner-only secrets while still allowing contributors to run tests with shared credentials, you must combine Branch Protection, Repository Secrets, and Environment Secrets.

## Securing the Main Branch

You must ensure no one but the repository owner (or trusted maintainers) can alter the code or workflows running on the protected branch.

1. Go to **Settings** > **Branches** > **Add branch protection rule**.
2. Set the branch name pattern to `main` (or your default branch).
3. Check **Require a pull request before merging** and **Require approvals**.
4. Check **Do not allow bypassing the above settings** to enforce this for all contributors.

## Configuring "Owner-Only" Secrets using Environments

Environments allow you to restrict secrets to specific branches or require manual approval, preventing contributors from accessing them via unauthorized workflow changes.

1. Go to **Settings** > **Environments** > **New environment** and create one named `owner-restricted` (or similar).
2. Under **Deployment branches and tags**, configure it to only allow the `main` branch. Alternatively, add yourself under **Required reviewers** so the CI pauses for manual approval.
3. Add your highly sensitive credentials under **Environment secrets** (e.g., `PROD_API_KEY`).

Because the environment is restricted to the `main` branch, any malicious script pushed to a feature branch that attempts to use the `owner-restricted` environment will be blocked by GitHub.

## Configuring "Shared" Secrets for Contributors

For credentials that contributors need for testing but should remain hidden from the broad public (e.g., read-only users or forks), use standard Repository Secrets.

1. Go to **Settings** > **Secrets and variables** > **Actions**.
2. Add your shared credentials under **Repository secrets** (e.g., `STAGING_DB_PASSWORD`).

Contributors pushing to feature branches can run CI that uses these shared secrets. However, if a public user forks the repo and submits a PR, GitHub completely blocks fork PRs from accessing any secrets by default.

## Example Workflow Configuration

Split your CI into two separate workflows based on the required privilege level.

### Contributor Testing CI (Shared Secrets)

This workflow runs on all branches and PRs, utilizing the shared repository secrets.

```yaml
name: Shared Contributor CI
on:
  push:
    branches: [ "**" ]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run Tests with Shared Secrets
        env:
          DB_PASS: ${{ secrets.STAGING_DB_PASSWORD }} # Pulls from Repository Secrets
        run: ./run-tests.sh --password "$DB_PASS"
```

### Owner-Only CI (Environment Secrets)

This workflow is restricted to the main branch and uses the protected environment to access sensitive owner-only secrets.

```yaml
name: Owner Protected CI
on:
  push:
    branches: [ "main" ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: owner-restricted # Critical: Fetches secrets from the protected environment
    steps:
      - name: Deploy / Run Secure Task
        env:
          API_KEY: ${{ secrets.PROD_API_KEY }} # Pulls from Environment Secrets
        run: ./secure-task.sh --key "$API_KEY"
```

## Source Links

- [GitHub Actions: Using environments for deployment](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
- [GitHub Actions: Using secrets in GitHub Actions](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions)
- [GitHub Actions: Managing branch protection rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule)
