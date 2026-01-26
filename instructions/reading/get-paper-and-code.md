# How to Get Paper and Source Code

This guide outlines the process for setting up a new research paper for analysis. We store the paper's documentation/source in `[PAPER_DOC_DIR]` (defaults to `paper-source/`) and its implementation code in `extern/orphan/`.

## Required Information

Before starting, gather the following details:
- **Paper Codename**: A short, lowercase identifier (e.g., `vibetensor`).
- **ArXiv ID**: The numeric identifier from the URL (e.g., `2601.16238`).
- **GitHub URL**: The link to the official implementation repository.

## Setup Procedure

### 1. Document the Paper (in `[PAPER_DOC_DIR]`)

For each paper, create a subdirectory named after the **Paper Codename**. This directory should manage the paper's LaTeX source and PDF version.

- **Automated Download (`bootstrap.sh`)**: Create a script that downloads the PDF from ArXiv and extracts the LaTeX source (from the ArXiv e-print URL) into a local `tex/` directory. It should use a temporary system directory for the download and cleanup afterwards.
- **Git Hygiene (`.gitignore`)**: Ensure that the downloaded PDF and the `tex/` directory are ignored by Git. These are heavy artifacts that should be fetched via the bootstrap script, not stored in the repository.
- **Documentation (`README.md`)**: Include the full title of the paper, the ArXiv ID, and instructions on how to run the bootstrap script.

### 2. Reference the Implementation (in `extern/orphan/`)

We keep the paper's source code available for reference without integrating it into our primary codebase.

- **Clone for Reference**: Clone the GitHub repository into `extern/orphan/<paper-codename>`.
- **Optimization**: Use a shallow clone (`--depth 1`) since the code is for reference only and we do not need the full commit history.
- **Git Hygiene**: Add the new directory name to `extern/orphan/.gitignore`. The contents of `extern/` are meant to be external to our project's version control.

## Organizational Structure

The resulting structure should look like this:

```text
/
├── extern/
│   └── orphan/
│       ├── .gitignore  <-- Add <paper-codename> here
│       └── <paper-codename>/  <-- Cloned with --depth 1
└── [PAPER_DOC_DIR]/
    └── <paper-codename>/
        ├── .gitignore  <-- Ignores PDF and tex/
        ├── README.md
        ├── bootstrap.sh
        ├── <paper-codename>.pdf  <-- Fetched by bootstrap
        └── tex/  <-- Extracted by bootstrap
```
