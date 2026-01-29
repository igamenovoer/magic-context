import os
import sys
import argparse
from pathlib import Path

def generate_readme_content(dir_path, description):
    """Generates README content including existing files."""
    content = [description.strip(), ""]
    
    try:
        # List items, excluding the README we are about to create
        items = sorted([p for p in dir_path.iterdir() if p.name != "README.md"])
        if items:
            content.append("\n## Existing Content")
            content.append("The following files and directories were present when this README was generated:\n")
            for item in items:
                type_str = "DIR" if item.is_dir() else "FILE"
                content.append(f"- `{item.name}` ({type_str})")
    except Exception as e:
        content.append(f"\n(Could not list existing content: {e})")
        
    return "\n".join(content) + "\n"

def create_file(path, content=""):
    path = Path(path)
    # Ensure parent exists
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        
    if not path.exists():
        with open(path, "w") as f:
            f.write(content)
        print(f"Created file: {path}")
    else:
        print(f"Skipped (exists): {path}")

def create_dir_with_readme(path, description):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created dir:  {path}")
    else:
        print(f"Dir exists:   {path}")
    
    readme_path = path / "README.md"
    if not readme_path.exists():
        content = generate_readme_content(path, description)
        create_file(readme_path, content)
    else:
        print(f"Skipped (README exists): {readme_path}")

def main():
    parser = argparse.ArgumentParser(description="Scaffold Python Project Structure")
    parser.add_argument("project_root", help="Root directory of the project")
    parser.add_argument("--package-name", help="Name of the python package (default: project_root name)", default=None)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    project_name = root.name
    package_name = args.package_name if args.package_name else project_name.replace("-", "_").lower()

    print(f"Scaffolding project '{project_name}' in '{root}'...")

    # Define directories and their descriptions
    # List of tuples: (relative_path, description)
    # Ordered to ensure parents are processed before children (though mkdir -p handles this, standard order is nice)
    directories = [
        (".github", "# GitHub Configuration\n\nConfiguration for GitHub features."),
        (".github/workflows", "# GitHub Actions Workflows\n\nAutomated CI/CD pipelines go here.\n- Documentation deployment\n- Automated testing\n- Package building"),
        ("src", "# Source Code\n\nContainer for the main package code (src layout)."),
        (f"src/{package_name}", f"# {package_name}\n\nMain package code.\n- Contains core library functionality.\n- Includes package initialization."),
        ("extern", "# External Dependencies\n\nThird-party code."),
        ("extern/tracked", "# Tracked External Dependencies\n\nGit submodules pinned via .gitmodules.\nUse for dependencies you want reproducible and pinned."),
        ("extern/orphan", "# Orphaned External Dependencies\n\nLocal clones/checkouts (git-ignored).\nUse for local-only clones you do not want to commit."),
        ("scripts", "# Scripts\n\nCommand-line interface tools.\nEntry points for terminal commands."),
        ("tests", "# Tests\n\nComprehensive testing for all functionality."),
        ("tests/unit", "# Unit Tests\n\nFast, deterministic unit tests.\nMirrors source structure by module/functionality."),
        ("tests/integration", "# Integration Tests\n\nI/O, service, or multi-component tests.\nExercises external services or filesystem I/O."),
        ("tests/manual", "# Manual Tests\n\nManually executed scripts (not CI-collected).\nFor heavy or environment-specific checks."),
        ("docs", "# Documentation\n\nDocumentation source (Markdown).\nBuilt with MkDocs Material."),
        ("context", "# AI Context\n\nAI Assistant Workspace.\nContains project design documents, implementation plans, and reference materials. See [context-dir-guide.md](context-dir-guide.md)."),
        ("context/archived", "# Archived Context\n\nOld plans, historical task breakdowns, and superseded docs."),
        ("tmp", "# Temporary Files\n\nTemporary working files."),
    ]

    for path_str, desc in directories:
        full_path = root / path_str
        create_dir_with_readme(full_path, desc)

    # Specific non-README files
    files = [
        (f"src/{package_name}/__init__.py", f'"""Top-level package for {project_name}."""\n'),
        ("extern/.gitignore", "orphan/*\n!orphan/README.md\n"),
    ]

    for path_str, content in files:
        create_file(root / path_str, content)

    print("\nProject structure created successfully.")

if __name__ == "__main__":
    main()