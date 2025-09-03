you are tasked to write or modify source code, follow these guidelines

# Guidelines for Writing or Modifying Source Code
- ALWAYS use `context7` to find documentations if you encouter problems calling 3rd python library, or if you think needed or in doubt
- use online search to find best practices or solutions if needed or in doubt
- **DO NOT use unicode emojis in your console output like print statements, as it may cause issues in some environments**
- you CAN use unicode emojis in GUI code, like web-based applications, as they are generally well supported in modern browsers
- NEVER modify pyproject.toml directly due to missing packages, use pixi installation commands instead

# Python Code Guidelines

- Avoid using relative imports in Python code, prefer to use absolute imports to ensure clarity and avoid import errors.
- place temporary scripts and data in `<workspace>/tmp` directory, better create subdirs for different purposes, do not place them in the main source code directories.
- when creating python stubs, throw `NotImplementedError` for methods that are not implemented yet, and use `pass` for methods that are placeholders or do nothing.

## CRITICAL: Strongly Typed Code Requirements

your python code should be written in a strongly typed manner, and use `mypy` to validate the types in your code after editing. For `pixi`, use `pixi run -e dev mypy <your file or dir>` to run mypy.

- if you are not sure about a type, use `Any` as a fallback
- if you are unsure about a type in a 3rd party library, use `context7` MCP to find out info before you proceed
- if `context7` MCP returns something inconsistent with the code, try to create a minimal example code to directly inspect the type

# How to run code

If you want to run something, you should follow these instructions.

**Watch out for pixi usage**
Look for `pyproject.toml` or `pixi.lock` in the root directory of the project. If you find them, it means the project uses `pixi` as the package manager. Then, you should 
- run the code using `pixi run -e dev <your command>` for development tasks, unless specified otherwise
- `pixi run <your command>` for deployment tasks.

**Avoid inline python code**
If you want to run a small snippet of code, you can create a temporary script in the `tmp` directory under the root of the workspace, and run it using `pixi run -e dev python <your temp script>`, or other environments specified by user. You can reuse the same script (overwrite it) for different tasks if needed, all files in `tmp` directory are considered temporary files and can be deleted at any time.

# CLI tools

To run CLI commands, pay attention to the following:
- for any interactive process that may block the terminal, timeout within 10 seconds
- for anything you need to wait for timeout, timeout in LESS THAN 15 seconds

These tools are usually available, you shall try to use them if needed:
- `jq` for json processing
- `curl` for http requests
- `yq` for yaml processing

# Temporary Files

Any temporary files you create should be placed in the `tmp` directory under the root of the workspace, unless otherwise specified. 
