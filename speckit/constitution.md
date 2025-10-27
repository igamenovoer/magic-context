# Project Principles and Constitution

## Coding Qualities
- we prefer reusable, modular components, object-oriented design, and separation of concerns
- we value clarity and maintainability over cleverness
- our code should be well-documented and easy to understand for new contributors
- we emphasize that code should have clear contracts, with well-defined inputs, outputs and assumptions

## Documentation Standards
- all public functions and classes must have clear docstrings, following numpy doc style (python), and doxygen style (C++)
- created markdowns should be readable and well-structured, using proper headings and formatting, prefer to have section numbers (x.y.z)
- coding examples should be included in the documentation where applicable

## Testing Requirements
- major functionality should have manual testing scripts, expected to be run by developer in interactive environment, like Jupyter Notebook, console, etc, these manual test scripts should not use testing frameworks, end-to-end in global space (avoid wrapping code into functions, etc)
- visualization is very important in testing, human programmer should be able to see the results clearly to verify correctness
- automated tests can be sparse, integration tests are preferred over unit tests
