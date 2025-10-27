you are tasked to analyze and document the architecture of a given set of source code, follow these guidelines to document:

# Architecture Documentation Guidelines

## Principles

- first identify the known pattern of the architecture, such as MVC, MVVM, or any other design pattern that is used widely in public projects, if exists, or if specified explictly in the context, name this `arch-pattern`
- create diagram to illustrate the architecture, if the project is based on known frameworks or libraries, use `context7` to find the official documentation of the framework or library, it usually suggests the best practices and patterns to follow
- create digrams use `mermaid` UML diagram, just include the source code of the diagrams in the markdown file, tools use be able the preview this. For `mermaid` diagrams, make sure you do not have long texts in blocks, break them into multiple lines if necessary.
- if `context7` mcp is available, use it to find out latest `mermaid` syntax and examples first
- output the documentations to `context\design` (UNLESS specified otherwise by user), name it `arch-(pick a name).md`, and all diagrams should be placed in a subdirectory named after the markdown file.

### Must Include Diagrams

- **Package Diagram**: This UML diagram should show the high-level structure of the project, including the main packages and their relationships.

- **Class Diagram**: This UML diagram should show the main classes in the project, their attributes, methods, and relationships.

- **(for GUI Project) Sequence Diagram**: This UML diagram should show the flow of control in a specific use case or scenario, illustrating how objects interact over time.

## Appendix: Mermaid Syntax Notes

- Prefer HTML line breaks (`<br/>`) inside labels; raw `\n` is not supported in modern Mermaid versions.
- Quote participant or node labels that contain spaces or punctuation: `participant API as "Router Service"`.
- Avoid non-ASCII identifiers; use transliterations (e.g., `TianshuRouter`) for class names.
- Keep statements one per line and ensure diagram type keywords (`graph`, `classDiagram`, `sequenceDiagram`) are lowercase.
- Validate diagrams with the Mermaid Live Editor when possible before finalizing docs.
