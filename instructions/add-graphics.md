you are tasked to add graphics to the given documents, with the following guidelines.


# Guidelines for Graphics Insertion

- use `mermaid`, `dot` or `plantuml` to generate graphics
- for UML diagram, prefer `mermaid` over `plantuml` over `dot`, `mermaid` has UML support
- if in doubt or encounter errors, consult `context7` about how the markup language works
- if external files are generated (like .svg), the graphics should appear in the source document (via reference, never copy graphics content like svg content directly into document), and add a link to the graphics source code (like .puml, .dot)

## Using `mermaid`
- mermaid scripts can be stored directly within markdown document, no need to compile it
- make sure text within nodes is not too long to be truncated. If it is too long, break them into separate lines.

## Using `plantuml`
- do not embed plantuml scripts inside markdown, just create svg alongside
- the generated files (`.puml` and `.svg`) should be stored in the same directory as the source files, in a subdir named after the given documentation (excluding extension), UNLESS specified otherwise
- try `plantumlc` first, if failed, try online graph generation via `https://www.plantuml.com/plantuml/svg/`, like this:

```bash
curl -s -H "Content-Type: text/plain" --data-binary @diagram.puml \
  https://www.plantuml.com/plantuml/svg/ -o diagram.svg
```
- do not try to install `plantuml` locally, it is too complicated

## Using `dot`
- the generated graphics should be in SVG format if possible, or PNG if SVG is not supported
- the generated files (`.puml` and `.svg`) should be stored in the same directory as the source files, in a subdir named after the given documentation (excluding extension), UNLESS specified otherwise