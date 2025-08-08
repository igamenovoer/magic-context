you are tasked to add graphics to the given documents, with the following guidelines.


# Guidelines for Graphics Insertion

- use `dot` or `plantuml` to generate graphics
- if in doubt or encounter errors, consult `context7` about how the markup language works
- the generated graphics should be in SVG format if possible, or PNG if SVG is not supported
- the generated files should be stored in the same directory as the source files, in a subdir named after the given documentation (excluding extension), UNLESS specified otherwise
- after generated, the graphics should appear in the source document, replacing the existing graphics, and add a link to the graphics source code (like `dot` or `plantuml` source code), the source code should be in the same directory as the generated graphics. After linking to external source code, remove the source code from the original document.
- DO NOT use inline SVG, the svg should be external file