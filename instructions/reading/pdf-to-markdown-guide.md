you are tasked to convert a PDF file into a Markdown note, extracting figures and inserting them into the Markdown. Follow these steps and defaults unless the user specifies otherwise.

---

## 0. Environment & Tool Priority

1) **Look for a Pixi env first**
- Prefer running Python and related tools via `pixi run ...` to use the repo’s managed environment.
- Quick check:
  - `pixi run python -c "import sys; print(sys.version)"`

2) **If Pixi exists, check for PyMuPDF**
- PyMuPDF imports as `fitz`:
  - `pixi run python -c "import fitz; print('pymupdf ok')"`
- If PyMuPDF is usable, prefer a **file-specific conversion script** (PyMuPDF-driven) over “one-size-fits-all” tools like `markitdown` or `pandoc`.

3) **Temporary scripts and scratch outputs**
- Save one-off scripts in `<workspace>/tmp/<task-name>/` (preferred) or use system temp via Python’s `tempfile`.
- Keep the workspace clean: treat these as disposable helpers unless the user asks to keep them.

---

## 1. Decide Strategy (Deterministic)

Use this decision order:

1) **PyMuPDF usable (preferred)**:
- Use PyMuPDF to extract text blocks and render figure/table regions to images, then generate Markdown with relative image links.

2) **PyMuPDF usable + assisting tools available**:
- Also check for `markitdown`, `pandoc`, and a LaTeX toolchain, and use them as *assisting tools* (e.g., cross-check text, handle special cases, or leverage LaTeX sources if present).

3) **PyMuPDF NOT usable (fallback)**:
- Fall back to using `markitdown` / `pandoc` / LaTeX-based approaches directly (pick what’s available and most reliable for the given PDF).

Notes:
- `markitdown` is often the simplest fallback when it supports the PDF and its dependencies are present.
- `pandoc` and LaTeX are most useful when you have a convertible intermediate (e.g., LaTeX source, HTML, DOCX) rather than raw PDF alone.

---

## 2. Output Defaults (IMPORTANT)

Unless the user specifies otherwise:

- **`<output-dir>`**: the same directory as the input PDF file.
- **Markdown output path**: `<output-dir>/<pdf-stem>.md` (same name, different extension).
- **Figures directory**: `<output-dir>/figures/`.
- **Image references in Markdown**: use relative paths from the `.md` file, e.g. `![Caption](figures/figure-1.png)`.

---

## 3. PyMuPDF-First Workflow (Recommended)

### 3.1 Inspect the PDF
- Determine page count and metadata (`fitz.open(...).page_count`, `doc.metadata`).
- Render a few pages to confirm readability and whether figures are rasterized or vector-like.

### 3.2 Extract text with layout awareness
- Use `page.get_text("dict")` and iterate blocks/lines/spans.
- Reconstruct reading order (single-column vs two-column) and filter headers/footers.
- Preserve headings where possible; otherwise keep plain paragraphs with blank lines between blocks.

### 3.3 Detect and extract figures/tables
- Prefer caption-driven extraction:
  - Detect caption lines like `Figure N: ...`, `Fig. N: ...`, `Table N: ...`.
  - Compute a clip rectangle for the region above the caption (bounded within the page / column).
  - Render via `page.get_pixmap(clip=clip, dpi=...)` and save to `<output-dir>/figures/`.
- Keep filenames stable and predictable:
  - `figure-1.png`, `figure-2.png`, ...
  - `table-1.png`, `table-2.png`, ...

### 3.4 Generate Markdown
- Insert extracted images near the caption location:
  - `![Figure N: <caption>](figures/figure-N.png)`
- Keep separators if useful (optional but consistent):
  - `<!-- Page K -->` and `---` between pages.

### 3.5 Validate
- Ensure every referenced image exists on disk.
- Ensure paths are relative and correct from the Markdown file location.

---

## 4. Tooling Checks (Assisting / Fallback)

When needed, check availability (use `pixi run ...` when in a Pixi repo):

- **`markitdown`**:
  - `pixi run markitdown --version` (or `pixi run python -c "import markitdown"`)
- **`pandoc`**:
  - `pandoc --version`
- **LaTeX toolchain** (for LaTeX sources / rebuild workflows):
  - `pdflatex --version`

If PyMuPDF is usable, use these tools only to assist (e.g., confirm headings, recover text from hard pages, or leverage LaTeX sources).
If PyMuPDF is not usable, choose the most direct tool-based conversion path available and still follow the output defaults in section 2.

