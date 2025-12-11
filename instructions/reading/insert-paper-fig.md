you are tasked to extract one or more figures from a source paper and insert them into this repository’s paper-reading notes. Follow these steps and conventions so the process works for any paper.

---

## 0. Environment & Tools

- **Check for Pixi env first**:
  - Prefer running Python and related tools via `pixi run ...` so you use the project’s managed environment.
  - Example: `pixi run python - << 'PY' ... PY`
- **Check for `pymupdf` availability**:
  - If the repo already declares `pymupdf` (or `fitz`) as a dependency (e.g., in `pixi.toml`, a Pixi-supported `pyproject.toml`, or the current Pixi environment), you may use it directly.
  - If you need `pymupdf` and it is **not** installed / importable, do **not** silently assume it exists:
    - Explain to the user that you need `pymupdf` (PyMuPDF) to render/convert PDF pages to images.
    - Ask the user to install it (e.g., by adding it to `pixi.toml` or running `pixi add pymupdf` / equivalent) before you proceed with PDF-to-image conversions.

---

## 1. Figure Selection & Context

- Identify which figure(s) to extract based on the user request or your own note structure:
  - Use the LaTeX label/ID when available (e.g., `\label{fig:patterns}` → `{FIG_ID}`).
  - Keep the original figure numbering and caption from the paper.
- Locate the figure’s source:
  - **Preferred**: Vector/PDF assets from LaTeX (e.g., `paper-source/.../resources/{FIG_FILE}.pdf`).
  - **Fallback**: Direct extraction from the compiled PDF (e.g., `paper-source/{PAPER_NAME}.pdf`) using a page snapshot.

---

## 2. Output Location & Naming

- Notes for a paper live under:
  - `notes/{PAPER_NAME}/main-note.md`
- All figures for that paper must be stored under:
  - `notes/{PAPER_NAME}/figures/`
- Naming convention (generic, but descriptive):
  - `{FIG_ID}.png` if a meaningful label exists, or
  - `{PAPER_NAME}_figure_{N}.png` if you only know the figure index.
- Paths in markdown must be **relative to `main-note.md`**, e.g.:
  - `![Alt text](figures/{FIG_ID}.png)`

---

## 3. Extraction Workflow

### 3.1 From a LaTeX resources directory (preferred)

When a figure is already available as its own PDF/PNG/SVG:

1. Locate the file referenced by `\includegraphics` in LaTeX:
   - Example: `\includegraphics{resources/{FIG_FILE}.pdf}`
2. If it is already a bitmap (PNG/JPEG), copy it into the note’s figures directory:
   - `notes/{PAPER_NAME}/figures/{TARGET_NAME}.png`
3. If it is a vector PDF, convert it to PNG at a reasonable DPI (e.g., 150–300):
   - Recommended: use Python + PyMuPDF (or another standard tool in this repo):
     ```bash
     pixi run python - << 'PY'
     import fitz
     from pathlib import Path

     src = Path("{SRC_PDF_PATH}")   # e.g., paper-source/.../resources/{FIG_FILE}.pdf
     out = Path("notes/{PAPER_NAME}/figures/{TARGET_NAME}.png")

     with fitz.open(src) as doc:
         page = doc[0]
         pix = page.get_pixmap(dpi=200)
         out.parent.mkdir(parents=True, exist_ok=True)
         pix.save(out)
     PY
     ```
   - Adjust DPI if the image looks too blurry or too large.

### 3.2 From the compiled PDF (fallback)

If there is no standalone figure file and you only have the full paper PDF:

1. Identify the page index for `{FIG_ID}` in `{PAPER_PDF}` (e.g., by reading the PDF or using a PDF viewer).
2. Render that page (or a tight crop around the figure) to PNG:
   ```bash
   pixi run python - << 'PY'
   import fitz
   from pathlib import Path

   pdf = Path("{PAPER_PDF}")  # e.g., paper-source/{PAPER_NAME}.pdf
   out = Path("notes/{PAPER_NAME}/figures/{TARGET_NAME}.png")
   page_index = {PAGE_INDEX}  # zero-based

   with fitz.open(pdf) as doc:
       page = doc[page_index]
       pix = page.get_pixmap(dpi=200)
       out.parent.mkdir(parents=True, exist_ok=True)
       pix.save(out)
   PY
   ```
3. (Optional) If cropping is needed, use PyMuPDF’s `page.get_pixmap(clip=...)` or an image tool to crop to just the figure.

---

## 4. Markdown Insertion Pattern

For each inserted figure, follow this pattern in `notes/{PAPER_NAME}/main-note.md`:

```markdown
![{Short descriptive title for the figure}](figures/{TARGET_NAME}.png)  
> "{Exact or lightly edited caption from the paper. Keep important details about what is being visualized and any key interpretation hints.}" (Figure {N}; {SOURCE_PATH_IN_REPO_OR_PAPER})
```

Guidelines:
- Use a short but informative alt text as the first string inside `![...]`.
- In the blockquote, include:
  - The original caption text (possibly mildly shortened if extremely long).
  - A reference to where it came from:
    - `Figure {N}; paper-source/.../tex/{FILE}.tex` if from LaTeX.
    - `Figure {N}; {PAPER_PDF}` if from a compiled PDF.

Place the figure near the section that discusses the corresponding content (e.g., after the bullets that explain the figure’s concepts).

---

## 5. Placeholders & Reuse

When using this prompt for a new paper, substitute:

- `{PAPER_NAME}`: short handle for the paper, matching the note directory name.
- `{FIG_ID}`: LaTeX label or a short descriptive ID for the figure (e.g., `weight_sensitivities`).
- `{TARGET_NAME}`: final image filename under `figures/` (often derived from `{FIG_ID}`).
- `{SRC_PDF_PATH}`: path to the source figure PDF within the paper’s LaTeX resources.
- `{PAPER_PDF}`: path to the compiled paper PDF, if needed.
- `{PAGE_INDEX}`: zero-based page index of the figure in the PDF.

Always keep:
- Figures under `notes/{PAPER_NAME}/figures/`.
- Paths in markdown relative to `main-note.md`.
- Captions grounded in the original paper text.
