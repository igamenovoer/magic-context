you are tasked to extract one or more **algorithm blocks** (e.g., `algorithm` or `algorithm2e` environments) from a LaTeX paper and turn them into reusable figures (SVG/PNG) for inclusion in reading notes. The goal is to preserve the original formatting as closely as possible while removing page noise (page numbers, headers/footers, excess whitespace).

---

## 0. Environment & Tools

- **LaTeX toolchain**:
  - Prefer using the system’s LaTeX tools: `pdflatex`, `pdfcrop`, and `pdftocairo` or `dvisvgm`.
  - Before using them, check that the binaries exist:
    - `which pdflatex`, `which pdfcrop`, `which pdftocairo`, `which dvisvgm`.
  - If a required tool is missing, do **not** guess: explain which binary is needed and ask the user whether they can install it.

- **Python & pixi (optional)**:
  - For any auxiliary scripting (e.g., inspecting directories, small conversions), prefer `pixi run python ...` if a Pixi environment is configured.
  - Only use Python when it clearly simplifies the task; the core pipeline should stay LaTeX-based.

---

## 1. Locate the Algorithm in the Source

Given a LaTeX paper under something like `paper-source/primary/tex/`:

- **Find the defining environment**:
  - Search for `\begin{algorithm}` / `\end{algorithm}`, or `\begin{algorithm2e}` / `\end{algorithm2e}`, etc.
  - Also search for the caption text or label: e.g. `\caption{...}` or `\label{alg:...}`.
  - Record:
    - `{ALGO_ENV}`: the environment name (`algorithm`, `algorithm2e`, etc.).
    - `{ALGO_LABEL}`: the LaTeX label (e.g. `alg:main`), if present.
    - `{ALGO_FILE}`: the `.tex` file that contains the environment (e.g. `method.tex`).

- **Identify the paper’s main class/style**:
  - Open the main TeX file (e.g. `main.tex`) and note:
    - `\documentclass[...] {<class>}` (e.g. `article`, `neurips_2023`, `iclr2024_conference`).
    - Any style-specific packages (e.g. `\usepackage[preprint]{neurips_2023}`).
  - You’ll need these to reproduce the same typography for the algorithm block.

---

## 2. Build a Minimal Standalone LaTeX Document

Create a dedicated TeX file under a temporary directory (e.g. `tmp/{PAPER_NAME}_alg/{ALGO_ID}.tex`) that contains **only**:

1. The minimal preamble needed to match the paper’s formatting:
   - Use the same class and style options as the main paper whenever possible:
     ```latex
     \documentclass{article}
     % Or: \documentclass[preprint]{neurips_2023}
     % Or: \usepackage[preprint]{neurips_2023} with article
     ```
   - Copy over any style package files if they are local (e.g. `neurips_2023.sty`) into the temp folder so LaTeX can find them.
   - Include only the packages required for the algorithm environment and math:
     ```latex
     \usepackage[utf8]{inputenc}
     \usepackage{amsmath,amsfonts}
     \usepackage{algorithm}
     \usepackage{algorithmicx}
     \usepackage[noend]{algpseudocode}  % or algorithm2e, etc.
     \usepackage{xspace}                % only if used
     ```
   - **Disable page numbers/headers/footers**:
     ```latex
     \pagestyle{empty}
     ```

2. The algorithm block itself, copied verbatim from `{ALGO_FILE}`:
   - Include the full `\begin{algorithm} ... \end{algorithm}` (or appropriate env).
   - Include the original `\caption{...}` and `\label{...}`.
   - Include any helper macros defined nearby (e.g. `\newcommand{\algname}{...}`) if needed.
   - If the algorithm uses `minipage`/subblocks, retain them as-is to match layout.

3. A minimal document body:
   ```latex
   \begin{document}

   \begin{algorithm}
   \caption{...}
   % body copied from paper
   \end{algorithm}

   \end{document}
   ```

**Important**:
- Do **not** wrap the algorithm in additional `center`/`minipage` unless the original does; the closer you stay to the original snippet, the closer the formatting will be.
- If line breaks in the standalone version differ from the paper, consider copying exactly the snippet as LaTeX sees it (including manual `\\` or `\Statex` lines), and keep the same class/style to minimize differences.

---

## 3. Compile & Remove Page Number / Extra White Space

From the temp directory (e.g. `tmp/{PAPER_NAME}_alg/`):

1. **Compile to PDF**:
   ```bash
   pdflatex -interaction=nonstopmode -halt-on-error {ALGO_ID}.tex
   ```

2. **Crop tightly to the algorithm block**:
   - Use `pdfcrop` to remove surrounding page margins and whitespace:
     ```bash
     pdfcrop --margins '0 0 0 0' {ALGO_ID}.pdf {ALGO_ID}-crop.pdf
     ```
   - The `--margins '0 0 0 0'` option asks `pdfcrop` to crop exactly to the bounding box of the algorithm (no extra padding).

If compilation fails due to missing packages or macros:
- Read the `.log` file to identify what is missing.
- If it’s a local style file (e.g., `neurips_2023.sty`), copy it from the paper source into your temp directory.
- If it’s a missing package from TeX Live, tell the user exactly which package is needed and ask whether they can install it (do not fake the package).

---

## 4. Convert to SVG or PNG

Once you have `{ALGO_ID}-crop.pdf`:

- **Preferred: SVG (vector)**:
  - Use `pdftocairo` or `dvisvgm`:
    ```bash
    pdftocairo -svg {ALGO_ID}-crop.pdf {ALGO_ID}-crop.svg
    ```
  - Result will be `{ALGO_ID}-crop.svg`, which preserves vector quality and text.

- **Alternative: PNG** (if SVG is not desired or supported):
  - You can render a high-DPI PNG:
    ```bash
    pdftocairo -png -r 300 {ALGO_ID}-crop.pdf {ALGO_ID}-crop
    ```
  - This typically produces `{ALGO_ID}-crop-1.png`.

- **Scaling (size) behavior**:
  - By default, **do not rescale** the algorithm; keep the 1× size that matches the paper’s layout.
  - If the user explicitly asks for larger/smaller versions:
    - For SVG: multiply `width`/`height` attributes, or let the consumer (e.g., markdown renderer) scale via CSS or image attributes.
    - For PNG: re-run `pdftocairo` with a different DPI (e.g., `-r 150` vs `-r 300`) or use an image tool (`convert -resize ...`).

---

## 5. File Placement & Markdown Insertion

Place the final algorithm figure under the corresponding note directory:

- Output path:
  - `notes/{PAPER_NAME}/figures/{ALGO_ID}.svg` (or `.png`)

- In `notes/{PAPER_NAME}/main-note.md`, insert near the section that describes the algorithm (e.g., the method section):

```markdown
**Algorithm {N} ({Algorithm short name})**  

![Algorithm {N}: {short description}](figures/{ALGO_ID}.svg)  
> "{Original caption text from the paper, possibly lightly shortened but preserving key semantics.}" (Algorithm {N}; paper-source/primary/tex/{ALGO_FILE})
```

Guidelines:
- Use a short, descriptive alt text in `![...]`.
- Keep the caption blockquote faithful to the paper’s caption.
- Reference the source location in the repository (`paper-source/primary/tex/{ALGO_FILE}`) so readers know where the definition came from.

---

## 6. Placeholders & Reuse

When reusing this procedure for other papers, substitute:

- `{PAPER_NAME}`: short handle for the paper (e.g., `SpQR`).
- `{ALGO_ID}`: a stable identifier for the algorithm, often matching its LaTeX label (e.g., `alg_main` or `spqr_algorithm1`).
- `{ALGO_FILE}`: the `.tex` file containing the algorithm (e.g., `method.tex`).
- `{ALGO_ENV}`: the environment type (`algorithm`, `algorithm2e`, etc.).

Always:
- Match the original class/style as closely as possible.
- Disable page decorations with `\pagestyle{empty}`.
- Crop with `pdfcrop --margins '0 0 0 0'` to remove all extra whitespace.
- Only rescale (2×, 0.5×, etc.) if the user explicitly requests a size change.

