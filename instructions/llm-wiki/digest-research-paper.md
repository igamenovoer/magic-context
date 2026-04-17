# Digest a Research Paper Into an LLM Wiki

Use this instruction when converting a research paper into a structured wiki entry. The goal is not only to summarize the paper, but to preserve the parts that make the paper useful later: what problem motivated it, what method it proposes, why the method is designed that way, what the experiments prove, where the method is limited, and what future problems can borrow ideas from it.

This is a self-contained generic research-paper digest instruction.

---

## Required Inputs

Before writing the digest, gather:

1. **Paper source**: PDF, arXiv HTML, publisher page, extracted Markdown, or LaTeX source.
2. **Target wiki location**: the target `llm-wiki` knowledge base. If it has `wiki/index.md`, read it first and link to existing entity/concept pages instead of duplicating them.
3. **Reference code status**: whether the user provided code, a public repo exists, or no implementation is available.
4. **Output path**: if the user does not specify one, create a wiki source entry under the target wiki's normal paper/source location.

If any input is missing, state that in the digest instead of guessing.

---

## Required Sections

The digest SHALL include the following sections in this order. If the paper has no material for a required section, keep the heading and write one explicit sentence such as "The paper does not report this."

### 0. Metadata

- **Full title**
- **Authors**
- **Venue / year**
- **Links**: PDF, arXiv/DOI, project page, code, dataset
- **Research area**
- **Keywords**
- **Paper handle**: a short stable name for wiki links
- **Date ingested**: YYYY-MM-DD

### 1. TL;DR

Write 3-6 bullets using bold key terms:

- **Problem**: the main problem or gap.
- **Core idea**: the central insight.
- **Method**: the proposed method at one sentence of detail.
- **Result**: the strongest result with exact numbers when reported.
- **Why it matters**: what later work or engineering work can reuse.

### 2. Motivation And Problem Setting

Explain why the paper was written.

Cover:

- **Problem definition**: the task, setting, assumptions, and inputs/outputs.
- **Prior limitations**: what earlier methods, systems, or theories fail to handle.
- **Why now**: any new data scale, hardware trend, application demand, bottleneck, or theoretical gap that makes the problem important.
- **Evaluation target**: what success means for this paper.

Use citations from the paper for the claimed motivation. Prefer direct blockquotes for the authors' own framing.

### 3. Proposed Method

Describe what the paper proposes.

Cover:

- **High-level method**: the method in plain language.
- **Main components**: modules, algorithms, losses, training stages, system components, or theoretical steps.
- **Inputs and outputs**: tensor shapes, data structures, API contracts, mathematical objects, or protocol boundaries when relevant.
- **Algorithm flow**: include pseudocode or a Mermaid diagram when it clarifies the method.
- **Relationship to prior work**: what is inherited, modified, or replaced.

Do not invent implementation details. If the paper is abstract or underspecified, say so.

### 4. Design Rationale

Explain why the proposed method is designed this way.

For each major design choice, answer:

- **Motivating bottleneck or gap**: what problem the design choice addresses.
- **Mechanism**: how the design choice addresses that problem.
- **Rejected alternatives**: alternatives the paper compares against, ablates, or implicitly avoids.
- **Evidence**: ablations, analysis, theory, qualitative examples, or measurements that support the choice.

This section is essential. The digest SHALL not only say what the method is; it SHALL explain the rationale of the method's design.

When the rationale is inferred rather than directly stated, mark it as `(inferred)` and explain the basis for the inference.

### 5. Experiments And Results

Capture experimental evidence precisely.

Cover:

- **Datasets / benchmarks**: names, versions, splits, sizes, preprocessing, and evaluation protocol.
- **Baselines**: methods compared against, including versions or citations when provided.
- **Metrics**: accuracy, loss, BLEU, F1, mAP, latency, throughput, cost, memory, human preference, theorem proof rate, or other reported metrics.
- **Main results**: reproduce key numbers in compact Markdown tables when useful.
- **Ablations**: what each component contributes.
- **Scaling / robustness**: how results vary with size, data, distribution shift, noise, compute budget, or task difficulty.
- **Qualitative results**: examples, failure cases, visualizations, or case studies.
- **Statistical confidence**: standard deviation, confidence intervals, number of runs, or significance tests if reported.

Preserve numbers exactly as reported. Label whether a number is benchmark-level, end-to-end, component-level, simulated, or theoretical.

### 6. Limitations

Separate limitations into:

- **Author-stated limitations**: constraints or weaknesses the paper explicitly acknowledges.
- **Experiment coverage gaps**: missing baselines, missing datasets, narrow tasks, small sample size, limited hardware, or weak ablations.
- **Method assumptions**: assumptions that may fail in practice.
- **Deployment risks**: cost, latency, robustness, fairness, privacy, safety, reproducibility, or maintenance concerns when relevant.

Do not exaggerate. If a limitation is your own inference, mark it as `(inferred)`.

### 7. When This Paper Is Useful

Explain what kinds of future problems can borrow ideas from this paper.

Cover:

- **Problem patterns**: the recurring problem shape this paper addresses.
- **Reusable ideas**: mechanisms, algorithms, data structures, losses, evaluation designs, or system architecture patterns that can transfer.
- **Good fit**: conditions where the method is likely to help.
- **Bad fit**: conditions where the method is unlikely to help.
- **Adaptation hints**: how to modify the idea for adjacent tasks.

This section should answer: "When faced with what problems can this paper provide ideas?"

### 8. Source Code Notes

If source code is available, record:

- repo URL and commit hash or release tag,
- license,
- key implementation files,
- whether the code matches the paper,
- important deviations from the paper.

If no source code is available, write: "No public implementation was analyzed for this digest."

Only cite code when the user supplied code or a public implementation was intentionally fetched. Do not assume code exists.

### 9. Wiki Links And Follow-Ups

Add wiki maintenance notes:

- **Existing pages to link**: related entities, concepts, methods, datasets, models, or systems.
- **New pages to create**: concepts or entities that deserve standalone wiki pages.
- **Related papers**: predecessors, baselines, follow-up work, or surveys.
- **Open questions**: questions a future reader should investigate.

---

## Citation Rules

- Ground every important claim with a citation to the paper, code, or an authoritative source.
- Use blockquotes for key claims from the paper:

```markdown
> "Direct quote from the paper." (Section 3.2; paper-source/path-or-url)
```

- Put citations immediately after the claim they support.
- Cite exact tables, figures, algorithms, equations, or sections when referencing results.
- Do not overquote long passages. Quote only the words needed to ground the claim, then explain in your own words.

---

## Style Rules

- Write as a wiki page for future retrieval, not as a review essay.
- Prefer concrete names over vague labels such as "the method" after first mention.
- Use tables for results, ablations, and method comparisons.
- Use Mermaid diagrams for algorithms, pipelines, or system structure when useful.
- Mark uncertainty explicitly: `unknown`, `not reported`, or `(inferred)`.
- Keep claims falsifiable. Avoid generic praise such as "novel and effective" unless the paper substantiates it.
- Preserve exact reported numbers and units.
- Distinguish paper claims from your own analysis.

---

## Quality Checklist

Before saving the digest, verify:

- [ ] Motivation explains the problem, prior limitations, and why the problem matters.
- [ ] Proposed method is clear enough that a reader can restate the algorithm or system.
- [ ] Design rationale explains why major design choices were made.
- [ ] Experiments include datasets, baselines, metrics, exact main results, and ablations when reported.
- [ ] Limitations separate author-stated limits from inferred limits.
- [ ] "When This Paper Is Useful" maps the paper to reusable problem patterns and adaptation hints.
- [ ] Code notes state whether implementation was analyzed.
- [ ] Wiki links and follow-ups connect the digest to existing or future wiki pages.
- [ ] Every important claim has a nearby citation.
