you are tasked to find related works of a given topic or a given paper, conduct it in this way.

# Guidelines for Finding Related Works

## Universal guidelines

- rank the paper by relevance to the topic
- if it has github repository, include a link to it
- if it has online tutorial, include a link to it, find on `https://medium.com/` and `https://www.zhihu.com/` ONLY.
- if it has youtube demo, include a link to it
- technical reports from large groups like Google, Microsoft, OpenAI, etc. are considered as papers
- save the information in a markdown file, name it `survey-<topic or paper title>.md`, if that name is already taken, use a new name, do not overwrite existing files unless you are explicitly requested to do so. By default, save the file in workspace root, unless specified otherwise.

### Report Format

```markdown

# Related Works on [Topic or Paper Title]

## Overview

- Information about the topic or paper

## Related Papers

### <field_name>

#### [Paper Title](link_to_paper)
- **Venue**: Journal or Conference Name, or ArXiv ID, year
- **Github Page**: [Github Link](link_to_github)
- **Tutorial**: [Tutorial Link](link_to_tutorial)
- **YouTube Demo**: [Demo Link](link_to_demo)
- **Summary**: Brief summary of the paper, its contributions, and relevance to the topic.

more papers ...

### <field_name>

...

```

## If you are given a topic

- identify the main concepts and keywords related to the topic
- find latest research papers related to this topic
- for each found paper, summarize its main contributions and how it relates to the topic, be concise, people will read the paper if they want more details

## If you are given a paper

- identify the main area of research and the key contributions of the paper
- if the paper can be found on arXiv, and has html version, use the html version to extract the abstract, main contributions, and references
- use that information to find related works