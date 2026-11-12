# Upstream contribution notes (drafts, nothing submitted)

Checked 2026-11-12. All three frameworks now want third-party integrations as **your own PyPI package** with a
listing in their docs. They no longer take integration code in the framework's own repo. `chopper-rag` already has the
shape they expect. The open question is naming: one package with three extras (current), or three framework-named
packages (below).

## LangChain

- **Where:** a standalone PyPI package. "All new integrations must be published as independent packages to PyPI (e.g.
  `langchain-yourprovider`)". They no longer accept integration PRs to `langchain-ai/*`, and the guide does not mention
  `langchain-community`. Document compressors are on neither the "integrate these" nor the "not these" list.
  Document transformers and loaders are on the "not these" list, so a listing may be questioned.
- **Name, if split out:** `langchain-chopper` (package `langchain_chopper`, class `ChopperCompressor`).
- **Steps:** implement the package → run the standard tests where they apply (there is no standard-test suite for
  compressors) → publish to PyPI → file the **Integration listing issue** in `langchain-ai/docs` (template
  `06-integration-submission.yml`). After a maintainer applies the `integration-run` label, automation adds a row to
  `scripts/data/integration_external_docs.yaml`. Do not open a manual PR or an MDX page; automation closes them. A
  hosted docs page needs 50k+ monthly downloads or a maintainer feature.
- **Links:** https://docs.langchain.com/oss/python/contributing/integrations-langchain ·
  https://docs.langchain.com/oss/python/contributing/publish-langchain ·
  https://github.com/langchain-ai/docs/issues/new?template=06-integration-submission.yml
- **Listing text:**
  > **Chopper (context compressor).** `ChopperCompressor` is a `BaseDocumentCompressor` that cuts exact quotes
  > ("chops") out of retrieved documents with a fine-tuned mmBERT encoder (0.31B / 0.14B, MIT, runs locally on CPU or
  > GPU). It typically keeps 3–5% of the retrieved text and returns one Document per chop, with character offsets back
  > into the source. On multi-hop MuSiQue, Qwen3-14B answering from the chops beats answering from all 20 snippets at
  > 2–4 hops, using ~9% of the tokens. Works inside `ContextualCompressionRetriever`. Package: `chopper-rag[langchain]`,
  > model: huggingface.co/wiiiktor/chopper_mmbert.

## LlamaIndex

- **Where:** a standalone package. CONTRIBUTING.md: "we are no longer accepting new integration packages in this
  repository". New integrations "should be maintained in their own repositories and published to PyPI independently".
  A PR that adds a new `pyproject.toml` is closed automatically. Existing postprocessors live under
  `llama-index-integrations/postprocessor/llama-index-postprocessor-*`, so a split-out package would be named
  `llama-index-postprocessor-chopper` (import `llama_index.postprocessor.chopper`) to match the convention.
- **Their requirements for code they host** (a good bar for ours as well): unit tests that mock remote systems, at least
  50% coverage, `uv` + pre-commit + `make lint`, and an AI-use disclosure in the PR. I could not find a current
  LlamaHub listing procedure for external packages (the community integrations page does not describe one), so ask on
  their Discord or GitHub Discussions.
- **Links:** https://github.com/run-llama/llama_index/blob/main/CONTRIBUTING.md ·
  https://developers.llamaindex.ai/python/framework/community/integrations/
- **Listing / announcement text:**
  > **ChopperPostprocessor.** A node postprocessor that replaces retrieved nodes with the exact quotes that answer the
  > query, cut by a fine-tuned mmBERT encoder (local, MIT). Each output TextNode keeps the source metadata, character
  > offsets and a SOURCE relationship to the retrieved node, and it exposes an ANSWER/SEARCH decision for agentic
  > re-retrieval. `pip install "chopper-rag[llama-index]"`;
  > `index.as_query_engine(similarity_top_k=16, node_postprocessors=[ChopperPostprocessor()])`.

## Haystack

- **Where:** `deepset-ai/haystack-core-integrations` takes new integrations only through their enhancement process, and
  only if they serve "the majority of our users". For anything narrower: "consider writing and distributing the
  integration on your own". So: keep our own package, then open a PR to **`deepset-ai/haystack-integrations`** that adds
  `integrations/chopper.md` (start from `draft-integration.md`). The page then appears on
  haystack.deepset.ai/integrations.
- **Frontmatter they require:** `layout: integration`, `name`, `description`, `authors` (name, optional socials),
  `type` (here: `Custom Component`), and at least one of `pypi` or `repo`. Optional: `report_issue`, `logo`, `version`,
  `toc`. The body should hold install and usage instructions.
- **Note:** the component was tested on haystack-ai 3.5.0 (installed). It uses only `@component`, `warm_up`, and
  `to_dict`/`from_dict`, which exist unchanged since 2.0.
- **Links:** https://github.com/deepset-ai/haystack-integrations ·
  https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md
- **Draft `integrations/chopper.md`:**

```markdown
---
layout: integration
name: Chopper
description: Cut exact quotes that answer the query out of retrieved documents with a local mmBERT context compressor
authors:
    - name: wiiiktor
      socials:
        github: wiiiktor
pypi: https://pypi.org/project/chopper-rag
repo: https://github.com/wiiiktor/chopper_demo
type: Custom Component
report_issue: https://github.com/wiiiktor/chopper_demo/issues
version: Haystack 2.0
toc: true
---

`ChopperDocumentCompressor` goes after a retriever or ranker. It replaces the documents with the exact quotes
("chops") that answer the query, typically 3–5% of the text, never rewritten, each with character offsets into its
source. It also outputs an `ANSWER` / `SEARCH` decision. Model: huggingface.co/wiiiktor/chopper_mmbert (0.31B or
0.14B, MIT, runs locally).

## Installation
pip install "chopper-rag[haystack]"

## Usage
(the Pipeline example from the chopper-rag README)
```

## Before any submission

- The test suite downloads the real weights. Their LangChain and LlamaIndex rules want unit tests that mock remote
  systems, so add a mocked test (a fake tokenizer and encoder) if the code ever goes into their repos.
- The PyPI name `chopper-rag` and the GitHub repo URL in `pyproject.toml` (`wiiiktor/chopper_demo`) must exist first.
