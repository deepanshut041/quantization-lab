# Semantic Search With a Compressed Index

Build a local search engine that returns relevant passages with source links. Compare full-precision, INT8 and packed-binary embedding indexes, then test whether rescoring recovers useful ranking quality within a smaller memory budget.

**Status — 2026-10-26:** runnable Python boilerplate is complete. Frozen Python 3.12.13 setup, 10 scaffold checks, source/wheel builds and isolated installed execution pass. Model references remain pinned. Model compatibility, data acquisition, model execution, quantization, task evaluation and the application remain pending.

## Run the boilerplate

In the workspace TUI, open **Projects → Quantization Lab**, select this project, choose **Project status**, **Check setup** or **Show configuration**, then press **Run** (`g`). The same panel provides **Setup env**, **Checks** and **Build pkg**. These runners operate on the boilerplate; model execution remains pending.

Run these commands from this project folder. It has its own Python 3.12.13 environment and lockfile.

```bash
uv sync --frozen
uv run --frozen python src/check_setup.py
uv run --frozen python src/compressed_semantic_search.py status
uv run --frozen python src/show_config.py
uv run --frozen pytest -q
uv build
```

The installed command is `compressed-semantic-search`; for example, `uv run --frozen compressed-semantic-search status`. An explicit `--config path/to/config.toml` selects another project configuration. Defaults live in [src/core/project.toml](src/core/project.toml) and travel with the installed package.

Entry scripts live directly in `src/`. The `core`, `data`, `quantization`, `evaluation` and `utils` folders separate responsibilities; the last model/data stages contain no implemented algorithms yet. The status/check commands read configuration and report setup. Future workflow commands return `not_implemented` with exit code 2 and create no outputs. Each project uses its own environment.

Runtime dependencies are currently empty. Model frameworks and backend-specific dependencies will be selected during the compatibility probe. Private study records, environments, data, model artifacts and generated runs are ignored and excluded from distributions.

## Current model target

Use **[Google EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)**, released on [6 October 2026](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/). It is a current multimodal embedding model; begin with its text path and keep other modalities outside the initial comparison.

| Item | Planned reference |
| --- | --- |
| Official checkpoint | `google/embeddinggemma-2` |
| Revision checked on 26 October 2026 | `914f7f89142e33e77833254d9c9b90c3cef7303b` |
| Initial vector width | 768 dimensions |
| Initial inference arithmetic | FP32, or verified BF16 support |
| Initial stored reference vectors | FP32 |

The publisher lists 740M parameters for the full model and a 270M text component with optional modality encoders. Its model card specifies task-specific query/document formatting and requires BF16 or FP32 inference; FP16 is unsuitable for its activation range. Supported shorter vector widths include 512, 256 and 128, with normalization required after truncation. Treat dimension reduction as its own later comparison.

This metadata was checked against the official model card and Hugging Face API; weights were not acquired and local compatibility has not been tested. Pin the exact encoder, tokenizer, prompts, compute dtype and output normalization at kickoff. Recheck releases before that freeze; preserve the reference when a newer model is introduced later.

## The useful application

The user enters a question and receives ranked passages, document titles and source links. A comparison view shows which relevant results move or disappear under each index representation. It also shows measured memory and latency so an index choice has an observable consequence.

The quantization target is the stored embedding vector. Keep encoder weights fixed initially. The resulting index can later be consumed by the sibling [document assistant](../document-assistant/README.md) and the existing planned [Semantic Notes Search](../../transformer-labs/semantic-notes-search/README.md). Preserve that project's encoder-learning/fine-tuning scope; this lab owns the compression comparison.

## Ordered build route

All phases are pending. Explain the consequential maths and inspect actual results as each phase is implemented.

| Phase | Work | Evidence needed to continue |
| --- | --- | --- |
| 00 — Encoder and data contract | Verify the pinned model's text path, formatting, finite vectors and dimensions; select a public corpus with relevance labels | Frozen source identities, input policy and development/final boundaries |
| 01 — FP32 search | Encode once, save stable document/vector IDs and establish exact retrieval | Reproducible scores and a working relevance evaluator |
| 02 — INT8 | Fit scalar ranges from declared calibration data and quantize corpus vectors | Independent numeric checks, measured ranking changes and true saved bytes |
| 03 — Packed binary | Pack signs/bits, implement consistent query scoring and compare results | Correct bit/ID handling and an honest quality result, including a poor outcome |
| 04 — Retrieval and rescoring | Retrieve additional binary candidates and rescore them | Measured quality recovery versus added storage and latency |
| 05 — Dimension study | Compare selected supported widths using the fixed encoder | Isolated dimension/normalization effects before combining compression settings |
| 06 — Saved search tool | Reload indexes and provide ranked passages with a comparison interface | Repeatable search after restart and a report covering quality, memory and latency |

## Data and quality measurement

[BEIR](https://github.com/beir-cellar/beir) provides public corpora, relevance judgments and evaluation tooling. SciFact is a candidate for a small correctness start; FiQA is a candidate for the first larger study. Select one for the main comparison, record membership/hashes and keep query-label tuning separate from final evaluation. A small probe is not the final dataset result.

Keep documents, chunking, encoder prompts, normalization and query membership fixed across index representations. The index may contain the evaluation corpus, while held-out relevance labels remain outside fitting/tuning. Declare exactly which corpus/vector data determine calibration ranges; do not silently fit them from final evaluation queries.

Measure Recall@10 and nDCG@10, inspect lost relevant passages and retain per-query rankings. Where judgments are incomplete, distinguish an unjudged passage from a known irrelevant passage. Use validation to choose candidate count and settings, then run the frozen final comparison.

## Compression experiments

The initial sequence is FP32 retrieval, calibrated INT8 retrieval, binary retrieval and binary retrieval plus rescoring. Keep search settings controlled before adding approximate indexes. With the encoder fixed, quantization changes the stored vectors and scoring path; it does not establish an encoder speedup.

For INT8, study one calibration change at a time and record the range/scale policy. For binary, explain sign encoding, bit packing and Hamming-based candidate scoring. Test useful retrieval empirically: a current embedding model is not automatically trained to preserve its ranking after binarization.

For rescoring, compare a bounded candidate-count sweep and declare whether the second stage uses INT8 or FP32 vectors. A rescorer cannot recover a relevant passage that never reaches its candidate set. The [Sentence Transformers guide](https://www.sbert.net/examples/sentence_transformer/applications/embedding-quantization/README.html) and its [retrieval demo source](https://huggingface.co/spaces/sentence-transformers/quantized-retrieval/blob/main/app.py) provide practical implementation references.

Only after the precision comparison, study a supported shorter vector width. Match query/corpus dimensions and normalize the shortened vectors consistently. Freeze that dimension before combining it with INT8 or binary storage, so the source of a quality change stays identifiable.

## Memory, speed and reproducibility

Report embedding payload bytes and total deployed storage separately. Include calibration parameters, document IDs, search structures, source-text metadata and every rescoring store. Retaining an FP32 analysis archive does not make it part of the deployment unless the serving path reads it; report both scopes explicitly.

Measure peak process/device memory, index construction time and repeated query latency, including median and tail behavior. Separate query encoding from index search/rescoring and also report end-to-end latency. Distinguish warm and cold reads; hold device, thread count, batch size and caching policy fixed within a comparison.

Save the encoder/data identities, index format/configuration, ID mapping, calibration state and hashes. Check finite vectors, normalization, boundary values, packing/padding, rank/score consistency and reload behavior during implementation. The scaffold checks pass; vector arithmetic, encoder execution and retrieval-quality checks remain pending.

## Completion and next action

The result is a working search engine, reloadable indexes, quality/memory/latency evidence and concrete failure examples. The chosen index must satisfy a declared task-quality budget; compression and speedup are measured outcomes.

Next at implementation kickoff: verify the current pinned encoder on a small public text sample, choose the main BEIR dataset and freeze the first FP32/INT8 experiment. Continue the private setup records through the existing project-learning layout when that probe starts. This is an ordinary folder tracked by `ai_projects`, with no nested Git repository.

[Back to Quantization Lab](../README.md) · [Projects checkpoint](../../../docs/guide/resume.md)
