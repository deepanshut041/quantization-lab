# Quantization Lab

This collection is the independent `quantization-lab` Git repository, checked out at `ai_projects/projects/quantization-lab/` as a submodule of `ai_projects`. Its project folders are ordinary directories tracked by this collection. Shared tutorials and learning checkpoints remain in the umbrella Docs submodule. Local bare origin: workspace `.ai/repositories/quantization-lab.git`.

Commit project changes in this collection first, then commit the updated collection pointer in `ai_projects`. Local commits are not automatically copied to the bare origin; run `git push origin main` in this collection before another checkout needs the new commit. No external hosting is configured.

Build useful applications while learning how quantization changes their memory use, speed and task quality. The two projects below were approved for folder/README preparation after reviewing upstream implementations and evaluation tools. An 8B LLM was an example of practical scope; current model targets are selected for each application rather than imposing one model size on the collection.

**Status — 2026-10-26:** both projects have runnable Python boilerplates, independent Python 3.12.13 environments, locked tooling, pinned model configuration and direct/module/console commands. All 19 scaffold checks, source/wheel builds and isolated installed checks pass. Model compatibility, data/model acquisition, quantization, task evaluation and applications remain pending.

This is an ordinary folder tracked by `ai_projects`. Both projects are ordinary child folders. Shared guidance remains in [Projects docs](../../docs/README.md).

## Project folders and current models

| Project | Folder / README | Planned starting model |
| --- | --- | --- |
| Local Document Assistant Under a Memory Budget | [document-assistant](document-assistant/README.md) | [Gemma 4 E4B instruction-tuned](https://huggingface.co/google/gemma-4-E4B-it), with E2B as a smaller fallback |
| Semantic Search With a Compressed Index | [compressed-semantic-search](compressed-semantic-search/README.md) | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), released 6 October 2026 |

Each project README owns its setup commands, source revision, model-specific constraints, ordered build route and completion evidence. Recheck releases before implementation, then freeze revisions within a comparison. Current source availability does not establish local compatibility or measured quality.

## What people are actually building

| Example reviewed | What exists upstream | What it suggests for this lab |
| --- | --- | --- |
| [OpenVINO document Q&A notebook](https://docs.openvino.ai/2024/notebooks/llm-rag-langchain-with-output.html) | A RAG application with embeddings, reranking and a language model, including 4/8-bit weight compression | Measure compression through a usable document assistant |
| [Optimum Intel optimization](https://huggingface.co/docs/optimum-intel/openvino/optimization) | Data-free and data-aware weight compression, configurable group size/precision ratio, AWQ and GPTQ options | Compare methods under one supported runtime, rather than only downloading differently named files |
| [Sentence Transformers retrieval demo source](https://huggingface.co/spaces/sentence-transformers/quantized-retrieval/blob/main/app.py) | Binary candidate search and INT8 rescoring over a reported 41-million-text Wikipedia corpus | Build and evaluate a small-memory search application |
| [KIVI](https://github.com/jy-yuan/KIVI) | A research implementation of quantized KV cache with long-context evaluation | Study weights and runtime conversation memory separately; its original CUDA implementation is a reference, not an assumed local dependency |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp#quantization) | Integer-quantized speech models and transcription tools | A credible later audio project; the two proposals below first connect model compression and retrieval |

These examples establish that the application patterns exist. Their published performance does not predict our hardware results. The following project designs and recommendation are this lab's proposals.

## Idea 1: Local Document Assistant Under a Memory Budget

**Useful result:** a local tool that answers questions about selected documents and shows the supporting passages. A comparison view exposes whether a smaller model variant missed evidence, changed an answer or invented unsupported content.

**Central question:** which compression recipe keeps the assistant useful within a declared memory and response-time budget?

### What we build

1. Validate the planned Gemma 4 target and establish a floating-point reference on a supported runtime. Freeze the final source/runtime after a hardware capacity check.
2. Build a local document-to-answer path with visible sources. Freeze document parsing, input passages and prompts during the weight-compression study.
3. Produce our own compressed artifacts and retain their recipe, source identity and hashes.
4. Add repeatable quality, memory and latency evaluation, plus a saved-model reload check and local application.

### Experiments worth doing

| Stage | Change | Hold fixed | What we learn |
| --- | --- | --- | --- |
| Precision baseline | Floating point, 8-bit and 4-bit weights where supported | Model, inputs, runtime, placement, decoding and cache policy | The measured size/quality/speed tradeoff |
| Calibration | Data-free compression versus calibration using general text or the application domain | Target precision and other compression settings | Whether representative calibration preserves useful behavior |
| Method | Basic 4-bit compression versus a supported AWQ or GPTQ recipe | Checkpoint, calibration membership/budget and evaluation | How activation/error-aware compression changes the result |
| Granularity | One bounded group-size or mixed-precision change | The selected method and task | Scale overhead and sensitive-layer tradeoffs |
| Later context study | Supported KV-cache precisions across bounded input lengths | The chosen weight artifact and per-length task inputs | How runtime context memory differs from weight storage |

Start with the precision baseline; proceed through the later stages individually after reviewing each result. This is not an exhaustive cross-product. Optimum Intel/NNCF and llama.cpp are candidate routes; select one initial route and verify its actual model/device support. An AWQ configuration that is skipped by the exporter is not a successful AWQ comparison. The [Qwen llama.cpp guide](https://qwen.readthedocs.io/en/stable/quantization/llama.cpp.html) also provides a calibration/importance-matrix path for supported LLMs.

### How we judge the result

Use public documents and a frozen set of questions with checked answers and evidence passages, including questions the documents cannot answer. Keep calibration and tuning examples outside final evaluation. Review answer correctness, evidence correctness and unsupported answers with a declared rubric; perplexity is only an additional diagnostic. [LongBench](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md) provides candidate document-Q&A evaluation tasks; any subset must be declared and reported as such.

Measure actual peak host/device memory, model bytes, prompt-processing time, time to first token and generation speed. Keep preparation/calibration cost separate from inference. A reference that requires different device placement remains a quality reference; its timing is not a controlled quantization speed comparison.

**Learning:** group-wise scales and zero points, outlier sensitivity, calibration, activation-aware versus error-aware methods, selective precision, KV cache and reliable application evaluation. Explain each mechanism with small calculations alongside the real implementation.

**Finish line:** an application that reloads the selected candidate, a reproducible compression/evaluation pipeline and an evidence-based recommendation for the declared budget, including failure examples. No speedup or quality threshold is promised before measurement.

## Idea 2: Semantic Search With a Compressed Index

**Useful result:** a local search tool for a substantial public text collection, returning ranked passages and source links. Its comparison view shows what retrieval quality and memory change when stored embedding vectors use lower precision.

**Central question:** how much index memory can we save while still finding the documents users need?

### What we build

1. Freeze a public retrieval corpus and its relevance judgments. [BEIR](https://github.com/beir-cellar/beir) provides evaluation tooling and candidate datasets: SciFact for a small correctness start, or FiQA for a larger first study. Choose and pin one before the main comparison.
2. Encode the corpus once with the planned EmbeddingGemma 2 text path and retain an FP32 reference for analysis. Check useful binary retrieval empirically before adopting that arm.
3. Build FP32, calibrated INT8 and packed-binary retrieval variants with consistent document IDs.
4. Add a two-stage search: retrieve extra candidates using binary vectors, then rescore them using higher-precision query/vector information.
5. Save/reload the indexes and expose ranked results, source links and per-query comparisons in a local search application.

### Experiments worth doing

Compare FP32 search, INT8 search, binary search and binary search plus INT8 or FP32 rescoring. Then vary the candidate count and one calibration policy at a time. Keep the embedding model, documents, chunking, normalization and query set fixed. Begin with controlled search settings so changes in approximate indexing do not masquerade as quantization effects.

Measure Recall@10 and nDCG@10 against relevance judgments, inspect queries whose relevant documents disappear, and report query latency, index-build time, peak memory and total saved bytes. Include rescoring stores, IDs and index overhead in total storage; distinguish embedding payload compression from whole-system savings. A rescorer can reorder retrieved candidates but cannot recover a relevant document excluded from its candidate set.

The [Sentence Transformers embedding-quantization guide](https://www.sbert.net/examples/sentence_transformer/applications/embedding-quantization/README.html) provides scalar/binary quantization, rescoring and benchmark examples. This study targets stored vectors. Quantizing the encoder weights is a later, separate experiment.

**Learning:** scalar and binary quantization, packed bits, dot products versus Hamming distance, ranking distortion, calibration and candidate-retrieval/rescoring tradeoffs.

**Finish line:** a working search engine with reloadable indexes and a measured quality/memory/latency report. Its selected index can later be consumed by the existing planned [Semantic Notes Search](../transformer-labs/semantic-notes-search/README.md); preserve that project's own encoder-learning/fine-tuning scope instead of creating a competing notes application.

## Recommendation and next action

Start with the document assistant if model quantization is the main interest. It includes a real application, multiple compression mechanisms and quality failures we can inspect. The compressed-index project then adds a different quantization target and can eventually supply retrieval for the assistant.

Next at implementation kickoff: verify the document assistant's model/runtime compatibility and freeze its first bounded reference/8-bit experiment. The existing [learning checkpoint](../../docs/guide/resume.md), Attention/Bike evidence and separate [Research](../../../ai_research/README.md) experiments remain unchanged. Both projects are registered in the workspace TUI under Projects → Quantization Lab, with status/setup/configuration runners and setup/test/build actions. Documentation-reader registration remains pending.
