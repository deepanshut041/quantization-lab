# Local Document Assistant Under a Memory Budget

Build a local application that answers questions about documents and shows the passages supporting each answer. Use the application to measure how model quantization changes correctness, memory use and response time, then choose a saved model that meets a declared deployment budget.

**Status — 2026-10-26:** runnable Python boilerplate is complete. Frozen Python 3.12.13 setup, 9 scaffold checks, source/wheel builds and isolated installed execution pass. Model references remain pinned. Model compatibility, data acquisition, model execution, quantization, task evaluation and the application remain pending.

## Detailed tutorial

Start the [complete Document Assistant written tutorial](../../../docs/tutorials/projects/document-assistant/quantization-lab/v1/lessons/01-start.md), or open Projects Documentation → Tutorials → Document Assistant. It covers all seven planned phases through 24 connected lessons with worked maths, complete runnable teaching references, diagrams, practice and implementation/readiness records. Teaching fixtures are identified explicitly; actual model implementation and results remain pending.

## Run the boilerplate

In the workspace TUI, open **Projects → Quantization Lab**, select this project, choose **Project status**, **Check setup** or **Show configuration**, then press **Run** (`g`). The same panel provides **Setup env**, **Checks** and **Build pkg**. These runners operate on the boilerplate; model execution remains pending.

Run these commands from this project folder. It has its own Python 3.12.13 environment and lockfile.

```bash
uv sync --frozen
uv run --frozen python src/check_setup.py
uv run --frozen python src/document_assistant.py status
uv run --frozen python src/show_config.py
uv run --frozen pytest -q
uv build
```

The installed command is `document-assistant`; for example, `uv run --frozen document-assistant status`. An explicit `--config path/to/config.toml` selects another project configuration. Defaults live in [src/core/project.toml](src/core/project.toml) and travel with the installed package.

Entry scripts live directly in `src/`. The `core`, `data`, `quantization`, `evaluation` and `utils` folders separate responsibilities; the last model/data stages contain no implemented algorithms yet. The status/check commands read configuration and report setup. Future workflow commands return `not_implemented` with exit code 2 and create no outputs. Each project uses its own environment.

Runtime dependencies are currently empty. Model frameworks and backend-specific dependencies will be selected during the compatibility probe. Private study records, environments, data, model artifacts and generated runs are ignored and excluded from distributions.

## Starting models

Use **[Google Gemma 4 E4B instruction-tuned](https://huggingface.co/google/gemma-4-E4B-it)** as the primary planned model. The family was released on [2 April 2026](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/). It provides a current model architecture for learning weight compression and document answering. Begin with text inputs; multimodal document understanding is a later extension.

| Role | Official checkpoint | Revision checked on 26 October 2026 |
| --- | --- | --- |
| Primary target | `google/gemma-4-E4B-it` | `ee0ef6023621cff504d758262d4e04895a5af4a2` |
| Smaller capacity fallback | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |

The publisher distinguishes effective computation from total parameters: E4B has about 4.5B effective parameters and 8B including its embedding tables; E2B has about 2.3B effective and 5.1B including embeddings. Budget using actual artifacts and measured memory. The architecture combines local and global attention, so later cache measurements must account for the actual layer structure rather than assuming a uniform cache per layer.

These are planned checkpoints, not locally tested models. The official Hugging Face model API supplied the revisions above; no weights were downloaded. At implementation kickoff, recheck current releases, then freeze one checkpoint, tokenizer, processor and chat template. A later model update receives a separate comparison record.

## What the application does

The user selects a document, asks a question and receives an answer with document/section references and supporting text. The application shows when the supplied material does not contain an answer. A comparison view exposes the floating-point reference and compressed variants, their outputs and their measured resource use.

Start with plain-text or Markdown documents with stable passage IDs. Later import formats need a separately checked parsing step. For the weight study, provide the same frozen passages to every model variant. Add retrieval only after this isolates generation errors from retrieval errors; the sibling [compressed search project](../compressed-semantic-search/README.md) can supply that component later.

## Learning through the implementation

Connect each compression change to a small worked calculation and the actual model tensors it affects. Cover scale and zero point, rounding and clipping, group size, outliers, calibration and error propagation. Then connect those mechanisms to changed answers and measured resource use. Routine implementation accompanies the explanations; Python syntax revision is unnecessary.

## Ordered build route

All phases are pending. Complete and review one bounded phase before expanding the next.

| Phase | Work | Evidence needed to continue |
| --- | --- | --- |
| 00 — Compatibility and budget | Pin the model/runtime, inspect available RAM/device memory, validate conversion/chat behavior and run a small reference probe | A reproducible working reference and an explicit memory/context/output budget |
| 01 — Document contract | Prepare public documents, stable passages, questions, checked answers and evidence labels | Frozen calibration/development/final membership and a clear correctness rubric |
| 02 — Reference and first compression | Preserve floating-point behavior and produce one 8-bit artifact | Matched outputs, task scores, resource measurements and fresh-process reload |
| 03 — Low-bit study | Add a basic 4-bit recipe; inspect regressions against the reference | Artifact identities, actual tensor precision and per-question failures |
| 04 — Calibration and method | Compare one calibration change, then one supported advanced recipe | Matched comparisons that isolate the changed mechanism |
| 05 — Context memory | Hold weights fixed and study supported cache policies and input lengths | Measured cache/total memory, response quality and latency by length |
| 06 — Saved application | Serve the chosen artifact through a local interface and retain its comparison report | Repeatable document answering and model loading after restart |

## Quantization experiments

Start with floating point versus 8-bit versus 4-bit weights on the same source model. For llama.cpp, candidates include Q8_0 and Q4_K_M; those labels describe presets with metadata and potentially mixed tensor precision, not exact whole-model bits per weight. Record the actual artifact composition.

After that baseline, study general-text versus document-domain calibration, a bounded group-size/selected-precision change, and a supported calibration-aware method. [llama.cpp contains a Gemma 4 architecture implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma4.cpp); verify the chosen pinned build's conversion, per-layer embeddings and device execution before relying on it. The [Qwen quantization guide](https://qwen.readthedocs.io/en/stable/quantization/llama.cpp.html) explains the general importance-matrix workflow; it is a mechanism reference, not proof of Gemma compatibility.

[Optimum Intel/NNCF](https://huggingface.co/docs/optimum-intel/openvino/optimization) offers another candidate route for AWQ/GPTQ and mixed-precision studies. Check its support for the exact model before selecting that route. A requested option that the exporter skips does not establish that the method ran. Treat comparisons that change runtime or placement as deployment comparisons, separate from same-runtime compression studies.

KV-cache precision is a later independent variable. Pin the weight artifact first, then change only a cache policy supported by the selected architecture/backend. Begin with bounded input lengths; the publisher's context limit does not establish a locally usable context budget. [KIVI](https://github.com/jy-yuan/KIVI) is a reference for cache-quantization ideas, with its own implementation requirements.

## Data and evaluation

Use public documents and questions with checked answers and supporting passages, including cases where no answer is available. [LongBench](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md) is a candidate source of document-Q&A tasks. Declare any selected subset before evaluation. Keep calibration text, tuning labels and final-test labels separate, and retain document provenance and licenses.

Measure answer correctness, supporting-evidence correctness, unsupported answers, missing answers and truncations. Use a declared review rubric for ambiguous responses. Perplexity can diagnose a numerical shift, but it does not replace application evaluation.

Record model bytes, host/device peak memory, prompt-processing time, time to first token, generation throughput and end-to-end latency. Report conversion/calibration cost separately. Fix prompts, chat template, thinking mode, decoding policy, token budgets, batch/concurrency, runtime build and placement. Repeat timed runs and distinguish warm from cold execution. If the floating-point reference needs different placement, its quality comparison remains useful while its timing is reported separately.

Each run must retain source/model/data identities, the exact compression recipe, actual completed work, predictions and failures. Choose thresholds and the candidate using development results before opening final evaluation. Fresh-process loading must reproduce the saved candidate's documented behavior.

## Completion and next action

The result is a working local assistant, reloadable compressed artifacts, a reproducible comparison and an explanation of the quality/memory/latency tradeoff, including failures. All of those remain pending; a speedup or preserved quality is not assumed.

Next at implementation kickoff: recheck the model release and runtime, freeze the first small reference/8-bit comparison and its budget, then select backend-specific dependencies for that probe. The private setup records use the existing project-learning layout. This project is an ordinary folder tracked by `ai_projects`; it has no nested Git repository.

[Back to Quantization Lab](../README.md) · [Projects checkpoint](../../../docs/guide/resume.md)
