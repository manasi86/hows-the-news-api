---
icon: material/newspaper
status: new
---

# `hows-the-news` User Guide

[![Python Blueprint][python-blueprint-badge]](https://github.com/johnthagen/python-blueprint)
[python-blueprint-badge]: https://img.shields.io/badge/%F0%9F%97%BA%EF%B8%8F-python--blueprint-2dcf59.svg

!!! info

    `hows-the-news` is a FastAPI service that checks whether a piece of text is
    news content, summarizes it, and analyzes the sentiment of the summary
    (positive, negative, or neutral) using a FreeLLM-backed chat completions API.

## Installation

First, [install `uv`](https://docs.astral.sh/uv/getting-started/installation):

=== "macOS and Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

Then install the `hows-the-news` package and its dependencies:

```bash
uv sync
```

## Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Set your FreeLLM API key in `.env`:

```dotenv
LLM_API_KEY=your-lm-api-key-here
LLM_BASE_URL=https://llm.nalits.com/v1
LLM_MODEL=auto
```

## Quick Start

Run the API server:

```bash
uv run uvicorn api.main:app --reload
```

Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

### Summarize

```bash
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "The company announced record profits today."}'
```

Returns whether the text is news content and, if so, a summary.

### Analyse

```bash
curl -X POST http://127.0.0.1:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "The company announced record profits today."}'
```

Returns the sentiment of the text summary: `positive`, `negative`, or `neutral`.

### Health

```bash
curl http://127.0.0.1:8000/health
```

## Architecture

``` {.sourceCode .}
src
├── api
│   ├── main.py
│   ├── schemas.py
│   └── routes
│       ├── analyse.py
│       └── summarize.py
└── lib
    ├── config.py
    ├── llm.py
    ├── news.py
    ├── prompts.py
    └── sentiment.py
```

The `api` package exposes the HTTP endpoints and request/response schemas, while
all business logic lives in the `lib` package. The LLM client in `lib/llm.py`
calls the FreeLLM-compatible chat completions endpoint
(`{LLM_BASE_URL}/chat/completions`) using `LLM_API_KEY` for authentication.
