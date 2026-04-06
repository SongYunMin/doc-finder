# FastAPI-First Text Search Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the project into a FastAPI-first server structure where LangGraph is an internal orchestration layer for text search.

**Architecture:** Move the current single-file API into layered modules: `app` for server bootstrapping, `api` for routers, `schemas` for request/response models, `services` for application logic, and `graphs` for LangGraph workflows. Keep behavior intentionally simple: health check plus text search endpoint backed by one LangGraph graph.

**Tech Stack:** Python 3.13, FastAPI, LangGraph, pytest

---

### Task 1: Add failing tests for the new structure

**Files:**
- Modify: `tests/test_api.py`
- Create: `tests/test_search_service.py`

**Step 1: Write the failing test**

Add tests that require:
- `doc_finder.app:app` as the FastAPI entrypoint
- `POST /search/text` returning `query` and `answer`
- `SearchService.search()` returning a validated response model

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_api.py tests/test_search_service.py -q`
Expected: FAIL because the new modules do not exist yet

### Task 2: Implement the FastAPI-first layout

**Files:**
- Create: `src/doc_finder/app.py`
- Create: `src/doc_finder/api/router.py`
- Create: `src/doc_finder/api/routes/health.py`
- Create: `src/doc_finder/api/routes/search.py`
- Create: `src/doc_finder/schemas/search.py`
- Create: `src/doc_finder/services/search_service.py`
- Create: `src/doc_finder/graphs/search_graph.py`
- Modify: `README.md`
- Delete or stop using: `src/doc_finder/api.py`, `src/doc_finder/graph.py`

**Step 1: Write minimal implementation**

Implement:
- a single FastAPI app in `app.py`
- router composition in `api/router.py`
- `GET /health`
- `POST /search/text`
- `SearchService`
- one LangGraph graph module for text search

**Step 2: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest -q`
Expected: PASS

### Task 3: Verify the HTTP server entrypoint

**Files:**
- Modify: `README.md`

**Step 1: Run the server**

Run: `./.venv/bin/python -m uvicorn doc_finder.app:app --host 127.0.0.1 --port 8010`

**Step 2: Verify endpoints**

Run:
- `curl http://127.0.0.1:8010/health`
- `curl -X POST http://127.0.0.1:8010/search/text -H 'Content-Type: application/json' -d '{"query":"hello"}'`

Expected:
- `{"status":"ok"}`
- `{"query":"hello","answer":"hello"}`
