# FastAPI Wrapper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimal FastAPI layer that exposes the existing LangGraph through HTTP.

**Architecture:** Keep the current LangGraph graph unchanged and add a thin API module that owns routing and request/response validation. Expose only a health endpoint and a graph invocation endpoint so the server surface stays small and easy to extend.

**Tech Stack:** Python 3.13, FastAPI, LangGraph, pytest

---

### Task 1: Add failing API tests

**Files:**
- Modify: `tests/test_graph.py`
- Create: `tests/test_api.py`

**Step 1: Write the failing test**

Add tests for:
- `GET /health` returns `200` and `{"status": "ok"}`
- `POST /invoke` returns `200` and mirrors the query into `answer`

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_api.py -q`
Expected: FAIL because the FastAPI app module does not exist yet

### Task 2: Implement the FastAPI wrapper

**Files:**
- Modify: `pyproject.toml`
- Create: `src/doc_finder/api.py`
- Modify: `README.md`

**Step 1: Write minimal implementation**

Add:
- `fastapi`
- `uvicorn`

Create:
- FastAPI app factory
- `GET /health`
- `POST /invoke`

Reuse the existing `build_graph()` function directly.

**Step 2: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest -q`
Expected: PASS

### Task 3: Verify the server entrypoint

**Files:**
- Modify: `README.md`

**Step 1: Run the app locally**

Run: `./.venv/bin/python -m uvicorn doc_finder.api:app --host 127.0.0.1 --port 8000`

**Step 2: Verify the endpoint**

Run: `curl http://127.0.0.1:8000/health`
Expected: `{"status":"ok"}`
