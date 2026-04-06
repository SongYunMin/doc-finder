# Main Server Entrypoint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `python -m doc_finder.main` start the FastAPI server and move the old CLI smoke check into a separate module.

**Architecture:** Keep the existing FastAPI app in `doc_finder.app` and change `doc_finder.main` into a thin `uvicorn` launcher. Preserve the old search smoke-check behavior by moving it into a dedicated `doc_finder.cli` module so server and CLI concerns are separated cleanly.

**Tech Stack:** Python 3.13, FastAPI, Uvicorn, LangGraph, pytest

---

### Task 1: Add failing tests for the new entrypoints

**Files:**
- Create: `tests/test_main.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing test**

Add tests that require:
- `doc_finder.main.main()` to call `uvicorn.run("doc_finder.app:app", ...)`
- `doc_finder.cli.main()` to print the text search smoke-check output

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_main.py tests/test_cli.py -q`
Expected: FAIL because the new behavior and module do not exist yet

### Task 2: Implement the split entrypoints

**Files:**
- Modify: `src/doc_finder/main.py`
- Create: `src/doc_finder/cli.py`
- Modify: `README.md`

**Step 1: Write minimal implementation**

Implement:
- server launcher in `main.py`
- old CLI behavior in `cli.py`
- README examples for both commands

**Step 2: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest -q`
Expected: PASS
