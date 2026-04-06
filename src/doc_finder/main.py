import argparse

import uvicorn


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the doc-finder API server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable autoreload for development.",
    )
    args = parser.parse_args(argv)

    uvicorn.run(
        "doc_finder.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
