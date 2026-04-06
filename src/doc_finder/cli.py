import argparse

from doc_finder.schemas.search import TextSearchRequest
from doc_finder.services.search_service import SearchService


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the text search CLI smoke check.")
    parser.add_argument(
        "--query",
        default="langgraph basic setup",
        help="Query text to push through the search service.",
    )
    args = parser.parse_args(argv)

    service = SearchService()
    result = service.search(TextSearchRequest(query=args.query))
    print(result.model_dump())


if __name__ == "__main__":
    main()
