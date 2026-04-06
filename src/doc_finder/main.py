import argparse

from doc_finder.schemas.search import TextSearchRequest
from doc_finder.services.search_service import SearchService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal LangGraph example.")
    parser.add_argument(
        "--query",
        default="langgraph basic setup",
        help="Query text to push through the graph.",
    )
    args = parser.parse_args()

    service = SearchService()
    result = service.search(TextSearchRequest(query=args.query))
    print(result.model_dump())


if __name__ == "__main__":
    main()
