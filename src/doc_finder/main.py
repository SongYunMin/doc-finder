import argparse

from doc_finder.graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal LangGraph example.")
    parser.add_argument(
        "--query",
        default="langgraph basic setup",
        help="Query text to push through the graph.",
    )
    args = parser.parse_args()

    graph = build_graph()
    result = graph.invoke({"query": args.query})
    print(result)


if __name__ == "__main__":
    main()
