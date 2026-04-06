from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TextSearchState(TypedDict, total=False):
    query: str
    answer: str


def _copy_query_to_answer(state: TextSearchState) -> TextSearchState:
    return {"answer": state["query"]}


def build_text_search_graph():
    workflow = StateGraph(TextSearchState)
    workflow.add_node("copy_query_to_answer", _copy_query_to_answer)
    workflow.add_edge(START, "copy_query_to_answer")
    workflow.add_edge("copy_query_to_answer", END)
    return workflow.compile()
