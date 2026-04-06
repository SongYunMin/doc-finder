from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict, total=False):
    query: str
    answer: str


def _copy_query_to_answer(state: GraphState) -> GraphState:
    return {"answer": state["query"]}


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("copy_query_to_answer", _copy_query_to_answer)
    workflow.add_edge(START, "copy_query_to_answer")
    workflow.add_edge("copy_query_to_answer", END)
    return workflow.compile()
