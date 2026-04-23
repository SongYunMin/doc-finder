from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class TagSearchState(TypedDict, total=False):
    query: str
    top_k: int
    review_status: str | None
    normalized_query: str
    query_variants: list[str]
    exact_candidates: list
    semantic_candidates: list
    results: list[dict[str, object]]


def build_tag_search_graph(repository, query_normalizer, embedding_service):
    def normalize_query(state: TagSearchState) -> TagSearchState:
        normalized = query_normalizer.normalize(state["query"])
        return {
            "normalized_query": normalized.normalized_query,
            "query_variants": normalized.variants,
        }

    def search_exact_tags(state: TagSearchState) -> TagSearchState:
        return {
            "exact_candidates": repository.search_by_normalized_tags(
                normalized_tags=state["query_variants"],
                review_status=state.get("review_status"),
                limit=state.get("top_k", 10),
            )
        }

    def search_embedding_candidates(state: TagSearchState) -> TagSearchState:
        # 동의어 정규화 전후 표현을 같이 써서 semantic 후보 recall이 너무 줄지 않게 한다.
        embedding = embedding_service.embed_text(" ".join(state["query_variants"]))
        return {
            "semantic_candidates": repository.search_by_embedding(
                embedding=embedding,
                review_status=state.get("review_status"),
                limit=max(state.get("top_k", 10) * 2, state.get("top_k", 10)),
            )
        }

    def merge_candidates(state: TagSearchState) -> TagSearchState:
        merged: dict[tuple[int, int, int], dict[str, object]] = {}

        for candidate in state.get("semantic_candidates", []):
            document = candidate.document
            merged[document.key] = {
                "unit_id": document.unit_id,
                "data_id": document.data_id,
                "image_id": document.image_id,
                "asset_path": document.asset_path,
                "preview_path": None,
                "matched_tags": [],
                "matched_display_tags": [],
                "confidence": document.confidence,
                "score": round(candidate.score, 6),
                "cms_ref": {
                    "unit_id": document.unit_id,
                    "data_id": document.data_id,
                },
                "_exact_hit": False,
            }

        for candidate in state.get("exact_candidates", []):
            document = candidate.document
            current = merged.get(document.key)
            # 객체명 검색은 exact 태그 hit가 semantic 후보보다 항상 위에 오도록 강하게 가중한다.
            boosted_score = round(2.0 + document.confidence, 6)
            if current is None:
                merged[document.key] = {
                    "unit_id": document.unit_id,
                    "data_id": document.data_id,
                    "image_id": document.image_id,
                    "asset_path": document.asset_path,
                    "preview_path": None,
                    "matched_tags": candidate.matched_tags,
                    "matched_display_tags": query_normalizer.display_tags(
                        candidate.matched_tags
                    ),
                    "confidence": document.confidence,
                    "score": boosted_score,
                    "cms_ref": {
                        "unit_id": document.unit_id,
                        "data_id": document.data_id,
                    },
                    "_exact_hit": True,
                }
                continue

            current["matched_tags"] = candidate.matched_tags
            current["matched_display_tags"] = query_normalizer.display_tags(
                candidate.matched_tags
            )
            current["score"] = max(float(current["score"]), boosted_score)
            current["_exact_hit"] = True

        ordered_results = sorted(
            merged.values(),
            key=lambda result: (
                not bool(result["_exact_hit"]),
                -float(result["score"]),
                -float(result["confidence"]),
            ),
        )

        for result in ordered_results:
            result.pop("_exact_hit", None)

        return {"results": ordered_results[: state.get("top_k", 10)]}

    workflow = StateGraph(TagSearchState)
    workflow.add_node("normalize_query", normalize_query)
    workflow.add_node("search_exact_tags", search_exact_tags)
    workflow.add_node("search_embedding_candidates", search_embedding_candidates)
    workflow.add_node("merge_candidates", merge_candidates)
    workflow.add_edge(START, "normalize_query")
    workflow.add_edge("normalize_query", "search_exact_tags")
    workflow.add_edge("search_exact_tags", "search_embedding_candidates")
    workflow.add_edge("search_embedding_candidates", "merge_candidates")
    workflow.add_edge("merge_candidates", END)
    return workflow.compile()
