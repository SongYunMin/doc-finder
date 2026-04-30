TAGGING_PROMPT = (
    "이미지를 검색용 태그로 분석해라. "
    "반드시 JSON object 하나만 반환한다. "
    "Markdown 코드블록, 설명, 중복 JSON, 기타 텍스트를 출력하지 않는다. "
    "필드: keyword_tags, normalized_tags, confidence, review_status. "
    "keyword_tags는 이미지에서 확인 가능한 검색 키워드만 넣는다. "
    "normalized_tags는 한국어 canonical tag를 우선한다. "
    "review_status는 approved 또는 pending 중 하나만 사용한다."
)
