# Hugging Face Florence-2 블로그 적응 메모

## 원문 기준

참고한 원문은 Hugging Face 블로그의 `Fine-tuning Florence-2`다.

원문 핵심:
- Florence-2는 이미지 + 텍스트 입력을 받아 텍스트를 생성하는 sequence-to-sequence 모델이다.
- 블로그 예제는 `DocVQA`를 대상으로 한다.
- 예제에서는 질문 앞에 `<DocVQA>` prefix를 붙인다.
- 제한된 자원 실험에서는 vision encoder를 freeze하고 작은 learning rate를 사용했다.

## 현재 프로젝트와의 차이

현재 목표는 `문서 질의응답`이 아니다.  
현재 목표는 `기하 이미지 -> 표준 태그 문자열`이다.

즉 차이는 이렇다.

- 원문 task: 질문-답변
- 현재 task: 태그 문자열 생성
- 원문 label: 정답 문장 또는 짧은 답
- 현재 label: 표준화된 태그 목록
- 원문 metric: DocVQA similarity
- 현재 metric: exact match, tag precision/recall/F1

## 그대로 가져갈 것

- 이미지 + 텍스트 prompt -> 텍스트 target 구조
- `AutoProcessor` 기반 배치 구성
- 작은 learning rate로 시작하는 보수적 전략
- 자원이 부족할 때 작은 batch size부터 시작하는 방식

## 조정해야 할 것

### 1. Prompt prefix

원문:

```text
<DocVQA>
```

현재 추천:

```text
<SearchTag>
```

### 2. Target text 형식

원문:

```text
The answer is 42
```

현재 추천:

```text
triangle; point_a; point_b; angle_label
```

### 3. 평가 기준

원문 metric은 현재 작업에 맞지 않는다.  
현재는 tag set 기준의 정밀도와 재현율이 더 중요하다.

## 경계할 점

- 블로그 코드를 그대로 가져오면 `DocVQA` 문제 설정이 섞인다.
- 현재 프로젝트에서는 기존 `keyword_tags`를 gold label로 착각하면 안 된다.
- 현재 목표는 “자연스러운 설명문”이 아니라 “검색에 쓸 수 있는 표준 태그 문자열”이다.

## 시작점 추천

- 체크포인트: `microsoft/Florence-2-base-ft`
- 이유:
  - 실험 회전이 빠르다.
  - 태스크 적합성 검증에 충분하다.
  - 초반에는 모델 크기보다 라벨 품질 영향이 더 크다.

## 다음 단계

이 문서를 읽은 뒤 바로 해야 하는 것은 아래다.

1. `label_taxonomy.example.yaml` 초안 채우기
2. `searchtag_dataset.sample.jsonl` 형식으로 20장 샘플 만들기
3. zero-shot baseline 측정
4. smoke fine-tuning 설계
