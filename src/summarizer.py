from transformers import pipeline

# HuggingFace 요약 파이프라인 로드
# (처음 실행 시 모델 다운로드됨)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def summarize_text(text: str) -> str:
    """
    입력된 긴 문장을 요약해서 문자열로 반환
    """
    result = summarizer(
        text,
        max_length=130,
        min_length=30,
        do_sample=False
    )
    return result[0]["summary_text"]