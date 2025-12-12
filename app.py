from src.summarizer import summarize_text

def main():
    print("=" * 50)
    print(" HuggingFace 기반 문서 요약기 ")
    print("=" * 50)
    print("문장을 입력하면 요약 결과를 출력합니다.")
    print("종료하려면 그냥 엔터를 누르세요.\n")

    while True:
        text = input("요약할 문장을 입력하세요:\n> ")

        if text.strip() == "":
            print("프로그램을 종료합니다.")
            break

        summary = summarize_text(text)
        print("\n[요약 결과]")
        print(summary)
        print("-" * 50)

if __name__ == "__main__":
    main()