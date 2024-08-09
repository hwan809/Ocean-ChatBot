import os

KEY = os.environ.get("OPENAI_API_KEY")

from openai import OpenAI

class YearDistribution:
    
    def __init__(self, model, KEY) -> None:
        self.model = model
        self.client = OpenAI(api_key = KEY)

    def Year(self, question) -> None:
        completion = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system",
            "content": '''지금 시점은 2024년이고 내가 프롬프트를 입력하면 요구하는 질문의 시점(연도, 년 빼고)만 답변해줘
            예시) 질문: 제작년의 행사는 어떤 팀들이 나왔어? 답변: 2022'''},
            {"role": "user", "content": question}
        ]
        )
        return completion.choices[0].message.content