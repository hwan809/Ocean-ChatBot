import os
import streamlit as st

KEY = st.secrets['OPENAI_API_KEY']

from openai import OpenAI

class YearDistribution:
    def __init__(self, model) -> None:
        self.model = model
        self.client = OpenAI(api_key=KEY)

    def Year(self, question) -> None:
        completion = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system",
            "content": '''지금 시점은 2024년이고 내가 프롬프트를 입력하면 요구하는 질문의 시점(연도, 년 빼고)만 답변해줘
            일반적으로 질문의 시점은 2024년이고, 시간에 관한 키워드가 있는 경우 2024년보다 이전 년도를 답변할 수 있어.
            예시 1) 질문: DO3은 어떤 팀이야? 답변: 2024
            예시 2) 질문: 제작년의 행사는 어떤 팀들이 나왔어? 답변: 2022
            예시 3) 질문: 김환의 팀은? 답변: 2024'''},
            {"role": "user", "content": question}
        ]
        )
        return completion.choices[0].message.content