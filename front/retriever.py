from langchain_chroma import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.self_query.chroma import ChromaTranslator

from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

from langchain.retrievers import EnsembleRetriever

import streamlit as st

openai_api_key = st.secrets['OPENAI_API_KEY']
LLM = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)

class RetrieverDatabase():

    metadata_field_info = [
        AttributeInfo(
            name="Team code",
            description="Unique code that the team has. alphabetical uppercase + double digit combination.",
            type="string",
        ),
        AttributeInfo(
            name="Title",
            description="the topic that the team studied/made",
            type="string",
        ),
        AttributeInfo(
            name="Teammate #1 name",
            description="A team member's name. name is two or three letters of Hangul.",
            type="string"
        ),
        AttributeInfo(
            name="Teammate #1 number",
            description="A team member's student number. The student number is four digits.",
            type="string"
        ),
        AttributeInfo(
            name="Teammate #2 name",
            description="A team member's name. name is two or three letters of Hangul.",
            type="string"
        ),
        AttributeInfo(
            name="Teammate #2 number",
            description="A team member's student number. The student number is four digits",
            type="string"
        ),
        AttributeInfo(
            name="Youtube link",
            description="A youtube video link from the team. The vido can be played by clicking on the link.",
            type="string"
        )
    ]

    examples = [
        (
            "A23 팀?",
            {
                "query": "작품 설명서",
                "filter": 'eq("Team code", "A23")',
            },
        ),
        (
            "이동윤은 뭐했어?",
            {
                "query": "작품 설명서",
                "filter": 'or(eq("Teammate #1 name", "이동윤"), eq("Teammate #2 name", "이동윤"))',
            },
        ),
        (
            "환경에 관한 주제로 연구한 팀을 알려줄래?",
            {
                "query": "환경에 관한 주제로 연구한 팀을 알려줄래?",
                "filter": "NO_FILTER",
            }   
        ),
        (
            "팀 번호가 B로 시작하는 프로젝트의 주제는 어떤 것이 있어?",
            {
                "query": "팀 번호가 B로 시작하는 프로젝트의 주제는 어떤 것이 있어?",
                "filter": "NO_FILTER",
            }
        ),
        (
            "머신러닝을 사용하지 않은 팀이 있을까?",
            {
                "query": "머신러닝을 사용하지 않은 팀이 있을까?",
                "filter": "NO_FILTER",
            }
        )
    ]

    def __init__(self, _vectorstore):
        self.vectorstore = _vectorstore

        # 문서 내용 설명과 메타데이터 필드 정보를 사용하여 쿼리 생성기 프롬프트를 가져옵니다.
        query_prompt = get_query_constructor_prompt(
            'Ocean ICT 대회에 참가한 팀의 작품 설명서.',
            self.metadata_field_info,
            examples=self.examples
        )

        # 구성 요소에서 구조화된 쿼리 출력 파서를 생성합니다.
        output_parser = StructuredQueryOutputParser.from_components()

        # 프롬프트, 언어 모델, 출력 파서를 연결하여 쿼리 생성기를 만듭니다.
        new_query_constructor = query_prompt | LLM | output_parser

        self.self_query_retriever = SelfQueryRetriever(
            query_constructor = new_query_constructor,
            vectorstore = self.vectorstore,
            structured_query_translator=ChromaTranslator(),
            search_kwargs={"k": 1}
        )

        self.vectorstore_retriever = _vectorstore.as_retriever()

        # 앙상블 retriever를 초기화합니다.
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.self_query_retriever, self.vectorstore_retriever],
            weights=[0.5, 0.5],
            search_type="mmr",
        )

    def get_self_query_retriever(self):
        return self.self_query_retriever
    
    def get_vectorstore_retriever(self):
        return self.vectorstore_retriever
    
    def get_ensemble_retriever(self):
        return self.ensemble_retriever