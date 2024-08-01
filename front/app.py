import streamlit as st
from db import GooglesheetUtils

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_chroma import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

from langchain.chains import RetrievalQA
openai_api_key = st.secrets['OPENAI_API_KEY']

# Set OpenAI API key
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5, openai_api_key=openai_api_key)

# Function to setup RAG pipeline
@st.cache_resource
def setup_rag_pipeline(_retriever):
    prompt = PromptTemplate.from_template(
    """당신은 부산과학고등학교의 행사 "Ocean ICT"의 도우미 챗봇인 "한바다" 입니다.
    검색된 정보를 사용하여 질문에 답합니다.
    
    팀에 대한 정보를 언급할 때 반드시 팀 코드를 같이 언급하세요.
    팀 코드는 대문자와 숫자 두 자리의 조합입니다.

    답을 모른다면 그냥 너의 정보와 함께 너는 Ocean ICT에 대해서만 답변할 수 있다고 말하면 됩니다.
    답을 안다면 있는 정보를 사용해 최대한 자세하게 답변할 수 있도록 합니다. 여러 줄에 걸쳐서 답변하세요.
    한국어로 친절하고, 친근하게 답하세요.

    #질문:
    {question}
    #정보:
    {context}

    #답변:"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_retriever,
    )

    return qa_chain


# Streamlit UI
st.title("한바다 🐋")
st.header("2024 Ocean ICT 챗봇 도우미")

vectorstore = Chroma(
    persist_directory="db/chroma_2023_pdfs",
    embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key)
)

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

# 문서 내용 설명과 메타데이터 필드 정보를 사용하여 쿼리 생성기 프롬프트를 가져옵니다.
prompt = get_query_constructor_prompt(
    'Ocean ICT 대회에 참가한 팀의 작품 설명서.',
    metadata_field_info,
    examples=examples
)

# 구성 요소에서 구조화된 쿼리 출력 파서를 생성합니다.
output_parser = StructuredQueryOutputParser.from_components()

# 프롬프트, 언어 모델, 출력 파서를 연결하여 쿼리 생성기를 만듭니다.
new_query_constructor = prompt | llm | output_parser

self_query_retriever = SelfQueryRetriever(
    query_constructor=new_query_constructor,
    vectorstore=vectorstore,
    structured_query_translator=ChromaTranslator()
)

from langchain.retrievers import EnsembleRetriever

# 앙상블 retriever를 초기화합니다.
ensemble_retriever = EnsembleRetriever(
    retrievers=[self_query_retriever, vectorstore.as_retriever()],
    weights=[0.5, 0.5],
    search_type="mmr",
)

# Setup RAG pipeline
qa_chain = setup_rag_pipeline(ensemble_retriever)

googlesheet = GooglesheetUtils()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = qa_chain.invoke(prompt)
        st.markdown(response['result'])

    st.session_state.messages.append({"role": "assistant", "content": response['result']})

    values = [[prompt, response['result']]]
    print(values)
    googlesheet.append_data(values, 'Sheet1!A1')