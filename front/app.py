import streamlit as st
from db import GooglesheetUtils
from loc_image import get_location_image
from datetime import datetime, timedelta

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_chroma import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

openai_api_key = st.secrets['OPENAI_API_KEY']

# Set OpenAI API key
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)

# Function to setup RAG pipeline
@st.cache_resource
def setup_rag_pipeline(_retriever):
    prompt = PromptTemplate.from_template(
    """당신은 부산과학고등학교의 행사 "Ocean ICT"의 도우미 챗봇인 "한바다" 입니다.
    검색된 정보를 사용하여 질문에 답합니다.
    팀에 대한 정보를 언급할 때 반드시 팀 코드를 같이 언급하세요.
    답을 모른다면 그냥 당신의 정보에 대해 언급하고, Ocean ICT에 대해서만 답변할 수 있다고 말하면 됩니다.
    절대로 유튜브 링크를 사용자에게 공유하지 말고, 아래 동영상을 참조해달라고 하세요.

    Ocean ICT는 ~~ 
    
    답을 안다면 1. 있는 정보를 사용한 답과, 2. 답을 도출하는 데 직접적으로 사용되는 문서의 팀 코드 목록을 문자 '|'로 구분해 안내합니다. 
    없으면 None으로 표시합니다. 하나 이상의 출처가 있는 경우 맨 뒤에 한꺼번에 표시하세요. 
    답변의 끝에는 '|'을 사용하지 않습니다.
    예시 답변: B03 팀과 A11 팀이 있습니다. | B03 | A11

    #질문:
    {question}
    #정보:
    2023년의 Ocean ICT에는 총 86팀이 참가하였다. 다음은 참가한 팀들의 포스터 중 질문과 관계된 일부이다.
    {context}

    #답변:"""
    )
    
    #예시) {'answer': 'B03 팀은..', 'source': 'B03'}

    chain = prompt | llm | StrOutputParser()

    return chain

def find_document(docs, team_code):
    for doc in docs:
        if doc.metadata['Team code'] == team_code:
            return doc
    return None

# Streamlit UI
st.title("한바다 🐬")
st.header("2024 Ocean ICT 챗봇 도우미")

vectorstore = Chroma(
    persist_directory="db/chroma_2024_pdfs",
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

# 문서 내용 설명과 메타데이터 필드 정보를 사용하여 쿼리 생성기 프롬프트를 가져옵니다.
query_prompt = get_query_constructor_prompt(
    'Ocean ICT 대회에 참가한 팀의 작품 설명서.',
    metadata_field_info,
    examples=examples
)

# 구성 요소에서 구조화된 쿼리 출력 파서를 생성합니다.
output_parser = StructuredQueryOutputParser.from_components()

# 프롬프트, 언어 모델, 출력 파서를 연결하여 쿼리 생성기를 만듭니다.
new_query_constructor = query_prompt | llm | output_parser

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

used_doc_vid = ''

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for i in range(len(st.session_state.messages)):
    message = st.session_state.messages[i]
    if message["role"] == "assistant":
        with st.chat_message(name="assistant", avatar='🐋'):
            st.markdown(message["content"])
    elif message["role"] == "video":
        with st.chat_message(name="assistant", avatar='🐋'):
            st.video(message["content"])    
    elif message["role"] == "image":
        with st.chat_message(name="assistant", avatar='🐋'):
            st.image(message["content"], width=360)    
    else:
        with st.chat_message(name="user"):
            st.markdown(message["content"])
        

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message(name="assistant", avatar='🐋'):
        docs = ensemble_retriever.invoke(prompt)
        stream = qa_chain.stream(
            {
                "context": docs,
                "question": prompt
            }
        )
        response = st.write_stream(stream)

    used_team_code = [i.strip() for i in response.split('|')[1:]]

    if len(used_team_code) == 1 and 'None' not in used_team_code:
        used_doc = find_document(docs, used_team_code[0])
        used_doc_vid = used_doc.metadata['Youtube link']

        play_video = lambda: st.session_state.messages.append({"role": "video", "content": used_doc_vid})
        show_loc_img = lambda: st.session_state.messages.append({"role": "image", "content": get_location_image(used_team_code)})
        
        col1, col2, col3 = st.columns([1, 1, 3])

        with col1:
            st.button('팀 영상 보기', on_click=play_video)
        with col2:
            st.button('팀 위치 보기', on_click=show_loc_img)

    st.session_state.messages.append({"role": "assistant", "content": response})        
    now = datetime.now() + timedelta(hours=9)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    values = [[prompt, response, timestamp]]
    googlesheet.append_data(values, 'Sheet1!A1')