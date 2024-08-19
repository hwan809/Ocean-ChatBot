import streamlit as st

from db import GooglesheetUtils

from loc_image import get_location_image
from retriever import CustomRetriever
from yeardistribution import YearDistribution

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
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)

@st.cache_resource
def setup_rag_pipeline():
    prompt = PromptTemplate.from_template(
    """당신은 부산과학고등학교의 행사 "Ocean ICT"의 도우미 챗봇인 "한바다" 입니다.
    "한바다"는 부산과학고 정보 R&E 학생들이 만들었습니다. 대표적으로 김환, 서재원, 김석현이 있습니다.
    검색된 정보를 사용하여 질문에 답합니다.

    답을 모른다면 그냥 당신의 정보에 대해 언급하고, Ocean ICT에 대해서만 답변할 수 있다고 말하면 됩니다.
    답을 안다면 주어진 정보를 사용해 답변합니다. 즉, 팀명, 탐구 내용, 팀 코드 등을 포함해서 답변합니다.
    답변 뒤에는 답을 도출하는 데 직접적으로 사용되는 문서의 팀 코드 목록을 문자 '|'로 구분해 안내합니다.
    출처가 없는 경우 '|' 을 사용하지 않습니다. 하나 이상의 출처가 있는 경우 맨 뒤에 한꺼번에 표시하세요. 답변의 끝에는 '|'을 사용하지 않습니다.
    출처를 표시하고 난 후에는 어떠한 글도 추가하지 않습니다.

    예시 답변 1: 1. B03 팀과 2. A11 팀이 있습니다. 이와 같이 다양한 팀이 참가합니다. | B03 | A11
    예시 답변 2: C05 팀은.. | C05

    #질문:
    {question} 자세하게 답변해줘.
    #정보:
    2024년에 열린 제 7회 Ocean ICT에는 총 96팀이 참가하였다. 다음은 참가한 팀들의 포스터 중 질문과 관계된 일부이다.
    {context}

    #답변:"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain

def find_document(docs, team_code, now_year):
    for doc in docs:
        if doc.metadata['Team code'] == team_code and \
            doc.metadata['Year'] == now_year:
            return doc
    return None

st.title("한바다 🐬")
st.header("2024 Ocean ICT 챗봇 도우미")

vectorstore = Chroma(
    persist_directory="db/chroma_2024_pdfs",
    embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key)
)
vectorstore_old = Chroma(
    persist_directory="db/chroma_19to23_pdfs",
    embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key)
)

retriever = CustomRetriever(vectorstore)
retriever_old = CustomRetriever(vectorstore_old)

qa_chain = setup_rag_pipeline()
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
    elif message["role"] == "user":
        with st.chat_message(name="user"):
            st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        now_retriever = None
        find_year = YearDistribution("gpt-4o")
        now_year = find_year.Year(prompt).replace('\n', '').strip()

        print(now_year)

        if now_year != '2024':
            now_retriever = retriever_old.get_ensemble_retriever()
        else:
            now_retriever = retriever.get_ensemble_retriever()
        docs = now_retriever.invoke(prompt)
        stream = qa_chain.stream(
            {
                "context": docs,
                "question": prompt
            }
        )

        with st.chat_message(name="assistant", avatar='🐋'):
            response = st.write_stream(stream)

        used_team_code = [i.strip() for i in response.split('|')[1:]]
        st.session_state.messages.append({"role": "assistant", "content": response})

        if len(used_team_code) == 1 and 'None' not in used_team_code:
            used_doc = find_document(docs, used_team_code[0], now_year)
            used_doc_vid = used_doc.metadata['Youtube link']
            show_loc_img = lambda: st.session_state.messages.append({"role": "image", "content": get_location_image(used_team_code[0])})

            st.video(used_doc_vid)
            st.session_state.messages.append({"role": "video", "content": used_doc_vid})

            col1, col2 = st.columns([1, 4])
            if now_year == '2024':
                with col1:
                    st.button('팀 위치 보기', on_click=show_loc_img)
            
        now = datetime.now() + timedelta(hours=9)
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        values = [[prompt, response, timestamp]]
        googlesheet.append_data(values, 'Sheet1!A1')
    except Exception as e:
        with st.chat_message(name="assistant", avatar='🐋'):
            response = st.markdown('다시 시도해 주세요!\n')
            st.markdown(e)
