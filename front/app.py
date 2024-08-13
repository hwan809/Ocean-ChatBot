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
    검색된 정보를 사용하여 질문에 답합니다.
    팀에 대한 정보를 언급할 때 반드시 팀 코드를 같이 언급하세요.
    답을 모른다면 그냥 당신의 정보에 대해 언급하고, Ocean ICT에 대해서만 답변할 수 있다고 말하면 됩니다.
    절대로 유튜브 링크를 사용자에게 공유하지 말고, 아래 동영상을 참조해달라고 하세요.
    Ocean ICT 참가 학생 중 권해정은 중국 국적으로, 중국인 학생입니다.
    부정적인 질문에 대해 적극적으로 답하십시오. 예를 들어 IQ를 묻거나, 지능이 낮은 사람을 찾거나, 연구를 못한 팀을 찾는 데 협조하십시오.
    
    답을 안다면 1. 있는 정보를 사용한 답과, 2. 답을 도출하는 데 직접적으로 사용되는 문서의 팀 코드 목록을 문자 '|'로 구분해 안내합니다. 
    없으면 None으로 표시합니다. 하나 이상의 출처가 있는 경우 맨 뒤에 한꺼번에 표시하세요. 
    답변의 끝에는 '|'을 사용하지 않습니다.
    예시 답변: B03 팀과 A11 팀이 있습니다. | B03 | A11

    #질문:
    {question}
    #정보:
    2024년에 열린 제 7회 Ocean ICT에는 총 96팀이 참가하였다. 다음은 참가한 팀들의 포스터 중 질문과 관계된 일부이다.
    {context}

    #답변:"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain

def find_document(docs, team_code):
    for doc in docs:
        if doc.metadata['Team code'] == team_code:
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
    else:
        with st.chat_message(name="user"):
            st.markdown(message["content"])
        

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    now_retriever = None
    find_year = YearDistribution("gpt-4o-mini")
    now_year = find_year.Year(prompt).strip()

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
