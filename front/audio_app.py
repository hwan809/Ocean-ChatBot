import streamlit as st
from db import GooglesheetUtils
from loc_image import get_location_image
import datetime
import asyncio
from typing import AsyncIterable
import numpy as np

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

from RealtimeTTS import TextToAudioStream, GTTSEngine
import sounddevice as sd

openai_api_key = st.secrets['OPENAI_API_KEY']

# Set OpenAI API key
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)

# Function to setup RAG pipeline
@st.cache_resource
def setup_rag_pipeline(_retriever):
    prompt = PromptTemplate.from_template(
    """당신은 부산과학고등학교의 행사 "Ocean ICT"의 도우미 챗봇인 "한바다" 입니다.
    검색된 정보를 사용하여 질문에 답합니다.
    
    팀에 대한 정보를 언급할 때 반드시 팀 코드를 같이 언급하세요.
    팀 코드는 대문자와 숫자 두 자리의 조합입니다.

    답을 모른다면 그냥 당신의 정보에 대해 언급하고,
    Ocean ICT에 대해서만 답변할 수 있다고 말하면 됩니다.
    
    절대로 유튜브 링크를 사용자에게 공유하지 말고, 아래 동영상을 참조해달라고 하세요.
    
    답을 안다면 있는 정보를 사용해 최대한 자세하게 답변할 수 있도록 하되, 자신의 소개는 할 필요가 없습니다. 여러 줄에 걸쳐서 답변하세요.
    한국어로 친절하고, 친근하게 답하십시오.

    #질문:
    {question}
    #정보:
    2023년의 Ocean ICT에는 총 86팀이 참가하였다. 다음은 참가한 팀들의 포스터 중 질문과 관계된 일부이다.
    {context}

    #답변:"""
    )

    chain = prompt | llm | StrOutputParser()

    return chain

# Streamlit UI
st.title("한바다 🐬")
st.header("2024 Ocean ICT 챗봇 도우미")

vectorstore = Chroma(
    persist_directory="db/chroma_2023_pdfs_new",
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

query_prompt = get_query_constructor_prompt(
    'Ocean ICT 대회에 참가한 팀의 작품 설명서.',
    metadata_field_info,
    examples=examples
)

output_parser = StructuredQueryOutputParser.from_components()

new_query_constructor = query_prompt | llm | output_parser

self_query_retriever = SelfQueryRetriever(
    query_constructor=new_query_constructor,
    vectorstore=vectorstore,
    structured_query_translator=ChromaTranslator()
)

from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[self_query_retriever, vectorstore.as_retriever()],
    weights=[0.5, 0.5],
    search_type="mmr",
)

qa_chain = setup_rag_pipeline(ensemble_retriever)
googlesheet = GooglesheetUtils()

youtube_link = ''

async def generate_response_and_audio(prompt: str, docs) -> AsyncIterable[tuple[str, np.ndarray]]:
    engine = GTTSEngine()
    audio_stream = TextToAudioStream(engine)
    
    response_stream = qa_chain.stream({"context": docs, "question": prompt})
    
    full_response = ""
    async for token in response_stream:
        full_response += token
        audio_chunk = audio_stream.feed(token)
        if audio_chunk is not None:
            yield token, audio_chunk

async def play_audio(audio_queue: asyncio.Queue):
    while True:
        chunk = await audio_queue.get()
        if chunk is None:  # None은 재생 종료를 의미합니다
            break
        sd.play(chunk, samplerate=22050)
        sd.wait()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
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

async def main():
    prompt = st.chat_input("질문을 입력하세요")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message(name="assistant", avatar='🐋'):
            docs = ensemble_retriever.invoke(prompt)
            response_placeholder = st.empty()
            full_response = ""
            
            audio_queue = asyncio.Queue()
            audio_task = asyncio.create_task(play_audio(audio_queue))
            
            async for token, audio_chunk in generate_response_and_audio(prompt, docs):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
                await audio_queue.put(audio_chunk)
            
            response_placeholder.markdown(full_response)
            await audio_queue.put(None)  # 오디오 재생 종료 신호
            await audio_task

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        youtube_link = docs[0].metadata['Youtube link']
        team_code = docs[0].metadata['Team code']

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        values = [[prompt, full_response, timestamp]]
        googlesheet.append_data(values, 'Sheet1!A1')

        col1, col2, col3 = st.columns([1, 1, 3])

        with col1:
            if st.button('팀 영상 보기'):
                st.session_state.messages.append({"role": "video", "content": youtube_link})
                st.experimental_rerun()
        with col2:
            if st.button('팀 위치 보기'):
                st.session_state.messages.append({"role": "image", "content": get_location_image(team_code)})
                st.experimental_rerun()

if __name__ == "__main__":
    asyncio.run(main())
