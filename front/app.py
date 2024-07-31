import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "ㄴㅁㅇㄻㄴㅇㄹ"  # 실제 사용 시 안전한 방법으로 API 키를 설정해야 합니다

# Function to load and process PDF
@st.cache_resource
def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore


# Function to setup RAG pipeline
@st.cache_resource
def setup_rag_pipeline(_vectorstore):
    prompt_template = """다음 맥락을 사용하여 주어진 질문에 대해 한국어로 답변해주세요. 
    만약 답을 모르겠다면, 모른다고 솔직히 말하고 추측하지 마세요.

    {context}

    질문: {question}
    한국어 답변:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )

    return qa_chain


# Streamlit UI
st.title("학사 관리 챗봇")

# Load and process the pre-saved PDF
pdf_path = "data/academic_rules.pdf"
vectorstore = load_and_process_pdf(pdf_path)

# Setup RAG pipeline
qa_chain = setup_rag_pipeline(vectorstore)

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