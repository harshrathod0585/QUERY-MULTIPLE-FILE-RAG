import streamlit as st
import os 
from dotenv import load_dotenv

load_dotenv()
os.environ['GROQ_API_KEY']=st.secrets['GROQ_API_KEY']
groq_api_key = st.secrets['GROQ_API_KEY']

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

llm = ChatGroq(model="Llama3-8b-8192",api_key=groq_api_key)

st.title("Nexora File Reader")
st.write("Upload Files and Get Answer ")
st.sidebar.write("Session_id")
session_id=st.sidebar.text_input("Enter_Session_id",value="gen-ai")
text_input = st.text_input("Ask Anything Regarding Uploaded Documents")
upload_files = st.file_uploader("Browse File",accept_multiple_files=True)

if upload_files : 
    documents = []
    for upload_file in upload_files:
        temppdf=f"./ext.pdf"
        with open(temppdf,"wb") as file:
            file.write(upload_file.getvalue())
            file_name = upload_file.name
        docs = PyPDFLoader(temppdf).load()
        documents.extend(docs)
    
    #vectore store
    text_split = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    split_document = text_split.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    vectorestore = FAISS.from_documents(split_document,embeddings)

    retrieval = vectorestore.as_retriever()

    history_prompt_for_system = (
        "Generates a response based on:"
        "1. User history (if available)"
        "2. Uploaded PDF content (if no relevant history)"
        "3. General knowledge (if neither exists)"
        
    )
    history_prompt= ChatPromptTemplate.from_messages(
        [
            ("system",history_prompt_for_system),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )
    system_prompt = (
        "Generates a response based on:"
        "1. User history (if available)"
        "2. Uploaded PDF content (if no relevant history)"
        "3. General knowledge (if neither exists)"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )
    history_retrieval= create_history_aware_retriever(llm,retrieval,history_prompt)#(model,vectore_store->retrieval,history_prompt ->which follow by this chain)
    #history prompt with no context it contain only what to do as system 
    document_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(history_retrieval,document_chain)
    
    def get_session_chat(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    
    if "store" not in st.session_state:
        st.session_state.store={}
        
    
    with_chat_history = RunnableWithMessageHistory(rag_chain,get_session_chat,
                                                input_messages_key="input",
                                                output_messages_key="answer",
                                                history_messages_key="chat_history")
    config = {"configurable":{"session_id":session_id}}
    if text_input: 
        response = with_chat_history.invoke(
            {"input":text_input},
            config=config
        )
        st.write(response['answer'])
        
        



