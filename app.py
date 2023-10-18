import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from htmlTemplates import css, bot_template, user_template
from langchain.prompts.prompt import PromptTemplate
import time
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,  # number of tokens overlap between chunks
        length_function=len,
        separators=['\n\n', '\n', ' ', '']
    )

    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    persist_dir = "./vdbStore"
    model_name ='sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    if os.path.exists(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_texts(texts = text_chunks, embedding = embeddings, persist_directory=persist_dir)
    vectorstore.persist()
    return vectorstore

def get_conversation_chain(vectorStore):
    llm = ChatOpenAI(temperature=0.0, model_name = 'gpt-3.5-turbo')

    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    
    _template = """ You are a technical support specialist. Given the following coversation and a follow-up question, use the context to answer the question.
    Follow Up Input: {question}
    Context: {context}
    Chat History: {chat_history}"""

    prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=_template)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorStore.as_retriever(),
        memory = memory,
        verbose = True,
        #combine_docs_chain_kwargs={'prompt': prompt},
        rephrase_question = False)
    
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title="Document Q&A with PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Document Q&A with PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Document Loader")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                print(len(text_chunks))
                st.write(text_chunks)

                print('started', time.time)
                # create vector store
                vectorstore = get_vector_store(text_chunks)
                print('ended', time.time)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()