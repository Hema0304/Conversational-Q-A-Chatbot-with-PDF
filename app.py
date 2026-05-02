##RAG Q&A conversation with PDF including chat history
import streamlit as st 
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os 
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings=load_embeddings()


# embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')


st.title("Conversation RAG with PDF uploads and chat history")
st.write("Upload pdf's and chat with their content")

api_key = st.text_input("Enter your Groq API key:",type="password")

if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="qwen/qwen3-32b")
    
    session_id=st.text_input("Session ID",value='default_session')
    
    
    if 'store' not in st.session_state:
        st.session_state.store={}
        
    uploaded_file=st.file_uploader("Choose a PDF file",type="pdf")
    
    
    if uploaded_file:
        documents=[]
        
        file_name = uploaded_file.name          # ✅ get name first
        temppdf=f"./{file_name}"
        with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())

            # file_name = uploaded_file.name
            # temppdf=f"./{file_name}"
            # with open(temppdf,"wb") as file:
            #     file.write(uploaded_file.getvalue())
               
                
        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
            
        #split and create embeddings for documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
                
        
        retriever = vectorstore.as_retriever()
            
        contextualize_q_system_prompt=(
            "given a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understood"
            "without the chat history. Do NOT answer the question,"
            "just reformulate it if needed and otherwise retrun it as is. "
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        #answer question
        system_prompt = (
            "you are an  assistant for question-answering tasks."
            "use the following pieces of retrieved context to answer"
            "the question. if you don't know the answer , say that you"
            "don't know .use three sentences maximum and keep the"
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human",'{input}'),
            ]
        )

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key = "answer"
        )
        
        user_input = st.text_input("Your Question:")
        
        if user_input:
            session_history=get_session_history(session_id)
            response= conversational_rag_chain.invoke(
                {"input" : user_input},
                config={
                    "configurable": {"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.success("Assistant: {response['answer']}")
            st.write("Chat History:",session_history.messages)
else:
    st.warning("please enter the groq API key")
        

