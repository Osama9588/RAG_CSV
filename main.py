import streamlit as st
from langchain_community.llms import Ollama
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import tempfile

DB_FAISS_PATH = 'vectorstore/db_faiss'

st.title("Financial Data Q&A with orca-mini")

uploaded_file = st.file_uploader("Upload a CSV file with financial statements", type=['csv'])

def load_llm():
    st.write("Loading the orca-mini model...")
    llm = Ollama(model='orca-mini')
    st.write("Model loaded successfully.")
    return llm

if uploaded_file:
    st.write("CSV file uploaded. Processing...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        uploaded_file_path = tmp_file.name

    st.write("Loading data from the CSV file...")
    loader = CSVLoader(file_path=uploaded_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    st.write("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    st.write("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    st.write("Creating FAISS vector store...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_FAISS_PATH)

    st.write("Loading the LLM...")
    llm = load_llm()
    retriever = db.as_retriever()

    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['k'] = 4

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    st.write("System is ready. Ask your questions below!")

    def conversational_chat(query, chat_history=[]):
        prompt = f"User asked: {query}\nPlease provide a concise and accurate response."
        result = chain({"question": prompt, "chat_history": chat_history})
        chat_history.append((query, result["answer"]))
        return result["answer"], chat_history

    query = st.text_input("Enter your question:")
    if st.button("Get Answer") and query:
        st.write("Processing your query...")
        answer, st.session_state.chat_history = conversational_chat(query, st.session_state.get("chat_history", []))
        st.write(f"**Bot:** {answer}")

    st.write("### Conversation History")
    for q, a in st.session_state.get("chat_history", []):
        st.write(f"**User:** {q}")
        st.write(f"**Bot:** {a}")
