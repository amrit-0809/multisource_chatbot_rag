# ================== IMPORTING LIBRARIES ==================
import os
import uuid
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from youtube_transcript_api import YouTubeTranscriptApi


# ================== MODELS ==================
# load_dotenv(dotenv_path="a.env")
GROQ_API_KEY = 'I will put my key here'   

if not GROQ_API_KEY:
    st.warning("Missing GROQ_API_KEY. Please add it to a.env")
    st.stop()

llm = ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=512,
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ================== UI HEADER ==================
st.title("BIG Chuchi BIG 🍑 Chatbot 💦")
st.write("Multi-Session Chat with PDF, YouTube & Web RAG")


# ================== STATE INIT ==================
if "store" not in st.session_state:
    st.session_state.store = {}
if "disp" not in st.session_state:
    st.session_state.disp = {}
if "messages_display" not in st.session_state:
    st.session_state.messages_display = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "session_id" not in st.session_state:
    st.session_state.session_id = "newchat"
if "newchat_counter" not in st.session_state:
    st.session_state.newchat_counter = 1
if "files_hash" not in st.session_state:
    st.session_state.files_hash = None


# ================== UTILS ==================
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

def _is_invalid_session_id(sid: str) -> bool:
    return (sid is None) or (not isinstance(sid, str)) or (sid.strip() == "")

def _generate_new_session_id() -> str:
    sid = f"newchat_{st.session_state.newchat_counter:03d}"
    st.session_state.newchat_counter += 1
    return sid

def bind_messages_display_to_current_session():
    sid = st.session_state.session_id
    if _is_invalid_session_id(sid):
        sid = _generate_new_session_id()
        st.session_state.session_id = sid
    if sid not in st.session_state.disp:
        st.session_state.disp[sid] = []
    st.session_state.messages_display = st.session_state.disp[sid]

def clear_vector_db():
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.files_hash = None

def switch_session(new_sid: str):
    st.session_state.user_input = ""
    if _is_invalid_session_id(new_sid):
        return
    st.session_state.session_id = new_sid
    if new_sid not in st.session_state.disp:
        st.session_state.disp[new_sid] = []
    st.session_state.messages_display = st.session_state.disp[new_sid]
    if new_sid not in st.session_state.store:
        st.session_state.store[new_sid] = ChatMessageHistory()
    clear_vector_db()
    return True

def create_new_session():
    new_sid = _generate_new_session_id()
    st.session_state.disp[new_sid] = []
    st.session_state.store[new_sid] = ChatMessageHistory()
    switch_session(new_sid)


# ================== YOUTUBE TRANSCRIPT LOADER ==================
def load_youtube_transcript(url: str):
    """Fetch transcript from YouTube and return as LangChain Document."""
    try:
        video_id = url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        text = " ".join([t['text'] for t in transcript if t['text'].strip() != ""])
        return [Document(page_content=text, metadata={"source": url})]
    except Exception as e:
        st.error(f"Could not fetch transcript for {url}: {e}")
        return []


# ================== RAG SOURCE UPLOAD ==================
st.subheader("📂 Upload or Link Knowledge Sources")

uploaded_files = st.file_uploader(
    "Choose PDF file(s)", type="pdf", accept_multiple_files=True
)
youtube_links = st.text_area("Paste YouTube links (comma separated)", "")
web_links = st.text_area("Paste Webpage links (comma separated)", "")


def compute_files_hash(files, yt, wb) -> str | None:
    if not files and not yt and not wb:
        return None
    names = tuple(sorted([f.name for f in files])) + tuple(sorted(yt.split(","))) + tuple(sorted(wb.split(",")))
    return str(names)


current_hash = compute_files_hash(uploaded_files, youtube_links, web_links)
rebuild = current_hash is not None and current_hash != st.session_state.files_hash
if current_hash is None:
    clear_vector_db()
if rebuild:
    documents = []

    # PDFs
    for uf in uploaded_files:
        temp_name = f"./tmp_{uuid.uuid4().hex}.pdf"
        with open(temp_name, "wb") as f:
            f.write(uf.getvalue())
        docs = PyPDFLoader(temp_name).load()
        documents.extend(docs)
        os.remove(temp_name)

    # YouTube
    if youtube_links:
        for link in youtube_links.split(","):
            link = link.strip()
            if link:
                docs = load_youtube_transcript(link)
                documents.extend(docs)

    # Webpages
    if web_links:
        urls = [url.strip() for url in web_links.split(",") if url.strip()]
        if urls:
            loader = WebBaseLoader(urls)
            docs = loader.load()
            documents.extend(docs)

    # Build Vectorstore
    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    st.session_state.vectorstore = FAISS.from_documents(splits, embedding=embeddings)
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    st.session_state.files_hash = current_hash
    st.success(f"Indexed {len(splits)} chunks from {len(documents)} sources.")


# ================== SIDEBAR CONTROLS ==================
def get_options():
    keys = list(st.session_state.disp.keys())
    if st.session_state.session_id not in keys:
        keys.append(st.session_state.session_id)
    return sorted(set(keys))

with st.sidebar:
    st.write(f"Current session: {st.session_state.session_id}")
    options = get_options()
    selected = st.selectbox(
        "Select session",
        options=options,
        index=options.index(st.session_state.session_id) if st.session_state.session_id in options else 0,
        key="session_selector",
    )
    if selected != st.session_state.session_id:
        if not (uploaded_files or youtube_links or web_links):
            switch_session(selected)
            st.success(f"Switched to: {selected}")
        else:
            st.error('First clear uploaded/linked sources!')

    if st.button("New session"):
        if not (uploaded_files or youtube_links or web_links):
            create_new_session()
            st.rerun()
            st.success(f"Created: {st.session_state.session_id}")
        else:
            st.error('First clear uploaded/linked sources!')


# ================== CHAINS ==================
def rag_chain(user_input):
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the latest question as standalone using chat history. Do not answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, contextualize_prompt
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use ONLY the retrieved context to answer. If unknown, say 'i don't know'.\n\ncontext:{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    core_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_chain = RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    res = conversational_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": st.session_state.session_id}},
    )
    return res['answer']

def chain(user_input):
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "Chat normally, considering prior messages."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    core_chain = chat_prompt | llm
    conversational_chain = RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )
    res = conversational_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": st.session_state.session_id}},
    )
    return res.content

def main_chain(user_input):
    if st.session_state.retriever is None:
        return chain(user_input)
    else:
        return rag_chain(user_input)


# ================== CHAT UI ==================
st.divider()
with st.form("ask"):
    user_input = st.text_input("Your question:", key="user_input")
    submitted = st.form_submit_button("Submit")

if submitted and user_input:
    answer = main_chain(user_input)
    st.session_state.messages_display.append((user_input, answer))

for user_msg, bot_msg in st.session_state.messages_display:
    st.write(f"**You:** {user_msg}")
    st.write(f"**Assistant:** \n{bot_msg}")
    st.write("---")

