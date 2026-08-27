import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ----------------- Page Configuration & Styling -----------------
st.set_page_config(
    page_title="Tollywood & TFI Database",
    page_icon="🎬",
    layout="centered"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .custom-header { text-align: center; margin-top: 2rem; margin-bottom: 0.5rem; }
    .custom-header h1 {
        font-size: 2.2rem; font-weight: 700; color: #ffffff;
        display: flex; align-items: center; justify-content: center; gap: 10px;
    }
    .custom-subtitle {
        text-align: center; color: #8b949e; font-size: 0.95rem; margin-bottom: 2rem;
    }
    .user-container {
        display: flex; align-items: center; background-color: #161b22;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; gap: 12px;
    }
    .bot-container {
        display: flex; align-items: flex-start; background-color: transparent;
        border-radius: 8px; padding: 4px 0px; margin-bottom: 20px; gap: 12px;
    }
    .avatar-user, .avatar-bot {
        border-radius: 6px; width: 32px; height: 32px; display: flex;
        align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0;
    }
    .avatar-user { background-color: #f87171; color: white; }
    .avatar-bot { background-color: #eab308; color: #161b22; }
    .user-text { color: #e6edf3; font-size: 0.95rem; }
    .bot-text { color: #c9d1d9; font-size: 0.95rem; line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

# ----------------- Setup RAG Pipeline -----------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

@st.cache_resource
def get_rag_chain():
    sample_docs = [
        Document(
            page_content=(
                "Telugu Cinema (Tollywood / TFI) is based in Andhra Pradesh and Telangana, centered in Film Nagar, Hyderabad. "
                "It is one of the largest film industries in India by box office revenue and ticket sales, renowned for "
                "commercial storytelling, mythological epics, and high-octane action spectacles."
            ),
            metadata={"topic": "TFI Overview"}
        ),
        Document(
            page_content=(
                "S. S. Rajamouli is one of India's most celebrated filmmakers, directing landmark films such as "
                "Magadheera, Eega, the Baahubali franchise (The Beginning and The Conclusion), and RRR. "
                "RRR brought global acclaim to TFI, winning the Academy Award (Oscar) for Best Original Song for 'Naatu Naatu', "
                "composed by M. M. Keeravani."
            ),
            metadata={"topic": "S. S. Rajamouli & Global Recognition"}
        ),
        Document(
            page_content=(
                "Pan-India Era: The Baahubali series pioneered the Pan-Indian cinema movement, bridging linguistic "
                "boundaries across North and South India. Other blockbuster franchises and films include Pushpa: The Rise, "
                "Pushpa 2: The Rule, Salaar: Part 1 – Ceasefire, Kalki 2898 AD, and Hanu-Man."
            ),
            metadata={"topic": "Pan-India Phenomenon"}
        ),
        Document(
            page_content=(
                "Prominent Stars: TFI boasts several top stars across generations, including N. T. Rama Rao, Akkineni Nageswara Rao, "
                "Chiranjeevi, Nandamuri Balakrishna, Nagarjuna, Venkatesh, followed by contemporary superstars like Prabhas, "
                "Mahesh Babu, Jr NTR, Allu Arjun, Ram Charan, Pawan Kalyan, and Nani."
            ),
            metadata={"topic": "Key Stars"}
        ),
        Document(
            page_content=(
                "Ramoji Film City, located in Hyderabad, is certified by the Guinness World Records as the world's largest "
                "film studio complex, serving as a primary production hub for Telugu and international filmmaking."
            ),
            metadata={"topic": "Ramoji Film City"}
        )
    ]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = FAISS.from_documents(sample_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for the Tollywood & TFI Database.
Answer the question accurately based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ----------------- UI Layout & Chat Flow -----------------
st.markdown("""
    <div class="custom-header">
        <h1>🎬 Tollywood & TFI Database</h1>
    </div>
    <div class="custom-subtitle">
        Ask me anything about Telugu cinema, Pan-India blockbusters, stars, and directors!
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
            <div class="user-container">
                <div class="avatar-user">👤</div>
                <div class="user-text">{msg['content']}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="bot-container">
                <div class="avatar-bot">🍿</div>
                <div class="bot-text">{msg['content']}</div>
            </div>
        """, unsafe_allow_html=True)

user_input = st.chat_input("E.g., What is the impact of Baahubali on Indian cinema?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"""
        <div class="user-container">
            <div class="avatar-user">👤</div>
            <div class="user-text">{user_input}</div>
        </div>
    """, unsafe_allow_html=True)

    if not GEMINI_API_KEY:
        response_text = "Please set the `GEMINI_API_KEY` environment variable to run queries."
    else:
        try:
            rag_chain = get_rag_chain()
            response_text = rag_chain.invoke(user_input)
        except Exception as e:
            response_text = f"Error retrieving answer: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.markdown(f"""
        <div class="bot-container">
            <div class="avatar-bot">🍿</div>
            <div class="bot-text">{response_text}</div>
        </div>
    """, unsafe_allow_html=True)
