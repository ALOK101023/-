import streamlit as st
import os
# नए वर्ज़न के लिए 'langchain_text_splitters' का उपयोग करें
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- पेज सेटअप ---
st.set_page_config(
    page_title="मजदूर अधिकार सहायक",
    page_icon="⚖️",
    layout="wide"
)

# --- API Key सेटअप (Streamlit Secrets के लिए) ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Secrets में 'OPENAI_API_KEY' नहीं मिला। कृपया डैशबोर्ड में Settings > Secrets में जाकर इसे जोड़ें।")
    st.stop()

# --- डेटा प्रोसेसिंग (RAG) ---
@st.cache_resource
def initialize_system():
    if not os.path.exists("workers_rights.txt"):
        st.error("workers_rights.txt फाइल नहीं मिली। कृपया इसे GitHub पर अपलोड करें।")
        return None

    with open("workers_rights.txt", "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = PromptTemplate(
        template="""आप एक कानूनी सहायक हैं जो भारतीय मजदूरों की मदद करते हैं। 
        दिए गए संदर्भ (Context) के आधार पर सरल हिंदी में जवाब दें।
        
        Context: {context}
        प्रश्न: {question}
        
        जवाब में कानून का नाम और हेल्पलाइन नंबर जरूर बताएं।
        अंत में लिखें: "अधिक जानकारी के लिए श्रम विभाग की हेल्पलाइन 1800-11-1363 पर कॉल करें।"
        """,
        input_variables=['context', 'question']
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel({'context': retriever | RunnableLambda(format_docs), 'question': RunnablePassthrough()})
        | prompt | llm | StrOutputParser()
    )
    return chain

# --- साइडबार (Sidepanel) ---
with st.sidebar:
    st.title("⚖️ सूचना पैनल")
    st.markdown("### 🌐 भाषा / Language\nयह ऐप पूरी तरह **हिंदी** और **English** में काम करता है।")
    st.divider()
    st.info("💡 टिप: यदि मालिक पैसे नहीं दे रहा या ओवरटाइम नहीं दे रहा, तो आप यहाँ पूछ सकते हैं।")

# --- मुख्य इंटरफेस ---
st.title("⚖️ मजदूर अधिकार सहायक")
st.caption("🚀 श्रमिक अधिकारों की जानकारी | Supports Hindi & English")

try:
    chain = initialize_system()
    if chain:
        # --- ऑटोमैटिक ऑप्शन (Pills) ---
        st.write("### मुख्य विषय चुनें:")
        options = ["न्यूनतम वेतन क्या है?", "मालिक पैसे नहीं दे रहा", "ओवरटाइम के नियम", "PF का पैसा", "साप्ताहिक छुट्टी"]
        selected_pill = st.pills("विषय:", options, selection_mode="single", label_visibility="collapsed")

        # चैट हिस्ट्री
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # इनपुट हैंडलिंग
        user_query = None
        if selected_pill:
            user_query = selected_pill
        if prompt_input := st.chat_input("अपनी समस्या यहाँ लिखें..."):
            user_query = prompt_input

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("कानूनी जानकारी खोजी जा रही है..."):
                    response = chain.invoke(user_query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"सिस्टम एरर: {str(e)}")
