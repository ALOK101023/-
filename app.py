import streamlit as st
import os
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Page Setup ---
st.set_page_config(
    page_title="मजदूर अधिकार सहायक",
    page_icon="⚖️",
    layout="centered"
)

# --- Security: Pull API Key from Streamlit Secrets ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("API Key नहीं मिली! कृपया Streamlit Secrets में 'OPENAI_API_KEY' जोड़ें।")
    st.stop()

# --- App Title ---
st.title("⚖️ मजदूर अधिकार सहायक")
st.markdown("अपनी समस्या **हिंदी** में लिखें और अपने कानूनी अधिकार जानें।")
st.warning("⚠️ यह ऐप केवल जानकारी के लिए है। कानूनी सलाह के लिए किसी वकील से मिलें।")

# --- RAG Logic (Processing the Text File) ---
@st.cache_resource
def initialize_system():
    # Ensure the data file exists
    if not os.path.exists("workers_rights.txt"):
        st.error("workers_rights.txt फाइल नहीं मिली।")
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
        template="""आप एक कानूनी सहायक हैं। उत्तर केवल हिंदी में दें।
        Context: {context}
        प्रश्न: {question}
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

# --- Chat Interface ---
try:
    chain = initialize_system()
    if chain:
        st.subheader("अपनी समस्या यहाँ लिखें:")
        user_input = st.text_area("जैसे: मालिक पैसे नहीं दे रहा है, क्या करूं?", height=100)

        if st.button("अधिकार जानें"):
            if user_input:
                with st.spinner("विश्लेषण किया जा रहा है..."):
                    response = chain.invoke(user_input)
                    st.subheader("कानूनी सलाह:")
                    st.write(response)
            else:
                st.error("कृपया पहले अपनी समस्या टाइप करें।")
except Exception as e:
    st.error(f"सिस्टम एरर: {e}")
