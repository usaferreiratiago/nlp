import os
import tempfile
import streamlit as st

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader


st.title('ChromaDB + Langchain Document Summarizer')

open_ai_key = st.text_input("OpenAI API KEY", type="password")
source_doc = st.file_uploader("Upload source document (PDF)", type="pdf")

if st.button("Summarize"):

    if not open_ai_key.strip() or not source_doc:
        st.write("Please provide the missing input")

    else:
        try:
            # salvar arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
                tmp_path = tmp_file.name

            # carregar PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load_and_split()

            # remover arquivo
            os.remove(tmp_path)

            # embeddings
            embeddings = OpenAIEmbeddings(api_key=open_ai_key)

            # vetor store
            vectordb = Chroma.from_documents(pages, embedding=embeddings)

            # LLM
            llm = OpenAI(temperature=0, api_key=open_ai_key)

            # summarize chain
            chain = load_summarize_chain(llm, chain_type="stuff")

            # busca semântica (opcional)
            docs = vectordb.similarity_search("summary")

            # resumo
            summary = chain.run(docs)

            st.write(summary)

        except Exception as e:
            st.write(f"An error occurred: {e}")