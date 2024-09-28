from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from agent_templates import get_questionnaire_template
from llm import get_chat_completion_llm
from utils import split_text


def build_rag_chain_from_docs(uploaded_file):
    article = uploaded_file.read().decode()
    retriever = build_retriever(article)

    questionnaire_prompt = ChatPromptTemplate.from_template(get_questionnaire_template())
    questionnaire_llm = get_chat_completion_llm()

    doc_chain = create_stuff_documents_chain(questionnaire_llm, questionnaire_prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    return chain, retriever


def build_retriever(article):
    splits = split_text(article, chunk_size=2000, chunk_overlap=200)
    vectorstore = FAISS.from_texts(texts=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return retriever
