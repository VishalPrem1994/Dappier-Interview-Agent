import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain

import os

# Show title and description.
st.title("💬 Interview Assistant")
st.write(
    "This is a simple Interviewer that asks questions based on the Job Description. Upload the job description file and answer the questions provied. After 4 Questions it will evaluate the candidate based on the responses."
)

openai_api_key = st.text_input("OpenAI API Key", type="password")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def click_button():
    st.session_state.pop("messages")
    uploaded_file = None


if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    uploaded_file = st.file_uploader("Upload Job Description", type=("txt", "md"))

    if (uploaded_file):

        article = uploaded_file.read().decode()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_text(article)

        vectorstore = FAISS.from_texts(texts=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        questionaire_llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
        evaluation_llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key)
        # Define prompt template
        template = """
        You are an interview assistant for generating a single question based on a job description.
        Make sure to ask questions about the skills and experience level required in the job description. 
        You will be provided with all previous questions asked.
        Use the provided context only to generate questions:

        <context>
        {context}
        </context>

        Previous Conversations: {input}
        Generate just one question that is not part of the previous conversation and nothing else.
        """

        st.button('Clear Chat', on_click=click_button)

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Tell me about yourself?"}
            ]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        prompt = ChatPromptTemplate.from_template(template)

        doc_chain = create_stuff_documents_chain(questionaire_llm, prompt)
        chain = create_retrieval_chain(retriever, doc_chain)

        if prompt := st.chat_input():
            if len(st.session_state.messages) >= 7:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                evaluation_template = """Evaluate the following responses of a candidate regarding a job.
                                """ + str(st.session_state.messages) + """ 
                                Give a score out of 10 based on the candidate's experience with the skills required in the job description.
                                The candidate should be able to give detailed responses to the questions asked to get a good score above 6.
                               Also respond with feedback on the candidate.  
                               Be strict in your evaluation.
                            
                            """
                response = evaluation_llm(evaluation_template)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})

                st.chat_message("user").write(prompt)
                print(str(st.session_state.messages))
                response = chain.invoke({"input": str(st.session_state.messages)})

                msg = response["answer"]

                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
