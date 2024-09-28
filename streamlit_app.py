import streamlit as st
from utils import *
from rag import build_rag_chain_from_docs
from llm import *

st.title("💬 Interview Assistant")
st.write(
    "This is a simple Interviewer that asks questions based on the Job Description. Upload the job description file and answer the questions provied. After 4 Questions it will evaluate the candidate based on the responses."
)

openai_api_key = st.text_input("OpenAI API Key", type="password")
proxy_curl_api_key = st.text_input("Linked Scraper API Key", type="password")
if openai_api_key and proxy_curl_api_key:
    os.environ["PROXYCURL_API_KEY"] = proxy_curl_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    uploaded_file = st.file_uploader("Upload Job Description", type=("txt", "md"))
    is_scraping_needed = st.checkbox("Scrape LinkedIn Profile")
    linkedin_link = st.text_input("Upload LinkedIn Profile Link (Add any input if Scrape Profile is not ticked. It will use the sample profile stored in file)", type="default")

    if uploaded_file and linkedin_link:
        if "profile" not in st.session_state:
            if is_scraping_needed:
                print("Running Scraper..")
                linkedin_data = scrape_linkedin_profile(linkedin_link)
                st.session_state["profile"] = linkedin_data
            else:
                print("Loading from File..")
                linkedin_data = read_sample_linked_in_profile()
                st.session_state["profile"] = linkedin_data
        else:
            linkedin_data = st.session_state.profile

        st.button('Clear Chat', on_click=clear_chat)

        chain, retriever = build_rag_chain_from_docs(uploaded_file)

        if "messages" not in st.session_state:
            load_first_message()

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if user_input := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            response = generate_response(chain, linkedin_data, retriever)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
