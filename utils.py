import os
import requests
import json
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter


def scrape_linkedin_profile(linkedin_profile_url: str):
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    header_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}

    response = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic
    )

    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
           and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data


def write_sample_linked_in_profile(linkedin_data):
    with open('demo_linkedin_profile.txt', 'w') as convert_file:
        convert_file.write(json.dumps(linkedin_data))


def read_sample_linked_in_profile():
    with open('demo_linkedin_profile.txt') as f:
        data = f.read()
    js = json.loads(data)
    return js


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def clear_chat():
    st.session_state.pop("messages")


def load_first_message():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Tell me about yourself?"}
    ]


def split_text(article, chunk_size=2000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_text(article)
    return splits
