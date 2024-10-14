import streamlit as st
from langchain_openai import OpenAI
import os
from agent_templates import get_final_evaluation_template, get_matcher_template
from langchain_openai import ChatOpenAI
from utils import split_text
import json
import statistics


def get_instruct_llm():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=os.environ["OPENAI_API_KEY"])
    return llm


def get_chat_completion_llm():
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.environ["OPENAI_API_KEY"])
    return llm


def evaluate_candidate(profile_match_score):
    evaluation_llm = get_chat_completion_llm()
    response = evaluation_llm(get_final_evaluation_template(profile_match_score,str(st.session_state.messages)))
    return response.content.strip()


def generate_question(chain):
    chain_response = chain.invoke({"input": str(st.session_state.messages)})
    return chain_response["answer"]


def generate_response(chain, linkedin_data, retriever):
    if len(st.session_state.messages) >= 7:
        st.chat_message("assistant").write("Evaluating Candidate...")
        profile_match_score = match_profile_and_job_description(linkedin_data, retriever)
        st.chat_message("assistant").write("Linkedin Profile Match Score: " + str(profile_match_score))
        response = evaluate_candidate(profile_match_score)
        return response
    else:
        msg = generate_question(chain)
        return msg


def match_profile_and_job_description(linked_in_data, retriever):
    llm = get_chat_completion_llm()
    splits = split_text(json.dumps(linked_in_data))
    match_scores = []
    for i in splits:
        matched_docs = retriever.invoke(i)
        response = llm(get_matcher_template(linked_in_data, matched_docs))
        match_scores.append(int(response.content.strip()))

    print("Final Scores")
    print(match_scores)
    return statistics.mean(match_scores)
