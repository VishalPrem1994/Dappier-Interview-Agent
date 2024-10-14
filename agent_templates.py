def get_questionnaire_template():
    return """
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


def get_final_evaluation_template(profile_match_score, conversation):
    return f"""
        Evaluate the following responses of a candidate regarding a job.
        {conversation}
        Give a evaluation score out of 10 based on the candidate's experience with the skills required in the job description.
        And finally give an fina score which is the average based on profile match score of {profile_match_score} and the evaluation score.
        The candidate should be able to give detailed responses to the questions asked to get a good score above 6.
        Also respond with feedback on the candidate.  
        Be strict in your evaluation.
        """


def get_matcher_template(linked_in_data, job_description):
    return f"""
        Match the following profile with the job description provided.
        Profile:
        {linked_in_data}
        
        Job Description:
        {job_description}
        
        Match the profile with the job description and provide a integer score out of 10 based on the match.
        Return only the score and nothing else
        """
