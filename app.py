from crewai import Agent, Task, Crew, LLM

import os
from dotenv import load_dotenv

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")



INTERVIEW_AGENT_PROMPT = """You are an expert Data Science Interview Question Agent
 with extensive knowledge in various data science domains. 
 Given a list of specific topics, 
 your task is to generate 10 well-crafted interview questions for each topic.
   Ensure that the questions assess both theoretical understanding and practical application, 
   varying in difficulty to suit different experience levels. Present each question clearly and concisely, 
   suitable for evaluating candidates' expertise and problem-solving abilities in data science."""



interview_agent = Agent(
    role="Interview Question Agent",
    goal="Provide algorithm interview questions on specified topics. Respond clearly and concisely (2-4 lines).",
    backstory=INTERVIEW_AGENT_PROMPT,
    verbose=False,
    llm=LLM(model="gpt-3.5-turbo",api_key=openai_api_key, base_url="https://api.openai.com/v1")
)



def interview_agent_function(topics):
    interview_task = Task(
        description=f"Generate interview questions for the following topics:\n{topics}\n\n"
                    f"Ensure each topic has 10 well-crafted questions that assess both theoretical understanding and practical application.",
        agent=interview_agent,
        expected_output="List of interview questions per specified topic."
    )
    crew = Crew(agents=[interview_agent], tasks=[interview_task], verbose=True)
    result = crew.kickoff()
    return result


final_result = interview_agent_function("Data Science")
print(final_result)
