import streamlit as st
from crewai import Agent, Crew, LLM, Task
from dotenv import load_dotenv
import os
from reportlab.lib.pagesizes import A4  # Added import for ReportLab
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer  # Added imports for ReportLab
import io  # For handling in-memory files

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the prompt for the interview question agent
INTERVIEW_AGENT_PROMPT = """You are an expert Data Science Interview Question Agent
with extensive knowledge in various data science domains. 
Given a list of specific topics, 
your task is to generate 10 well-crafted interview questions for each topic.
Ensure that the questions assess both theoretical understanding and practical application, 
varying in difficulty to suit different experience levels. Present each question clearly and concisely, 
suitable for evaluating candidates' expertise and problem-solving abilities in data science."""

# Initialize the interview question agent
interview_agent = Agent(
    role="Interview Question Agent",
    goal="Provide algorithm interview questions on specified topics. Respond clearly and concisely (2-4 lines).",
    backstory=INTERVIEW_AGENT_PROMPT,
    verbose=False,
    llm=LLM(model="gpt-3.5-turbo", api_key=openai_api_key, base_url="https://api.openai.com/v1")
)

def interview_agent_function(topics):
    interview_task = Task(
        description=f"Generate interview questions for the following topics:\n{topics}\n\n"
                    f"Ensure each topic has 10 well-crafted questions that assess both theoretical understanding and practical application.",
        agent=interview_agent,
        expected_output="List of interview questions per specified topic."
    )
    crew = Crew(agents=[interview_agent], tasks=[interview_task], verbose=False)
    result = crew.kickoff()
    
    # Extract the raw questions
    raw_questions = None
    for item in result:
        if item[0] == 'raw':
            raw_questions = item[1]
            break
    
    if raw_questions:
        # Split the raw string into individual questions
        question_lines = raw_questions.strip().split('\n')
        # Remove potential header lines
        if question_lines and ':' in question_lines[0]:
            question_lines = question_lines[1:]
        # Extract numbered questions
        questions = [line.strip().split('. ', 1)[1] for line in question_lines if '. ' in line]
        return questions
    return []

# Define the prompt for the answer agent
ANSWER_AGENT_PROMPT = """You are an expert Data Science Interview Answer Agent.
Given an interview question, provide a clear and concise answer that demonstrates a strong understanding of the topic."""

# Initialize the answer agent
answer_agent = Agent(
    role="Interview Answer Agent",
    goal="Provide clear and concise answers to interview questions.",
    backstory=ANSWER_AGENT_PROMPT,
    verbose=False,
    llm=LLM(model="gpt-3.5-turbo", api_key=openai_api_key, base_url="https://api.openai.com/v1")
)

def answer_agent_function(question):
    answer_task = Task(
        description=f"Provide a comprehensive answer to the following interview question:\n{question}",
        agent=answer_agent,
        expected_output="A clear and concise answer to the interview question."
    )
    crew = Crew(agents=[answer_agent], tasks=[answer_task], verbose=False)
    result = crew.kickoff()
    
    # Extract the raw answer
    for item in result:
        if item[0] == 'raw':
            return item[1].strip()
    return "No answer provided."

def generate_pdf(questions, answers):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Interview Questions and Answers", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Add questions and answers
    for idx, (question, answer) in enumerate(zip(questions, answers), 1):
        q = Paragraph(f"<b>{idx}. {question}</b>", styles['Heading3'])
        story.append(q)
        a = Paragraph(f"Answer: {answer}", styles['BodyText'])
        story.append(a)
        story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    st.title("Data Science Interview Question and Answer Generator")

    st.sidebar.header("Input")
    topic = st.sidebar.text_input("Enter a topic:", "")

    if st.sidebar.button("Generate Questions"):
        if topic.strip() == "":
            st.sidebar.error("Please enter a valid topic.")
        else:
            with st.spinner("Generating questions..."):
                questions = interview_agent_function(topic)
                if questions:
                    st.session_state['questions'] = questions
                    st.session_state['answers'] = []
                else:
                    st.sidebar.error("Failed to generate questions.")
            st.success("Questions generated successfully!")

    if 'questions' in st.session_state and st.session_state['questions']:
        st.header("Interview Questions")
        for idx, question in enumerate(st.session_state['questions'], 1):
            st.write(f"**{idx}. {question}**")

        # Add the "Generate Answers" button
        if st.button("Generate Answers"):
            with st.spinner("Generating answers..."):
                answers = []
                for question in st.session_state['questions']:
                    answer = answer_agent_function(question)
                    answers.append(answer)
                st.session_state['answers'] = answers
            st.success("Answers generated successfully!")

    if 'answers' in st.session_state and st.session_state['answers']:
        st.header("Questions and Answers")
        for idx, (question, answer) in enumerate(zip(st.session_state['questions'], st.session_state['answers']), 1):
            st.write(f"**{idx}. {question}**")
            st.write(f"*Answer:* {answer}\n")

        # Add the "Download PDF" button
        pdf = generate_pdf(st.session_state['questions'], st.session_state['answers'])
        st.download_button(
            label="Download Q&A as PDF",
            data=pdf,
            file_name="Interview_QA.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
