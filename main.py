import PyPDF2
import base64
import requests

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from rich.console import Console
from dotenv import load_dotenv


# Config
load_dotenv()
console = Console()
chatgpt = ChatOpenAI()

# Prompts
SUMMARIZE_PROMPT_TEMPLATE = """
You are a helpful study assistant. Your task is to summarize educational material for a student studying the topic "{topic}".

Summarize the following study material into clear, concise bullet points that capture the most important facts, definitions, and key ideas. Avoid fluff. Use plain and accessible language.

Study Material:
{study_material}

Summary:
"""

CREATE_QUIZ_PROMPT_TEMPLATE = """
You are a study assistant that creates practice quizzes from study notes.

Based on the following summary, generate 5 multiple-choice questions that test a student's understanding of the topic. Each question should have:

1 correct answer

3 plausible but incorrect options (distractors)

Clearly labeled options (A, B, C, D)

An answer key at the end

Keep the questions relevant to the summary content and designed for effective review.

Use the following output format when generating the output response:

output format instructions:
{format_instructions}

Study material summary:
{summary}
"""


def extract_pdf_content(path: str) -> str:
    """
    Extracts and returns all text content from a PDF file.

    Parameters:
        path (str): Path to the PDF file.

    Returns:
        str: Extracted and cleaned text from the entire PDF.
    """
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text_content = ""

        for page in reader.pages:
            text = page.extract_text()
            text_content += " " + text.strip()

        text_content = ' '.join(text_content.split())
    return text_content


def draw_mermaid_graph(chain):
    """
    Creates a Mermaid graph from the given chain, renders it via the mermaid.ink API,
    and saves the image as 'chain_workflow.png' in the current directory.

    Parameters:
        chain: A LangChain object that supports get_graph().draw_mermaid().

    Returns:
        None
    """
    mermaid_syntax = chain.get_graph().draw_mermaid()
    encoded = base64.urlsafe_b64encode(mermaid_syntax.encode("utf-8")).decode("utf-8")
    url = f"https://mermaid.ink/img/{encoded}"
    url_response = requests.get(url)

    if url_response.status_code == 200:
        with open("chain_workflow.png", "wb") as f:
            f.write(url_response.content)
        print("Chain workflow graph saved as chain_workflow.png")
    else:
        print(f"Error fetching image: {url_response.status_code}")


class QueryResponse(BaseModel):
    summary: str = Field(description="A concise summary of the study material in bullet-point format.")
    quiz: str = Field(description="A set of multiple-choice questions generated from the summary, including distractors and an answer key.")


parser = PydanticOutputParser(pydantic_object=QueryResponse)


# Inputs
inputs = [
    ("Prompt Engineering", extract_pdf_content(path="content/Prompt Engineering.pdf")),
    ("Prompt Engineering for Agents", (
        "Prompt engineering involves designing and refining inputs to language models to achieve desired outputs. "
        "In the context of agents, prompt engineering allows for better control over how an agent interacts with the environment "
        "and solves specific tasks. This is particularly useful in domains like robotics and conversational AI. "
        "By adjusting the structure and content of the prompts, users can enhance an agent's performance on specific tasks."
    ))
]

# Chain
summarize_prompt_template = PromptTemplate.from_template(SUMMARIZE_PROMPT_TEMPLATE)

summarize_chain =(
        summarize_prompt_template
        |
        chatgpt
        |
        StrOutputParser()
)

quiz_prompt_template = PromptTemplate(
    template=CREATE_QUIZ_PROMPT_TEMPLATE,
    input_variables=["summary"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

quiz_chain =(
        quiz_prompt_template
        |
        chatgpt
        |
        StrOutputParser()
)

combined_chain = (
    summarize_chain
    |
    quiz_chain
)

draw_mermaid_graph(combined_chain)


if __name__ == "__main__":
    inputs_data = [
        {
            "topic": topic,
            "study_material": study_material
        } for topic, study_material in inputs
    ]
    response = combined_chain.map().invoke(inputs_data)

    # Agent's responses
    for (topic, _), quiz in zip(inputs, response):
        print("\n" + "-" * 10 + " New quiz! ✨ " + "-" * 10)
        print("Topic: {}\n\n{}".format(topic, quiz))