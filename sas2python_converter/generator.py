from retriever import get_relevant_docs
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def generate_python_code(sas_code):
    relevant_docs = get_relevant_docs(sas_code)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    prompt_template = PromptTemplate.from_template(
        """You are an expert in converting SAS code to optimized Python code using pandas.
Use the following relevant examples as reference:
{context}

Now convert the below SAS code into concise Python code:
{sas_code}

Python:"""
    )

    prompt = prompt_template.format(context=context, sas_code=sas_code)

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    response = model.invoke(prompt)
    return response.content.strip()
