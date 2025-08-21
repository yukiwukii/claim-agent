from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class BinaryClassifier(BaseModel):
    claim_form_or_not: bool = Field(description="Whether the text is a claim form or not")

def binary_classifier(state):
    document = state.get("current_document", None)
    if not document:
        raise ValueError("No document found in the state.")
    
    prompt = (
        f"Classify the following document as a claim form or not:\n\n{document}\n\n"
    )

    chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
    structured_llm = chat.with_structured_output(BinaryClassifier)
    messages = [HumanMessage(content=prompt)]
    response = structured_llm.invoke(messages)
    return response