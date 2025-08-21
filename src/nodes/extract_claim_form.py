from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class CustomerData(BaseModel):
    name: str = Field(description="Full name of the person claiming")
    date_of_birth: str = Field(description="Date of birth")
    mailing_address: str = Field(description="Full mailing address")
    city: str = Field(description="City")
    postal_code: str = Field(description="Postal code")
    mobile: str = Field(description="Mobile phone number")
    email: str = Field(description="Email address")

class ClaimData(BaseModel):
    policy_number: str = Field(description="Policy number")
    reason_for_claim: str = Field(description="Reason for claiming")

def extract_customer_data(document):
    prompt = f"Extract customer data from the following claim form:\n\n{document}\n\n"
    
    chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
    structured_llm = chat.with_structured_output(CustomerData)
    messages = [HumanMessage(content=prompt)]
    response = structured_llm.invoke(messages)
    return response

def extract_claim_data(document):
    prompt = f"Extract claim data from the following claim form:\n\n{document}\n\n"
    
    chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
    structured_llm = chat.with_structured_output(ClaimData)
    messages = [HumanMessage(content=prompt)]
    response = structured_llm.invoke(messages)
    return response

def extract_claim_form_node(state):
    document = state.get("current_document", None)
    if not document:
        raise ValueError("No document found in the state.")

    customer_data = extract_customer_data(document)
    claim_data = extract_claim_data(document)

    return {
        "customer_data": customer_data,
        "claim_data": claim_data
    }
