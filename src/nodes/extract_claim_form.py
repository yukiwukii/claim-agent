from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

class CustomerData(BaseModel):
    name: str = Field(default=None, description="Full name of the person claiming")
    date_of_birth: str = Field(default=None, description="Date of birth")
    mailing_address: str = Field(default=None, description="Full mailing address")
    city: str = Field(default=None, description="City")
    postal_code: str = Field(default=None, description="Postal code")
    mobile: str = Field(default=None, description="Mobile phone number")
    email: str = Field(default=None, description="Email address")

class ClaimData(BaseModel):
    policy_number: str = Field(default=None, description="Policy number")
    reason_for_claim: str = Field(default=None, description="Reason for claiming")

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
