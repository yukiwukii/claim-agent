from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal, List
from dotenv import load_dotenv
import yaml
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
    claim_type: Literal[
        "travel_delay",
        "baggage_delay", 
        "trip_cancellation",
        "trip_interruption"
    ] = Field(description="Type of insurance claim based on the claim form content")

def load_claim_config(config_path: str = "src/utils/files_required.yml") -> dict:
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

def get_required_documents_for_claim_type(claim_type: str, config_path: str = "src/utils/files_required.yml") -> List[str]:
    config = load_claim_config(config_path)
    return config['claim_types'].get(claim_type, {}).get('document_classes', [])

def extract_customer_data(document):
    prompt = f"Extract customer data from the following claim form:\n\n{document}\n\n"
    
    chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
    structured_llm = chat.with_structured_output(CustomerData)
    messages = [HumanMessage(content=prompt)]
    response = structured_llm.invoke(messages)
    return response

def extract_claim_data(document):
    prompt = f"""Extract claim data from the following claim form and classify the type of claim:

{document}

Please identify:
1. The policy number
2. The reason for the claim
3. The type of claim based on the content:
   - travel_delay: Claims for expenses due to flight/travel delays (hotel, meals due to delays)
   - baggage_delay: Claims for delayed baggage and related expenses
   - trip_cancellation: Claims for cancelled trips before departure
   - trip_interruption: Claims for trips that were interrupted after departure

Look for keywords and context clues in the claim form to determine the appropriate claim type.
"""
    
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
    
    required_documents = get_required_documents_for_claim_type(claim_data.claim_type)
    print("Claim type is:", claim_data.claim_type)
    print("Required documents are:", required_documents)

    # all_classifications = state.get("all_classifications", {})

    return {
        "customer_data": customer_data,
        "claim_data": claim_data,
        "claim_type": claim_data.claim_type,
        "required_documents": required_documents
    }