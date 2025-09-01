from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from datetime import date, datetime
from decimal import Decimal
import os

class Passport(BaseModel):
    passport_number: str = Field(..., description="The passport number.")
    name: str = Field(..., description="The full name of the passport holder.")
    date_of_birth: date = Field(..., description="The date of birth of the passport holder in YYYY-MM-DD format.")
    nationality: str = Field(..., description="The nationality of the passport holder.")
    expiry_date: date = Field(..., description="The expiry date of the passport in YYYY-MM-DD format.")

class Receipt(BaseModel):
    vendor_name: str = Field(..., description="The name of the vendor or store.")
    transaction_date: date = Field(..., description="The date of the transaction in YYYY-MM-DD format.")
    total_amount: float = Field(..., description="The total amount of the transaction.")
    items: List[str] = Field(..., description="A list of items purchased.")

class Itinerary(BaseModel):
    confirmation_number: str = Field(..., description="The booking confirmation number.")
    name: str = Field(..., description="The name of the passenger.")
    departure_city: str = Field(..., description="The departure city.")
    arrival_city: str = Field(..., description="The arrival city.")
    departure_date: date = Field(..., description="The date of departure in YYYY-MM-DD format. Follow strictly this format.")
    arrival_date: date = Field(..., description="The date of arrival in YYYY-MM-DD format. Follow strictly this format.")
    flight_numbers: List[str] = Field(..., description="A list of flight numbers for the trip.")

class DelayNotification(BaseModel):
    airline: str = Field(..., description="The airline that issued the notification.")
    flight_number: str = Field(..., description="The flight number that was delayed.")
    original_departure_time: datetime = Field(..., description="The original departure time in YYYY-MM-DDTHH:MM:SS format.")
    new_departure_time: datetime = Field(..., description="The new, delayed departure time in YYYY-MM-DDTHH:MM:SS format.")
    delay_reason: str = Field(..., description="The reason for the delay, if provided.")

SCHEMA_MAP = {
    "passport": Passport,
    "receipt": Receipt,
    "itinerary": Itinerary,
    "delay_notification": DelayNotification,
}

def load_document_from_path(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Document file {filepath} not found.")
    except Exception as e:
        raise Exception(f"Error reading document {filepath}: {e}")

def extract_structured_data(document_chunk: str, schema: Type[BaseModel]) -> Optional[BaseModel]:
    prompt = f"""
    You are an expert at extracting information from documents.
    Please extract the relevant details from the following document chunk and format it according to the provided schema.
    If there is any missing or unclear information, leave it blank.

    Document Content:
    ---
    {document_chunk}
    ---
    """
    chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
    structured_llm = chat.with_structured_output(schema)
    messages = [HumanMessage(content=prompt)]
    
    try:
        response = structured_llm.invoke(messages)
        return response
    except Exception as e:
        print(f"Error extracting structured data: {e}")
        return None

def structured_output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    processed_documents = state.get("processed_documents", [])
    all_classifications = state.get("all_classifications", {})
    supporting_docs = state.get("supporting_documents_data", {})

    for doc_path in processed_documents:
        doc_filename = os.path.basename(doc_path)
        if doc_filename in supporting_docs:
            continue
            
        doc_classifications = all_classifications.get(doc_path, [])
        doc_content = load_document_from_path(doc_path)
        doc_lines = doc_content.splitlines()

        for classification in doc_classifications:
            class_name = classification['document_class']
            start_line = classification.get('start_line', 0)
            end_line = classification.get('end_line', 0)

            if class_name == 'irrelevant' or start_line <= 0 or end_line <= 0 or end_line < start_line:
                continue
                
            # Double entry TODO: Add a functionality.
            if class_name in supporting_docs:
                continue 

            schema = SCHEMA_MAP.get(class_name)
            if schema:
                document_chunk = "\n".join(doc_lines[start_line - 1 : end_line]) # TODO: Add save here.
                
                print(f"\n\nExtracting '{class_name}' details from {doc_filename} (lines {start_line}-{end_line})...")
                extracted_data = extract_structured_data(document_chunk, schema)
                
                if extracted_data:
                    if class_name not in supporting_docs:
                        supporting_docs[class_name] = []
                    supporting_docs[class_name].append(extracted_data.model_dump())
                    print(f"Successfully extracted: {extracted_data.model_dump()}")
                    
    return {"supporting_documents_data": supporting_docs}