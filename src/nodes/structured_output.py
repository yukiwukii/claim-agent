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
    If there is any missing or unclear information, leave it blank or use appropriate defaults.

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
        print(f"   ⚠️ Error extracting structured data: {e}")
        return None

def extract_lines_from_document(doc_content: str, start_line: int, end_line: int) -> str:
    doc_lines = doc_content.splitlines()
    start_idx = max(0, start_line - 1)
    end_idx = min(len(doc_lines), end_line)
    
    return "\n".join(doc_lines[start_idx:end_idx])

def structured_output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    document_queue = state.get("document_queue", [])
    document_filepath = state.get("document_filepath", "")
    all_classifications = state.get("all_classifications", {})
    supporting_docs = state.get("supporting_documents_data", {})
    
    # Determine which documents to process
    documents_to_process = []
    if document_queue:
        documents_to_process = document_queue
    elif document_filepath:
        documents_to_process = [document_filepath]
    
    if not documents_to_process:
        print("⚠️ No documents to extract structured data from")
        return state
    
    print("\n" + "=" * 60)
    print("STRUCTURED DATA EXTRACTION")
    print("=" * 60)
    
    for doc_path in documents_to_process:
        doc_filename = os.path.basename(doc_path)
        doc_classifications = all_classifications.get(doc_path, [])
        
        if not doc_classifications:
            print(f"\n⚠️ No classifications found for {doc_filename}")
            continue
        
        print(f"\nProcessing: {doc_filename}")
        print("-" * 40)
        
        try:
            doc_content = load_document_from_path(doc_path)
        except Exception as e:
            print(f"❌ Error loading document: {e}")
            continue
        
        for classification in doc_classifications:
            segment_id = classification.get('segment_id', 0)
            document_classes = classification.get('document_classes', [])
            start_line = classification.get('start_line', 0)
            end_line = classification.get('end_line', 0)
            description = classification.get('description', 'Unknown segment')
            
            if start_line <= 0 or end_line <= 0 or end_line < start_line:
                print(f"   ⚠️ Invalid line numbers for segment {segment_id}")
                continue
            
            document_chunk = extract_lines_from_document(doc_content, start_line, end_line)
            
            for class_name in document_classes:
                if class_name in ['irrelevant', 'claim_form']:
                    continue
                
                schema = SCHEMA_MAP.get(class_name)
                if not schema:
                    print(f"   ⚠️ No schema defined for '{class_name}'")
                    continue
                
                print(f"   Extracting '{class_name}' from segment {segment_id} (lines {start_line}-{end_line})")
                print(f"   Description: {description}")
                
                extracted_data = extract_structured_data(document_chunk, schema)
                
                if extracted_data:
                    data_dict = extracted_data.model_dump()
                    data_dict['_source'] = {
                        'filepath': doc_path,
                        'filename': doc_filename,
                        'segment_id': segment_id,
                        'start_line': start_line,
                        'end_line': end_line,
                        'description': description
                    }
                    
                    if class_name not in supporting_docs:
                        supporting_docs[class_name] = []
                    
                    supporting_docs[class_name].append(data_dict)
                    
                    print(f"      ✅ Successfully extracted {class_name}")
                    if class_name == 'passport':
                        print(f"         Name: {data_dict.get('name', 'N/A')}")
                        print(f"         Passport #: {data_dict.get('passport_number', 'N/A')}")
                    elif class_name == 'receipt':
                        print(f"         Vendor: {data_dict.get('vendor_name', 'N/A')}")
                        print(f"         Amount: ${data_dict.get('total_amount', 0):.2f}")
                    elif class_name == 'itinerary':
                        print(f"         Route: {data_dict.get('departure_city', 'N/A')} → {data_dict.get('arrival_city', 'N/A')}")
                    elif class_name == 'delay_notification':
                        print(f"         Flight: {data_dict.get('flight_number', 'N/A')}")
                        print(f"         Reason: {data_dict.get('delay_reason', 'N/A')}")
                else:
                    print(f"      ❌ Failed to extract {class_name}")
    
    print("\n" + "=" * 60)
    print("TOTAL DOCUMENTS")
    print("-" * 40)
    for doc_type, docs in supporting_docs.items():
        print(f"  • {doc_type}: {len(docs)} document(s)")
    # print(supporting_docs)
    print("=" * 60)
    
    return {
        "supporting_documents_data": supporting_docs,
        "document_queue": [],
        "document_filepath": ""
    }