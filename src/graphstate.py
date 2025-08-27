from src.nodes.binary_classifier import binary_classifier
from src.nodes.extract_claim_form import extract_claim_form_node
from src.nodes.verify_customer import verify_customer_node
from src.nodes.multiclass_classifier import multiclass_classifier
from src.nodes.structured_output import structured_output_node
from src.nodes.extract_claim_form import CustomerData, ClaimData

from typing import TypedDict, Optional, Dict, Literal, List, Any, Set
from datetime import date
from decimal import Decimal
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langsmith import Client
import os
import glob
import random
from uuid import uuid4
import json


class GraphState(TypedDict):
    # Document processing
    current_document: str
    document_filepath: Optional[str]
    
    # Extracted data
    customer_data: Optional[CustomerData]
    claim_data: Optional[ClaimData]
    supporting_documents_data: Dict[str, Any]
    
    # Claim type and requirements
    claim_type: Optional[Literal["travel_delay", "baggage_delay", "trip_cancellation", "trip_interruption"]]
    required_documents: List[str]
    
    # Verification status
    verification_status: Optional[Literal["NO_DATA", "INCOMPLETE_DATA", "VERIFIED", "NO_MATCH"]]
    
    # Document queue and processing
    document_queue: List[str]
    incomplete_documents: List[str]
    ## Below the three things are very similar, we could potentially merge them
    processed_documents: List[str]
    document_classes: Optional[Set[str]]
    all_classifications: Optional[Dict[str, List[str]]]
    
    # Session tracking
    session_id: Optional[str]

def load_document_from_path(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Document file {filepath} not found.")
    except Exception as e:
        raise Exception(f"Error reading document {filepath}: {e}")

def create_graph_state(document_filepath = None) -> GraphState:
    document_content = ""
    if document_filepath:
        try:
            document_content = load_document_from_path(document_filepath)
        except Exception as e:
            print(f"Warning: Could not load document {document_filepath}: {e}")
    
    return {
        "current_document": document_content,
        "document_filepath": document_filepath,
        "customer_data": None,
        "claim_data": None,
        "claim_type": None,
        "incomplete_documents": [],
        "supporting_documents_data": {},
        "required_documents": [],
        "verification_status": None,
        "document_queue": [],
        "processed_documents": [],
        "document_classes": set(),
        "all_classifications": {},
        "session_id": str(uuid4())
    }

def ocr_node(state: GraphState) -> Dict:
    document_filepath = state.get("document_filepath", "")
    if document_filepath and not state.get("current_document"):
        try:
            document_content = load_document_from_path(document_filepath)
            print(f"Placeholder for OCR processing...")
            return {"current_document": document_content}
        except Exception as e:
            print(f"OCR failed for {document_filepath}: {e}")
            return {"current_document": state.get("current_document", "")}
    
    return {"current_document": state.get("current_document", "")}

def has_claim_form_been_received(state: GraphState) -> str:
    """Route to binary classifier if no claim_data, else to multiclass classifier"""
    claim_data = state.get("claim_data")
    
    if not claim_data:
        return "binary_classifier"
    else:
        return "multiclass_classifier"

def route_after_binary_classification(state: GraphState) -> str:
    """Route after binary classification"""
    try:
        result = binary_classifier(state)
        if result.claim_form_or_not:
            return "extract_claim_form"
        else:
            return "add_to_queue"
    except Exception as e:
        print(f"Error in binary classification: {e}")
        return "add_to_queue"

def add_to_queue_node(state: GraphState) -> Dict:
    """Add current document to queue if it's not the claim form"""
    document_filepath = state.get("document_filepath", "")
    document_queue = state.get("document_queue", [])
    
    if document_filepath and document_filepath not in document_queue:
        document_queue.append(document_filepath)
        print(f"Added {os.path.basename(document_filepath)} to document queue. Queue now has {len(document_queue)} documents.")
    
    return {"document_queue": document_queue}

def nullify_claim_form_node(state: GraphState) -> Dict:
    """Reset claim form data if verification fails"""
    print("Customer verification failed. Nullifying claim form data.")
    return {
        "customer_data": None,
        "claim_data": {},
        "claim_type": None,
        "required_documents": [],
        "verification_status": None,
    }

def route_after_verification(state: GraphState) -> str:
    """Route after customer verification"""
    verification_status = state.get("verification_status")
    document_queue = state.get("document_queue", [])
    
    if verification_status == "VERIFIED":
        if len(document_queue) > 0:
            print(f"✅ Claim form verified! Now processing {len(document_queue)} queued documents...")
            return "multiclass_classifier"
        else:
            return END
    else:
        return "nullify_claim_form"
    
def is_results_clear_node(state: GraphState) -> Dict:
    supporting_docs = state.get("supporting_documents_data", {})
    required_documents = state.get("required_documents", [])
    
    incomplete_documents = []
    keys_to_remove = []
    
    for doc_type, data_list in supporting_docs.items():
        is_empty = False
        if not data_list:
            is_empty = True
        else:
            for data_item in data_list:
                if not data_item or any(not value for value in data_item.values()):
                    is_empty = True
                    break
        
        if is_empty:
            if doc_type in required_documents:
                incomplete_documents.append(doc_type)
            else:
                keys_to_remove.append(doc_type)
    
    updated_supporting_docs = supporting_docs.copy()
    for key in keys_to_remove:
        del updated_supporting_docs[key]
        print(f"🗑️ Removed empty non-required document type: {key}")
    
    if incomplete_documents:
        print(f"❌ Unclear required documents: {incomplete_documents}. Please re-upload.")
        for doc in incomplete_documents:
            if doc in updated_supporting_docs:
                del updated_supporting_docs[doc]
                # print(f"Removed unclear document: {doc}")
    else:
        print("✅ All required documents have valid data")
    
    return {
        "supporting_documents_data": updated_supporting_docs,
        "incomplete_documents": incomplete_documents
    }

def are_all_checklists_complete_node(state: GraphState) -> Dict:
    required_documents = state.get("required_documents", [])
    document_classes = state.get("document_classes", set())

    missing_documents = set(required_documents) - document_classes

    if not missing_documents:
        print("✅ All required documents have been received. Move on to stage 2.")
    else:
        print(f"❌ Missing documents: {missing_documents}")
    
    return {
        "missing_documents": list(missing_documents)
    }

def route_after_results_clear(state: GraphState) -> str:
    """Routing function after is_results_clear node"""
    incomplete_documents = state.get("incomplete_documents", [])
    
    if incomplete_documents:
        return END
    else:
        return "are_all_checklists_complete"

def route_after_checklists_complete(state: GraphState) -> str:
    # Placeholder until stage 2 is implemented.
    return END

def setup_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)
    
    workflow.add_node("ocr", ocr_node)
    workflow.add_node("binary_classifier", binary_classifier)
    workflow.add_node("multiclass_classifier", multiclass_classifier)
    workflow.add_node("extract_claim_form", extract_claim_form_node)
    workflow.add_node("verify_customer", verify_customer_node)
    workflow.add_node("add_to_queue", add_to_queue_node)
    workflow.add_node("nullify_claim_form", nullify_claim_form_node)
    workflow.add_node("structured_output", structured_output_node)
    workflow.add_node("is_results_clear", is_results_clear_node)
    workflow.add_node("are_all_checklists_complete", are_all_checklists_complete_node) 
    
    workflow.add_edge(START, "ocr")
    
    workflow.add_conditional_edges(
        "ocr", 
        has_claim_form_been_received,
        {
            "binary_classifier": "binary_classifier",
            "multiclass_classifier": "multiclass_classifier"
        }
    )
    
    workflow.add_conditional_edges(
        "binary_classifier", 
        route_after_binary_classification,
        {
            "extract_claim_form": "extract_claim_form",
            "add_to_queue": "add_to_queue"
        }
    )
    
    workflow.add_edge("multiclass_classifier", "structured_output")
    
    workflow.add_edge("extract_claim_form", "verify_customer")
    
    workflow.add_conditional_edges(
        "verify_customer", 
        route_after_verification,
        {
            "multiclass_classifier": "multiclass_classifier",
            "nullify_claim_form": "nullify_claim_form",
            END: END
        }
    )
    
    workflow.add_edge("add_to_queue", END)
    workflow.add_edge("nullify_claim_form", END)
    workflow.add_edge("structured_output", "is_results_clear")
    
    workflow.add_conditional_edges(
        "is_results_clear",
        route_after_results_clear,
        {
            "are_all_checklists_complete": "are_all_checklists_complete",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "are_all_checklists_complete",
        route_after_checklists_complete,
        {
            END: END
        }
    )

    return workflow

def run_workflow_with_document(document_filepath: str, existing_state: Optional[GraphState] = None):
    load_dotenv()
    
    workflow = setup_workflow()
    app = workflow.compile()
    
    if existing_state:
        initial_state = {
            **existing_state,
            "current_document": "",
            "document_filepath": document_filepath
        }
    else:
        initial_state = create_graph_state(document_filepath)
    
    print(f"Processing document upload: {os.path.basename(document_filepath)}")
    print("=" * 50)
    
    try:
        result = app.invoke(initial_state)
        return result
        
    except Exception as e:
        print(f"Error running workflow: {e}")
        return None

def simulate_document_uploads():
    print("Starting Document Upload Simulation")
    print("=" * 60)

    folder_path = "src/tests/test_cases/all_ok_anon/"
    test_files = glob.glob(os.path.join(folder_path, "*.txt"))
    random.shuffle(test_files)
    
    state = None
    for i, filepath in enumerate(test_files, 1):
        print(f"\nUpload #{i}: {os.path.basename(filepath)}")
        print("-" * 40)
        
        try:
            state = run_workflow_with_document(filepath, state)
        except FileNotFoundError:
            print(f"⚠️ File not found: {filepath}")
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")

if __name__ == "__main__":
    simulate_document_uploads()