from src.nodes.binary_classifier import binary_classifier
from src.nodes.extract_claim_form import extract_claim_form_node
from src.nodes.verify_customer import verify_customer_node

from typing import TypedDict, Optional, Dict, Literal
from datetime import date
from decimal import Decimal
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langsmith import Client
import os
from uuid import uuid4

with open("src/tests/txts/claim_form.txt", "r") as file:
    document = file.read()

class GraphState(TypedDict):
    current_document: str
    customer_data: Optional[Dict]
    claim_data: Optional[Dict]
    verification_status: Literal["NO_DATA", "INCOMPLETE_DATA", "VERIFIED", "NO_MATCH"]

def create_graph_state() -> GraphState:
    return {
        "current_document": document,
        "customer_data": None,
        "claim_data": None,
        "verification_status": "NO_DATA"
    }

def claim_form_or_not(state: GraphState) -> str:
    result = binary_classifier(state)
    return "extract_claim_form_node" if result.claim_form_or_not else END

def setup_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)
    
    workflow.add_node("classifier", binary_classifier)
    workflow.add_node("extract_claim_form_node", extract_claim_form_node)
    workflow.add_node("verify_customer_node", verify_customer_node)
    
    workflow.add_edge(START, "classifier")
    workflow.add_conditional_edges("classifier", claim_form_or_not)
    workflow.add_edge("extract_claim_form_node", "verify_customer_node")
    workflow.add_edge("verify_customer_node", END)
    
    return workflow

def run_workflow():    
    workflow = setup_workflow()
    app = workflow.compile()
    
    initial_state = create_graph_state()
    
    result = app.invoke(initial_state)
    
    print("Workflow Results:")
    result_check = binary_classifier(initial_state).claim_form_or_not
    print(f"Is Claim Form: {result_check}")
    
    if result_check:  # Extract when it IS a claim form
        print("\nCustomer Data:")
        customer_data = result.get('customer_data', {})
        print(customer_data)
        
        print("\nClaim Data:")
        claim_data = result.get('claim_data', {})
        print(claim_data)

        print("\nVerification Status:")
        print(result.get('verification_status', ''))
    else:
        print("Document is not a claim form - no extraction performed")

if __name__ == "__main__":
    run_workflow()