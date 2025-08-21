from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal, List, Dict
import yaml
import os

class ClaimTypeClassification(BaseModel):
    """Classification of claim type based on claim form data"""
    claim_type: Literal[
        "travel_delay",
        "baggage_delay", 
        "trip_cancellation",
        "trip_interruption"
    ] = Field(description="Type of insurance claim identified from the claim form")

def load_claim_config(config_path: str = "src/utils/files_required.yml") -> Dict:
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

def get_claim_type_document_classes(config_path: str = "src/utils/files_required.yml") -> Dict[str, List[str]]:
    config = load_claim_config(config_path)
    
    claim_type_classes = {}
    for claim_type, claim_config in config['claim_types'].items():
        claim_type_classes[claim_type] = claim_config['document_classes']
    
    return claim_type_classes

def claim_type_classifier(state, config_path: str = "src/utils/files_required.yml"):
    claim_data = state.get("claim_data", {})
    if not claim_data:
        raise ValueError("Claim form must be processed before claim type classification.")
    
    reason_for_claim = getattr(claim_data, 'reason_for_claim', None)
    
    if not reason_for_claim:
        raise ValueError("No reason_for_claim found in claim data.")
    
    prompt = f"""
    Based on the following claim information, classify the type of insurance claim.

    Claim reason: {reason_for_claim}

    Valid claim types:
    - travel_delay: Claims for expenses due to flight/travel delays
    - baggage_delay: Claims for delayed baggage and related expenses  
    - trip_cancellation: Claims for cancelled trips before departure
    - trip_interruption: Claims for trips that were interrupted after departure

    Classify this as exactly one of the above claim types based on the reason provided.
    """

    chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
    structured_llm = chat.with_structured_output(ClaimTypeClassification)
    messages = [HumanMessage(content=prompt)]
    response = structured_llm.invoke(messages)
    
    claim_type_classes = get_claim_type_document_classes(config_path)
    
    claim_type = response.claim_type
    valid_document_classes = claim_type_classes[claim_type]
    
    return {
        "claim_type": claim_type,
        "valid_document_classes": valid_document_classes,
        "claim_type_classification": response
    }

def get_document_classes_for_claim_type(claim_type: str, config_path: str = "claim_config.yaml") -> List[str]:
    """
    Utility function to get valid document classes for a given claim type
    """
    claim_type_classes = get_claim_type_document_classes(config_path)
    
    if claim_type not in claim_type_classes:
        raise ValueError(f"{claim_type} is not in the files_required.yml config file.")
    
    return claim_type_classes[claim_type]

def setup_claim_classification_workflow():
    from langgraph.graph import StateGraph, START, END
    
    # This would be part of your larger workflow
    workflow = StateGraph(GraphState)  # Your existing GraphState
    
    # Add the claim type classifier node
    workflow.add_node("claim_type_classifier", claim_type_classifier)
    
    # Route to structured output with claim type context
    def route_to_structured_output(state):
        claim_type = state.get("claim_type")
        if claim_type:
            return "structured_output"
        else:
            return END
    
    # Add conditional edge
    workflow.add_conditional_edges(
        "claim_type_classifier",
        route_to_structured_output,
        {
            "structured_output": "structured_output",
            END: END
        }
    )
    
    return workflow

# Example of how structured output node would use this information
def structured_output_with_claim_context(state):
    """
    Example of how the structured output node would use claim type information
    """
    document = state.get("current_document")
    claim_type = state.get("claim_type")
    valid_classes = state.get("valid_document_classes", [])
    
    # Now the structured output can be tailored to the specific claim type
    # and only consider the valid document classes for that claim type
    
    prompt = f"""
    Extract structured information from this document for a {claim_type} claim.
    
    This document should be one of: {', '.join(valid_classes)}
    
    Document: {document}
    """
    
    # Continue with structured output logic...
    return {"structured_output": "..."}