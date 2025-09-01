from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from collections import defaultdict

load_dotenv()

def load_document_from_path(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Document file {filepath} not found.")
    except Exception as e:
        raise Exception(f"Error reading document {filepath}: {e}")

def create_dynamic_classification_model(valid_classes: List[str]) -> type:
    all_possible_classes = valid_classes + ["irrelevant"]
    ClassesLiteral = Literal[tuple(all_possible_classes)]
    
    class ClassificationWithEvidence(BaseModel):
        document_class: ClassesLiteral = Field(
            description="The classification of the document segment."
        )
        start_line: int = Field(
            description="The starting line number of the text segment that supports this classification. Use 0 if the class is 'irrelevant'."
        )
        end_line: int = Field(
            description="The ending line number of the text segment that supports this classification. Use 0 if the class is 'irrelevant'."
        )
        
    class DynamicDocumentClassification(BaseModel):
        classifications: List[ClassificationWithEvidence] = Field(
            description=f"A list of classifications found in the document. Valid classes are: {', '.join(all_possible_classes)}"
        )
    
    return DynamicDocumentClassification

def classify_single_document(document_content: str, required_documents: List[str], claim_type: str) -> List[Dict[str, Any]]:
    DynamicModel = create_dynamic_classification_model(required_documents)
    
    all_possible_classes = required_documents + ["irrelevant"]
    
    # Help number the text
    numbered_content = "\n".join([f"{i+1}: {line}" for i, line in enumerate(document_content.splitlines())])
    
    prompt = f"""
    You are an expert document classifier for travel insurance claims.

    Claim Type: {claim_type}

    Your task is to classify the document below into one or more of the following categories:
    {', '.join(all_possible_classes)}

    Document Content (with line numbers):
    ---
    {numbered_content}
    ---

    Instructions:
    1.  Carefully read the document content.
    2.  For each category you identify, you MUST provide the start and end line numbers for the block of text that contains the evidence for that classification.
    3.  A document can belong to multiple categories if it contains distinct sections of information. Each distinct section should be its own classification entry.
    4.  Make sure that there is no overlap between the evidence blocks for different classifications.
    5.  If the document is a claim form, or not from the list of categories, classify it as 'irrelevant'.
    6.  Be conservative: only assign a class if you are confident and can point to a specific block of lines.
    """

    chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
    structured_llm = chat.with_structured_output(DynamicModel)
    messages = [HumanMessage(content=prompt)]
    
    try:
        response = structured_llm.invoke(messages)
        return [c.dict() for c in response.classifications]
    except Exception as e:
        print(f"Error in document classification: {e}")
        return []

def _merge_classifications(classifications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges multiple classifications of the same type into a single entry
    spanning the minimum start line to the maximum end line."""
    
    if not classifications:
        return []

    grouped_by_class = defaultdict(list)
    for c in classifications:
        grouped_by_class[c['document_class']].append(c)

    merged_results = []
    for doc_class, class_list in grouped_by_class.items():
        if doc_class == 'irrelevant' or len(class_list) == 1:
            merged_results.extend(class_list)
            continue
        
        min_start = min(c['start_line'] for c in class_list)
        max_end = max(c['end_line'] for c in class_list)
        
        merged_results.append({
            "document_class": doc_class,
            "start_line": min_start,
            "end_line": max_end
        })
        
    return merged_results

def _process_single_document_and_update_state(
    document_content: str,
    filepath_or_name: str,
    required_documents: List[str],
    claim_type: str,
    all_classifications: Dict,
    processed_documents: List,
    document_classes: set
) -> None:
    try:
        raw_classifications = classify_single_document(document_content, required_documents, claim_type)
        
        classifications_with_evidence = _merge_classifications(raw_classifications)
        
        filename = os.path.basename(filepath_or_name) if filepath_or_name else "current_document"
        print(f"\n✅ Classified: {filename}")
        if classifications_with_evidence:
            for classification in classifications_with_evidence:
                print(f"   - Class: {classification['document_class']}")
                print(f"     Lines: {classification['start_line']}-{classification['end_line']}")
        else:
            print("   - No classifications found.")
            
        if filepath_or_name:
            all_classifications[filepath_or_name] = classifications_with_evidence
            if filepath_or_name not in processed_documents:
                processed_documents.append(filepath_or_name)
        
        new_classes = {c['document_class'] for c in classifications_with_evidence}
        document_classes.update(new_classes)
        
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(filepath_or_name)}: {e}")

def multiclass_classifier(state: Dict[str, Any]) -> Dict[str, Any]:
    document_queue = state.get("document_queue", [])
    current_document = state.get("current_document", "")
    document_filepath = state.get("document_filepath", "")
    required_documents = state.get("required_documents", [])
    claim_type = state.get("claim_type", "")
    processed_documents = state.get("processed_documents", [])
    all_classifications = state.get("all_classifications", {})
    document_classes = state.get("document_classes", set())
    
    if not required_documents or not claim_type:
        raise ValueError("Claim type and required documents must be determined first.")

    if document_queue:
        print(f"\nProcessing document queue ({len(document_queue)} documents)...")
        print("=" * 60)
        for filepath in document_queue:
            if filepath in all_classifications:
                continue
            document_content = load_document_from_path(filepath)
            _process_single_document_and_update_state(
                document_content, filepath, required_documents, claim_type,
                all_classifications, processed_documents, document_classes
            )
        
        print("\n" + "=" * 60)
        print(f"Queue processing complete. Processed {len(document_queue)} documents.")
        state.update({
            "document_queue": [],
        })
    
    elif current_document:
        _process_single_document_and_update_state(
            current_document, document_filepath, required_documents, claim_type,
            all_classifications, processed_documents, document_classes
        )
        print("\n" + "=" * 60)
        state.update({
            "current_document": "", 
            "document_filepath": ""
        })

    else:
        print("⚠️ No document to process (no queue and no current document)")

    state.update({
        "document_classes": document_classes,
        "processed_documents": processed_documents,
        "all_classifications": all_classifications,
    })
    return state