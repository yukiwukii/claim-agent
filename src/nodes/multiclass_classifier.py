from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

# Pass 1: Segmentation

class DocumentSegment(BaseModel):
    segment_id: int = Field(
        description="A unique identifier for this segment, starting from 1"
    )
    start_line: int = Field(
        description="The starting line number of this document segment"
    )
    end_line: int = Field(
        description="The ending line number of this document segment"
    )
    brief_description: str = Field(
        description="A brief description of what this segment appears to contain"
    )

class DocumentSegmentation(BaseModel):
    segments: List[DocumentSegment] = Field(
        description="List of distinct document segments found in the document"
    )

def segment_document(document_content: str, required_documents: List[str], claim_type: str) -> List[DocumentSegment]:
    
    # Number the lines for reference
    numbered_content = "\n".join([f"{i+1}: {line}" for i, line in enumerate(document_content.splitlines())])
    
    all_possible_classes = required_documents + ["irrelevant", 'claim_form']
    
    prompt = f"""
    You are an expert at analyzing document structure for travel insurance claims.
    
    Claim Type: {claim_type}
    
    Possible document types you might encounter:
    {', '.join(all_possible_classes)}
    
    Your task is to identify distinct logical sections or sub-documents within the provided text.
    Each segment should represent a cohesive piece of information that belongs together.
    
    Document Content (with line numbers):
    ---
    {numbered_content}
    ---
    
    Instructions:
    1. Identify natural boundaries between different types of content
    2. Each segment should be self-contained and logically complete
    3. Do not split related information (e.g., don't split a single receipt into multiple segments)
    4. If you see multiple receipts, each should be its own segment
    5. If the entire document appears to be one cohesive unit, return a single segment
    6. Segments should not overlap
    7. Segments should cover the entire document (no gaps)
    8. Use your knowledge of the possible document types to better identify boundaries
    """
    
    chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
    structured_llm = chat.with_structured_output(DocumentSegmentation)
    messages = [HumanMessage(content=prompt)]
    
    try:
        response = structured_llm.invoke(messages)
        return response.segments
    except Exception as e:
        print(f"Error in document segmentation: {e}")
        # Return the entire document as a single segment if segmentation fails
        return [DocumentSegment(
            segment_id=1,
            start_line=1,
            end_line=len(document_content.splitlines()),
            brief_description="Complete document"
        )]

# Pass 2: Classification

def create_dynamic_classification_model(valid_classes: List[str]) -> type:
    all_possible_classes = valid_classes + ["irrelevant", "claim_form"]
    ClassesLiteral = Literal[tuple(all_possible_classes)]
    
    class SegmentClassification(BaseModel):
        segment_id: int = Field(
            description="The ID of the segment being classified"
        )
        document_classes: List[ClassesLiteral] = Field(
            description=f"List of document classes that apply to this segment. Valid classes: {', '.join(all_possible_classes)}"
        )
    
    return SegmentClassification

def classify_segment(segment_text: str, segment_info: DocumentSegment, 
                     required_documents: List[str], claim_type: str) -> Dict[str, Any]:
    
    DynamicModel = create_dynamic_classification_model(required_documents)
    all_possible_classes = required_documents + ["irrelevant", "claim_form"]
    
    prompt = f"""
    You are an expert document classifier for travel insurance claims.
    
    Claim Type: {claim_type}
    
    You are analyzing a segment of a larger document.
    Segment Description: {segment_info.brief_description}
    Segment Lines: {segment_info.start_line} to {segment_info.end_line}
    
    Your task is to classify this segment into one or more of these categories:
    {', '.join(all_possible_classes)}
    
    Segment Content:
    ---
    {segment_text}
    ---
    
    Instructions:
    1. A segment can belong to multiple categories if it contains information for multiple document types
    2. For example, a single page might contain both a trip itinerary and a receipt
    3. Classify as 'irrelevant' if the segment doesn't match any of the required document types
    4. Be conservative: only assign classes you're confident about
    """
    
    chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
    structured_llm = chat.with_structured_output(DynamicModel)
    messages = [HumanMessage(content=prompt)]
    
    try:
        response = structured_llm.invoke(messages)
        return {
            "segment_id": segment_info.segment_id,
            "start_line": segment_info.start_line,
            "end_line": segment_info.end_line,
            "document_classes": response.document_classes,
            "description": segment_info.brief_description
        }
    except Exception as e:
        print(f"⚠️ Error classifying segment {segment_info.segment_id}: {e}")
        return {
            "segment_id": segment_info.segment_id,
            "start_line": segment_info.start_line,
            "end_line": segment_info.end_line,
            "document_classes": ["irrelevant"],
            "description": segment_info.brief_description
        }

def extract_lines_from_document(doc_content: str, start_line: int, end_line: int) -> str:
    doc_lines = doc_content.splitlines()
    start_idx = max(0, start_line - 1)
    end_idx = min(len(doc_lines), end_line)
    
    return "\n".join(doc_lines[start_idx:end_idx])

def process_document_two_pass(document_content: str, required_documents: List[str], 
                              claim_type: str) -> List[Dict[str, Any]]:
    
    print("Pass 1: Segmenting document...")
    segments = segment_document(document_content, required_documents, claim_type)
    print(f"   Found {len(segments)} segment(s)")
    print(f"   Segments: {[f'Segment {seg.segment_id}: ({seg.start_line}-{seg.end_line})' for seg in segments]}")
    classifications = []    
    
    print("Pass 2: Classifying segments...")
    for segment in segments:
        segment_text = extract_lines_from_document(document_content, segment.start_line, segment.end_line)
        classification = classify_segment(segment_text, segment, required_documents, claim_type)
        
        if classification["document_classes"] != ["irrelevant"]:
            classifications.append(classification)
            print(f"   Segment {segment.segment_id}: {classification['document_classes']}")
    
    return classifications

def load_document_from_path(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Document file {filepath} not found.")
    except Exception as e:
        raise Exception(f"Error reading document {filepath}: {e}")

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
        filename = os.path.basename(filepath_or_name) if filepath_or_name else "current_document"
        print(f"\nProcessing: {filename}")
        print("-" * 40)
        
        classifications = process_document_two_pass(document_content, required_documents, claim_type)
        
        if filepath_or_name:
            all_classifications[filepath_or_name] = classifications
            if filepath_or_name not in processed_documents:
                processed_documents.append(filepath_or_name)
        
        for classification in classifications:
            document_classes.update(classification['document_classes'])
        
        if not classifications:
            print("   ⚠️ No relevant classifications found")
            
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
        print(f"✅ Queue processing complete. Processed {len(document_queue)} documents.")
    
    elif current_document:
        _process_single_document_and_update_state(
            current_document, document_filepath, required_documents, claim_type,
            all_classifications, processed_documents, document_classes
        )
        print("\n" + "=" * 60)
    
    else:
        print("⚠️ No document to process (no queue and no current document)")
    
    state.update({
        "document_classes": document_classes,
        "processed_documents": processed_documents,
        "all_classifications": all_classifications,
    })
    
    return state