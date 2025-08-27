from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

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
    ClassesLiteral = Literal[tuple(valid_classes)]
    
    class DynamicDocumentClassification(BaseModel):
        document_classes: List[ClassesLiteral] = Field(
            description=f"List of document classes from: {', '.join(valid_classes)}"
        )
    
    return DynamicDocumentClassification

def classify_single_document(document_content: str, required_documents: List[str], claim_type: str) -> List[str]:
    DynamicModel = create_dynamic_classification_model(required_documents)
    
    prompt = f"""
    You are a document classifier for travel insurance claims.
    
    Claim Type: {claim_type}
    
    Classify the following document into one or more of these categories:
    {', '.join(required_documents)}
    
    Document Content:
    {document_content}
    
    Instructions:
    - A document can belong to multiple categories if it contains information for multiple document types
    - Only use the categories listed above for this specific claim type
    - Be conservative with classifications - only assign a class if you're reasonably confident
    """

    chat = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)
    structured_llm = chat.with_structured_output(DynamicModel)
    messages = [HumanMessage(content=prompt)]
    
    try:
        response = structured_llm.invoke(messages)
        return response.document_classes
    except Exception as e:
        print(f"Error in document classification: {e}")
        return []

def multiclass_classifier(state: Dict[str, Any]) -> Dict[str, Any]:
    document_queue = state.get("document_queue", [])
    current_document = state.get("current_document", "")
    document_filepath = state.get("document_filepath", "")
    required_documents = state.get("required_documents", [])
    claim_type = state.get("claim_type", "")
    processed_documents = state.get("processed_documents", [])
    
    all_classifications = state.get("all_classifications", {})
    
    if not required_documents:
        raise ValueError("No required documents found in state. Claim type must be determined first.")
    
    if not claim_type:
        raise ValueError("Claim type must be determined before document classification.")
    
    if document_queue:
        print(f"\nProcessing document queue ({len(document_queue)} documents)...")
        print("=" * 60)
        document_classes = state.get("document_classes", set())

        for filepath in document_queue:
            if filepath in all_classifications:
                continue
            try:
                document_content = load_document_from_path(filepath)
                classification = classify_single_document(document_content, required_documents, claim_type)
                
                filename = os.path.basename(filepath)
                print(f"\n✅ Classified: {filename}")
                print(f"   Classes: {classification}")
                
                all_classifications[filepath] = classification
                document_classes.update(classification)
                
                if filepath not in processed_documents:
                    processed_documents.append(filepath)
                
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(filepath)}: {e}")
        
        print("\n" + "=" * 60)
        print(f"Queue processing complete. Processed {len(processed_documents)} documents.")
        
        return {
            "document_classes": document_classes,
            "document_queue": [],
            "processed_documents": processed_documents,
            "all_classifications": all_classifications
        }
    
    else:
        if not current_document:
            print("⚠️  No document to process (no queue and no current document)")
            return {}
        
        classification = classify_single_document(current_document, required_documents, claim_type)
        
        filename = os.path.basename(document_filepath) if document_filepath else "current_document"
        print(f"\n✅ Classified: {filename}")
        print(f"   Classes: {classification}")
        
        if document_filepath:
            all_classifications[document_filepath] = classification
        
        if document_filepath not in processed_documents:
            processed_documents.append(document_filepath)
        
        document_classes = state.get("document_classes", set())
        document_classes.update(classification)
        # document_classes = set(dict.fromkeys([item for sublist in all_classifications.values() for item in sublist]))
        print(f"Document classes: {document_classes}")
        print(f"Required documents: {set(state['required_documents'])}")
        print("=" * 60)
        
        return {
            "document_classes": document_classes,
            "processed_documents": processed_documents,
            "all_classifications": all_classifications
        }