import json
from typing import Dict, Optional
from difflib import SequenceMatcher

def load_customer_database(db_path: str = "src/tests/customers.json") -> Dict:
    try:
        with open(db_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Database file {db_path} not found.")
        return {"customers": []}

def match_name(name1: str, name2: str, threshold: float = 0.8) -> bool:
    if not name1 or not name2:
        return False
    
    norm1 = ' '.join(name1.lower().split())
    norm2 = ' '.join(name2.lower().split())
    
    # Direct comparison
    if SequenceMatcher(None, norm1, norm2).ratio() >= threshold:
        return True
    
    # Try reversed order (First Last vs Last, First)
    parts1 = norm1.split()
    parts2 = norm2.split()
    
    if len(parts1) >= 2 and len(parts2) >= 2:
        reversed1 = f"{parts1[-1]} {parts1[0]}"
        if SequenceMatcher(None, reversed1, norm2).ratio() >= threshold:
            return True
    
    return False

def verify_customer_node(state):
    customer_data = state.get('customer_data')
    claim_data = state.get('claim_data')
    if not customer_data:
        print("Please reupload the claim form. No data is detected.")
        return {"verification_status": "NO_DATA"}
    
    extracted_name = customer_data.name
    extracted_policy = claim_data.policy_number
    
    if not extracted_name or not extracted_policy:
        print("Extracted name", extracted_name)
        print("Extracted policy number", extracted_policy)
        print("Please reupload the claim form. Either the name or policy number cannot be extracted.")
        return {"verification_status": "INCOMPLETE_DATA"}
    
    # Load database
    database = load_customer_database()
    customers = database.get('customers', [])
    
    for customer in customers:
        if customer.get('status') != 'active':
            continue
            
        db_name = customer.get('name', '')
        db_policy = customer.get('policy_number', '')
        
        name_matches = match_name(extracted_name, db_name)
        policy_matches = extracted_policy == db_policy
        
        if name_matches and policy_matches:
            return {
                "verification_status": "VERIFIED",
            }
    
    print("Please reupload the claim form. No matching customer found.")
    return {"verification_status": "NO_MATCH"}