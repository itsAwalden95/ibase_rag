import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, List, Optional
import os
import re
from pathlib import Path

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="ibm-granite/granite-embedding-125m-english")

def process_install_base(excel_path: str, client: chromadb.ClientAPI) -> None:
    """
    Process the install base Excel file using pandas and create a ChromaDB collection.
    """
    collection_name = "install_base"
    
    try:
        # Check if file exists
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found at path: {excel_path}")
            
        # Read Excel file using pandas
        print(f"Reading Excel file from: {excel_path}")
        df = pd.read_excel(excel_path)
        print(f"Successfully read {excel_path}")
        print(f"Found columns: {df.columns.tolist()}")
        
        # Create or get collection
        try:
            collection = client.get_collection(name=collection_name, embedding_function=sentence_transformer_ef)
            print(f"Using existing collection '{collection_name}'")
        except:
            collection = client.create_collection(name=collection_name, embedding_function=sentence_transformer_ef)
            print(f"Created new collection '{collection_name}'")

        # Process each row of the dataframe
        for idx, row in df.iterrows():
            # Convert row to dictionary and clean the data
            row_data = {str(k): str(v).strip() for k, v in row.to_dict().items() if pd.notna(v)}
            
            # Create searchable text representation
            text_repr = "\n".join([f"{key}: {value}" for key, value in row_data.items()])

            # Store in ChromaDB
            collection.add(
                documents=[text_repr],
                metadatas=[row_data],
                ids=[f"install_base_{idx}"]
            )
            
        print(f"Processed {len(df)} rows of data")
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

def query_install_base(query: str, client: chromadb.ClientAPI) -> Dict:
    """
    Query the install base data and return relevant information.
    """
    collection = client.get_collection(name="install_base", embedding_function=sentence_transformer_ef)
    
    # Determine query type and extract relevant information
    if re.search(r'serial.*number.*for', query, re.I):
        # Query for customer's serial numbers
        customer = re.search(r'for\s+([^?]+)', query, re.I)
        if customer:
            customer_name = customer.group(1).strip()
            results = collection.query(
                query_texts=[f"Customer: {customer_name}"],
                n_results=100  # Adjust based on expected maximum installations per customer
            )
            return format_customer_serials(results, customer_name)
            
    elif re.search(r'(machine|model).*type.*for.*\d{6,7}', query, re.I):
        # Query for specific machine information
        serial = re.search(r'\b(\d{6,7})\b', query)
        if serial:
            results = collection.query(
                query_texts=[f"Serial Number: {serial.group(1)}"],
                n_results=1
            )
            return format_machine_info(results, serial.group(1))
            
    # General query
    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=['metadatas', 'documents']
    )
    return format_general_results(results)

def format_customer_serials(results: Dict, customer_name: str) -> Dict:
    """Format results for customer serial number queries."""
    serials = []
    if results['metadatas']:
        for metadata in results['metadatas'][0]:
            if 'Serial Number' in metadata:
                serials.append(metadata['Serial Number'])
    
    return {
        'type': 'customer_serials',
        'customer': customer_name,
        'serials': serials,
        'response': f"Serial numbers for {customer_name}:\n" + "\n".join(serials) if serials else f"No serial numbers found for {customer_name}"
    }

def format_machine_info(results: Dict, serial: str) -> Dict:
    """Format results for machine information queries."""
    if results['metadatas'] and results['metadatas'][0]:
        metadata = results['metadatas'][0][0]
        info = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
        return {
            'type': 'machine_info',
            'serial': serial,
            'info': metadata,
            'response': f"Information for serial number {serial}:\n{info}"
        }
    return {
        'type': 'machine_info',
        'serial': serial,
        'info': None,
        'response': f"No information found for serial number {serial}"
    }

def format_general_results(results: Dict) -> Dict:
    """Format results for general queries."""
    return {
        'type': 'general',
        'results': results,
        'response': "\n\n".join(results['documents'][0]) if results['documents'] else "No relevant information found"
    }

if __name__ == "__main__":
    # Specify your paths here
    INSTALL_BASE_FILE = "/home/cecuser/github/RAG_public/db_files/install_27.xlsx"  # <-- UPDATE THIS PATH
    DB_PATH = "./db"
    
    # Create DB directory if it doesn't exist
    os.makedirs(DB_PATH, exist_ok=True)
    
    print(f"Using install base file: {INSTALL_BASE_FILE}")
    print(f"Using database path: {DB_PATH}")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=DB_PATH)
   # Test query to verify data
    try:
        collection = client.get_collection(name="install_base", embedding_function=sentence_transformer_ef)
        print(f"\nTesting collection:")
        print(f"Collection size: {collection.count()} documents")
        
        test_results = collection.query(
            query_texts=["test query"],
            n_results=1
        )
        print("Test query successful")
        print(f"Sample document: {test_results['documents'][0] if test_results['documents'] else 'No results'}")
    except Exception as e:
        print(f"Test query failed: {e}") 
    # Process the install base
    process_install_base(INSTALL_BASE_FILE, client)
