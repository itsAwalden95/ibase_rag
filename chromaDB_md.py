import sys
from enum import auto
from enum import Enum
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions
from chromadb import Collection
from transformers import AutoTokenizer, AutoModel
import torch
import psutil
import time
import os
import re
from langchain.text_splitter import MarkdownHeaderTextSplitter

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)

class CollectionStatus(Enum):
    COLLECTION_CREATED = auto()
    COLLECTION_EXISTS = auto()


def ensure_collection(client: chromadb.ClientAPI, collection_name: str) -> tuple[str, Optional[Collection]]:
    try:
        # Check if the collection already exists
        collection = client.get_collection(name=collection_name, embedding_function=sentence_transformer_ef)
        print(f"Collection '{collection_name}' already exists.")
        return "COLLECTION_EXISTS", collection
    except Exception:
        # If it doesn't exist, create a new collection
        try:
            collection = client.create_collection(name=collection_name, embedding_function=sentence_transformer_ef)
            print(f"Collection '{collection_name}' created successfully.")
            return "COLLECTION_CREATED", collection
        except Exception as e:
            print(f"Failed to create collection '{collection_name}': {e}")
            return "COLLECTION_CREATION_FAILED", None


def clean_text(raw_text: str) -> str:
    # Clean up the text to remove extra spaces and line breaks
    cleaned_text = raw_text.replace("\n", " ")
    cleaned_text = re.sub(r"\s+", " ", raw_text)
    return cleaned_text


# Insert a document into a collection
def insert_document(document_path: Path, collection: Collection) -> None:
    """
    Reads a markdown file, splits it into chunks, generates embeddings,
    and inserts the chunks into a ChromaDB collection.
    """
    # Read the markdown file content
    with open(document_path, 'r') as file:
        markdown_content = file.read()
    
    #markdown_content = clean_text(markdown_content)

    text = splitter.split_text(markdown_content)    
    
    document_name = document_path.stem.replace(" ", "-").replace("_", "-")

    # Get chunks of text from the markdown content
    document_chunks = []
    document_ids = []

    for chunk_index, (chunk) in enumerate(text):
    
        # Append results
        document_ids.append(f"{document_name}_chunk{chunk_index}")
        document_chunks.append(chunk.page_content)
    
    collection.add(
                documents=document_chunks,
                ids=document_ids
            )
    


## Folder_path = path to e config files
def load_files_into_chroma(folder_path, client: chromadb.ClientAPI, collection_name):
    # Iterate over files in the folder

    collection_status, collection = ensure_collection(client, collection_name)


    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Only process text files (or adapt as needed for other file types)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            # Open and read the content of each file
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            


            # Add the file content to the Chroma collection as a new chunk
            collection.add(
                documents=[file_content],
                metadatas=[{"filename": filename}],
                ids=[filename]
            )
            print(f"Added {filename} to Chroma database.")



def main() -> None:
    base_directory = Path(os.getcwd())
    db_directory = Path("./db")
    files_directory = Path("./db_files_md")  # Folder containing markdown files

    if not db_directory.exists():
        db_directory.mkdir()

    if not files_directory.exists():
        print("DB files were not copied! Abort.")
        sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=str(db_directory))

    ## Inserting e config files firt

    #load_files_into_chroma("./db_config", chroma_client, "e_config_files")



    # Define the groups of files (based on your example)
    file_groups = [
        "E1080.md",  # First collection should come from E1080_md. Goes into POWER10 collection
        "E1050.md",  # Second collection should come from E1050_md. Goes into POWER10 collection
        "S1012.md",  # Third collection should come from S1020_md. Goes into POWER10 collection
        "ScaleOut.md",  # Fourth collection should come from Scale Out. Goes into POWER10 collection
        "Openshift.md", # Fifth collection for Openshift
        "Ansible.md" # Sixth collection should come from Ansible
    ]

    # Iterate over the file groups and create a collection for each
    for i, file_name in enumerate(file_groups):
        print("Adding collection group", i + 1)

        # Adding all POWER10 files into the POWER10 collection
        if (file_name == "E1080.md" or file_name == "E1050.md" or file_name == "S1012.md" or file_name == "ScaleOut.md"):
            collection_name = "POWER10"
        # Remove the last three characters for the collection name
        else:
            collection_name = file_name[:-3]
        print(collection_name)

        # Ensure the collection exists or create it
        collection_status, collection = ensure_collection(chroma_client, collection_name)

        if collection_status == CollectionStatus.COLLECTION_EXISTS:
            print(f"Collection '{collection_name}' already exists. Skipping file insertion.")
        else:
            print(f"Creating collection '{collection_name}' and inserting documents.")
            
            # Process the file
            document_path = files_directory / file_name
            if document_path.exists():
                insert_document(document_path, collection)
                print(f"Inserted {file_name} into {collection_name}")
            else:
                print(f"File {file_name} not found!")
            
            # Delay between operations

    print("Setup completed.")

    # Example query for testing
    result = collection.query(
        query_texts=["What is IBM POWER"],
        n_results=5,
        include=["documents"]
    )
    print(result)


if __name__ == "__main__":
    main()
                                                       
