import gradio as gr
import chromadb
from llama_cpp import Llama
from chromadb.utils import embedding_functions
import sys
import traceback

# Initialize ChromaDB client with explicit error handling
try:
    print("Initializing ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path="./db")
    print("ChromaDB client initialized successfully")
except Exception as e:
    print(f"Error initializing ChromaDB client: {e}")
    sys.exit(1)

# Initialize sentence transformer
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

# Initialize LLM with explicit error handling
try:
    print("\nLoading LLM model...")
    llm = Llama(
        model_path="/home/cecuser/models/granite-7b-lab-Q4_K_M.gguf",
        n_ctx=4096,
        verbose=True
    )
    print("LLM model loaded successfully")
except Exception as e:
    print(f"Error loading LLM model: {e}")
    sys.exit(1)

def generate_response(query, chat_history):
    """Generate responses with detailed error logging."""
    print(f"\nReceived query: {query}")
    try:
        # Get collection
        try:
            collection = chroma_client.get_collection(
                name="install_base", 
                embedding_function=sentence_transformer_ef
            )
            print(f"Collection accessed successfully. Size: {collection.count()}")
        except Exception as e:
            print(f"Error accessing collection: {e}")
            chat_history.append((query, "Error: Database access failed"))
            return "", chat_history

        # Query collection
        try:
            print("Querying collection...")
            results = collection.query(
                query_texts=[query],
                n_results=5,
                include=['documents', 'metadatas']
            )
            print(f"Query returned {len(results['documents'][0])} results")
        except Exception as e:
            print(f"Error during query: {e}")
            chat_history.append((query, "Error: Search failed"))
            return "", chat_history

        # Prepare prompt
        context = "\n".join(results['documents'][0]) if results['documents'] and results['documents'][0] else ""
        prompt = f"""
Context:
{context}

Question:
{query}

Instructions:
Provide a detailed answer based on the context above.
If the information isn't in the context, say so.

Answer:
"""
        print("Generating LLM response...")
        
        try:
            # Generate response
            output = llm(
                prompt,
                max_tokens=2048,
                temperature=0.7,
                stop=["Question:", "Context:"],
                echo=False
            )
            
            if output and 'choices' in output and output['choices']:
                response = output['choices'][0]['text'].strip()
                print(f"Generated response: {response[:100]}...")  # Print first 100 chars
                chat_history.append((query, response))
            else:
                print("No response generated from LLM")
                chat_history.append((query, "Error: No response generated"))
                
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            traceback.print_exc()
            chat_history.append((query, f"Error during response generation: {str(e)}"))
            
        return "", chat_history

    except Exception as e:
        print(f"Unexpected error in generate_response: {e}")
        traceback.print_exc()
        chat_history.append((query, f"An error occurred: {str(e)}"))
        return "", chat_history

def main():
    # Test database connection
    try:
        collection = chroma_client.get_collection(
            name="install_base", 
            embedding_function=sentence_transformer_ef
        )
        print(f"\nDatabase initialized. Collection size: {collection.count()}")
    except Exception as e:
        print(f"Warning: Could not access collection: {e}")

    # Create the Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# IBM POWER Server Install Base Assistant")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Query")
        clear = gr.Button("Clear")
        state = gr.State([])
        
        msg.submit(generate_response, [msg, state], [msg, chatbot])
        clear.click(lambda: ([], []), None, [chatbot, state], queue=False)

    # Launch with debug mode enabled
    demo.launch(
        server_name="0.0.0.0", 
        server_port=8082, 
        enable_queue=True,
        
    )

if __name__ == "__main__":
    main()
