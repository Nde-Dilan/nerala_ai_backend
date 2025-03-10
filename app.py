from flask import Flask, request, jsonify
import os
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# Initialize Flask app
app = Flask(__name__)

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is required")
genai.configure(api_key=GOOGLE_API_KEY)

# Load Chroma vector DB
VECTOR_DB_PATH = "./vector_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
vector_store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model)

# Define the query endpoint
@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    # Retrieve relevant chunks from vector DB
    results = vector_store.similarity_search(user_query, k=5)
    retrieved_texts = [doc.page_content for doc in results]

    if not retrieved_texts:
        return jsonify({"response": "No relevant information found"}), 200

    # Prepare prompt for Gemini
    prompt = f"""
    You are an AI assistant. Based on the following retrieved information, answer the user's query.
    Information:
    {retrieved_texts}

    User Query:
    {user_query}
    """

    try:
        # Send request to Gemini
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        answer = response.text if response.text else "No relevant response generated."

        return jsonify({"response": answer})
    
    except Exception as e:
        return jsonify({"error": f"LLM processing failed: {str(e)}"}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
