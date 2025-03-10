import os
import argparse
from main import PDFProcessor

def interactive_query_session(processor, file_path):
    """Start an interactive query session."""
    print(f"Interactive query session for: {file_path}")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
            
        results = processor.query_document(file_path, query, k=3)
        
        if not results:
            print("No results found.")
            continue
            
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description="Interactive PDF querying")
    parser.add_argument("--file", "-f", required=True, help="Path to the PDF file")
    parser.add_argument("--process", "-p", action="store_true", help="Force processing the PDF")
    
    args = parser.parse_args()
    
    processor = PDFProcessor()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        return
        
    # Process file if requested or if not already processed
    file_hash = processor.get_file_hash(args.file)
    persist_directory = os.path.join(processor.config["db_directory"], file_hash)
    
    if args.process or not os.path.exists(persist_directory):
        print(f"Processing file: {args.file}")
        processor.process_file(args.file)
    
    # Start interactive query session
    interactive_query_session(processor, args.file)

if __name__ == "__main__":
    main()

# python interactive_test.py --file "./Dictionnaire-Fulfulde-français-english-et-images.pdf"

# python interactive_test.py --file "./Dictionnaire-Fulfulde-français-english-et-images.pdf" --process
