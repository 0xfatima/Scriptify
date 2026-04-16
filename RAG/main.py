from pdf_loader import extract_text_with_metadata
from embedder import embed_texts
from vector_store import add_documents, search
from llm import generate_answer

PDF_PATH = "2007.02343v2.pdf"


def chunk_text(text, chunk_size=500, overlap=100):

    sentences = text.split(". ")

    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "

    if current:
        chunks.append(current.strip())

    return chunks

def build_index(pdf_path):

    print("Loading PDF...")

    data = extract_text_with_metadata(pdf_path)

    all_chunks = []
    pages = []

    for d in data:
        chunks = chunk_text(d["text"])

        for c in chunks:
            all_chunks.append(c)
            pages.append(d["page"])

    print("Creating embeddings...")

    embeddings = embed_texts(all_chunks)

    print("Storing in Chroma DB...")

    add_documents(all_chunks, embeddings, pages)

    print("Index ready!\n")


def chat_loop():

    print("RAG system ready. Type 'exit' to quit.\n")

    while True:

        query = input("You: ")
        if query.lower() == "exit":
            break

        query_embedding = embed_texts([query])[0]

        retrieved = search(query_embedding, k=6)

        print("\n--- Retrieved Context ---")
        for r in retrieved:
            print(f"[Page {r['page']}] {r['text'][:200]}\n")

        answer = generate_answer(query, retrieved)

        print("\nAnswer:", answer, "\n")


if __name__ == "__main__":
    build_index(PDF_PATH)
    chat_loop()