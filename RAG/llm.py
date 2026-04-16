def generate_answer(query, retrieved_chunks):
    cleaned_chunks = []

    for c in retrieved_chunks:
        text = c["text"].strip()
        if len(text) < 10:
            continue
        cleaned_chunks.append(f"[p.{c.get('page', 0)}] {text}")

    context = "\n\n".join(cleaned_chunks[:10])

    from model_singleton import generate_rag_answer

    return generate_rag_answer(query=query, context=context)