from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL


def build_context(chunks):
    if not chunks:
        return None

    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        context_parts.append(
            f"Context {i+1} "
            f"(Source: {meta['source']}, "
            f"Page: {meta['page']}, "
            f"Type: {meta['type']}):\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def get_unique_sources(chunks):
    """
    Returns the list of unique source filenames present in the
    retrieved chunks. Used to tell the LLM how many reports are
    in the context so it knows whether to compare across them.
    """
    seen = []
    for chunk in chunks:
        src = chunk["metadata"]["source"]
        if src not in seen:
            seen.append(src)
    return seen


def generate_answer(query, chunks):
    if not chunks:
        return (
            "I could not find relevant information in the "
            "uploaded reports to answer this question."
        )

    context = build_context(chunks)
    sources = get_unique_sources(chunks)
    client = Groq(api_key=GROQ_API_KEY)

    # When more than one report is in the context, we add an extra
    # instruction telling the LLM to compare across them explicitly.
    # When only one report is present, that instruction is omitted
    # so the prompt stays clean and focused.
    if len(sources) > 1:
        source_list = ", ".join(sources)
        cross_doc_instruction = (
            f"\n\nThe context contains information from {len(sources)} "
            f"different reports: {source_list}. "
            "If the question can be answered by comparing across these "
            "reports, do so explicitly — for example: "
            "'According to [report A]... whereas [report B] states...'. "
            "Always name the specific report when citing a finding."
        )
    else:
        cross_doc_instruction = ""

    system_prompt = (
        "You are a financial research assistant analyzing equity research "
        "reports. Answer questions based strictly on the provided context "
        "from the reports. Always cite your sources by mentioning the "
        "source file and page number. If the answer is not present in the "
        "context, say 'This information is not available in the provided "
        "reports.' Never use information outside the provided context."
        + cross_doc_instruction
    )

    user_prompt = f"""Context from equity research reports:

{context}

Question: {query}

Answer based only on the above context, citing sources:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content