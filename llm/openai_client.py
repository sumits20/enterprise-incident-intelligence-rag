import os
from openai import OpenAI

def generate_answer(question: str, evidence_text: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not found."

    client = OpenAI(api_key=api_key)

    prompt = f"""
    You are an enterprise incident intelligence assistant.
    
    Rules:
    - Use ONLY the evidence provided.
    - Do NOT invent facts.
    - If evidence is insufficient, say so clearly.
    
    Return the answer in this structure:
    1. Likely resolution time (Bold) (hours or range)
    2. Responsible team (Bold)
    3. Recommended resolution steps (Bold) (bullet points)
    4. Evidence used (Bold) (Incident ID + Date)
    
    QUESTION:
    {question}
    
    EVIDENCE:
    {evidence_text}
    """.strip()

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )

    return response.output_text
