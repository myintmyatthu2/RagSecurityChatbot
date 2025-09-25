# src/rag_chain.py
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from typing import Any, Union

def build_rag_chain(
    llm: Union[BaseLLM, BaseChatModel],
    retriever: BaseRetriever,
    chain_type: str = "stuff",
    return_source_documents: bool = True,
    num_questions: int = 5  # <-- New parameter for number of quiz questions
) -> Any:
    """
    Builds and returns a LangChain RetrievalQA chain.
    If num_questions is set, output will be formatted as that many quiz questions with answers.
    """
    print(f"[INFO] Building RAG chain with chain_type='{chain_type}', num_questions={num_questions}...")

    # Prompt template with num_questions as variable
    custom_prompt_template = """
Answer the question based on the context below.

- Short to the main point
- Relevant and reasonable answer
- Human readable and understandable
- Explain using example for the main point related to the question (if possible, use Myanmar case study)
- Do not show source links or website links

- If the user requests a quiz:
  - Format the answer as a quiz with multiple choice questions (a, b, c, d).
  - Generate exactly {num_questions} question{'s' if num_questions > 1 else ''} with answers.
  - Focus on cybersecurity, AI, or related topics.
  - Keep questions clear, concise, and educational.


Context:
{context}

Question: {question}

Answer:
"""


    QA_CHAIN_PROMPT = PromptTemplate.from_template(custom_prompt_template)

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=return_source_documents,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        print("[INFO] RetrievalQA chain built successfully.")
        return qa_chain
    except Exception as e:
        print(f"[ERROR] Error building RAG chain: {e}")
        raise