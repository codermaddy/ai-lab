from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool as retriever_tool_decorator

BACKEND_BASE_URL = "http://localhost:8000"

def build_retriever_from_backend() -> "Chroma":
    """
    Fetch run summaries from the backend and build a Chroma-based retriever.

    This is where you consume Person1 manifest-derived summaries indirectly.
    """
    resp = requests.get(f"{BACKEND_BASE_URL}/runs/summaries", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # Expected structure:
    # [{"run_id": "...", "summary": "Text summary derived from manifest & W&B", ...}, ...]

    docs: List[Document] = []
    for item in data:
        docs.append(
            Document(
                page_content=item["summary"],
                metadata={"run_id": item["run_id"], "dataset": item.get("dataset")},
            )
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})


@retriever_tool_decorator("search_run_summaries")
def search_run_summaries_tool(query: str):
    """
    Retrieve text summaries of runs relevant to the user's query.

    Use this for fuzzy, high-level questions like:
    - 'What worked best on CIFAR-10?'
    - 'Summarize my experiments this month.'
    Always combine this with get_run_details for exact metrics.
    """
    # In a real app you might cache the retriever or refresh periodically.
    retriever = build_retriever_from_backend()
    results = retriever.get_relevant_documents(query)
    return [
        {
            "run_id": doc.metadata.get("run_id"),
            "dataset": doc.metadata.get("dataset"),
            "summary": doc.page_content,
        }
        for doc in results
    ]
