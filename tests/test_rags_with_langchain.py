import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric

# INDEXING
reader = PdfReader("/Users/B.Masoko/PycharmProjects/PythonProject/DeepEval/data/GeCompras_Handbook.pdf")
pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_text(pdf_text)

client = chromadb.Client()
col = client.create_collection("docs")  # uses a default embedding model
col.add(documents=chunks, ids=[f"c{i}" for i in range(len(chunks))])

# QUERY
q = "What is GeCompras all about?"
results = col.query(query_texts=[q], n_results=4)
context = "\n\n".join(results["documents"][0])

prompt = f"Answer using only this context:\n{context}\n\nQuestion: {q}"


# send prompt to Claude/GPT → grounded answer

def test_rag_contextual_recall():
    contextual_recall_metric = ContextualRecallMetric(threshold=0.5)

    test_case = LLMTestCase(
        input="What is GeCompras all about?",
        actual_output="GeCompras is a comprehensive guide to the GeCompras system and its various features.",
        expected_output="GeCompras is a comprehensive guide to the GeCompras system and its various features.",
        retrieval_context=[context],
    )

    assert_test(test_case, [contextual_recall_metric])
