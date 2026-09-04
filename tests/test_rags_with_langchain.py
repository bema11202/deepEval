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
collection = client.get_or_create_collection("chromadb_docs")  # uses a default embedding model
collection.add(documents=chunks, ids=[f"c{i}" for i in range(len(chunks))])
collection = client.get_collection(name="chromadb_docs")  # retrieve the collection for querying
# print(collection.peek())  # peek at the first few documents in the collection
# all_data = collection.get()
# QUERY
q = "What is GeCompras all about?"
results = collection.query(query_texts=[q], n_results=4)
context = "\n\n".join(results["documents"][0])  # concatenate the retrieved documents into a single context string

prompt = f"Answer using only this context:\n{context}\n\nQuestion: {q}"


# send prompt to Claude/GPT → grounded answer

def test_rag_contextual_recall():
    contextual_recall_metric = ContextualRecallMetric(threshold=0.5)

    test_case = LLMTestCase(
        input=q,
        actual_output="GeCompras is a comprehensive guide to the GeCompras system and its various features.",
        expected_output="GeCompras is a comprehensive guide to the GeCompras system and its various features.",
        retrieval_context=[context],
    )

    assert_test(test_case, [contextual_recall_metric])


# Add additional tests for other metrics as needed for GeCompras handbook evaluation.

def test_rag_contextual_recall_with_incorrect_answer():
    contextual_recall_metric = ContextualRecallMetric(threshold=0.5)

    test_case = LLMTestCase(
        input="What types of payments does GeCompras support?",
        actual_output="GeCompras supports various payment methods including credit cards, bank transfers, and digital wallets.",
        expected_output="GeCompras supports various payment methods including credit cards, bank transfers, and digital wallets.",
        retrieval_context=[context],
    )

    assert not contextual_recall_metric.measure(test_case)


def test_rag_contextual_relevance():
    contextual_recall_metric = ContextualRecallMetric(threshold=0.5)

    test_case = LLMTestCase(
        input="What is the purpose of GeCompras?",
        actual_output="The purpose of GeCompras is to provide a comprehensive guide to the GeCompras system and its various features.",
        expected_output="The purpose of GeCompras is to provide a comprehensive guide to the GeCompras system and its various features.",
        retrieval_context=[context],
    )
    assert_test(test_case, [contextual_recall_metric])

