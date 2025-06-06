from dagster import asset, Definitions
from dagster_aws.s3 import S3PickleIOManager, S3Resource

from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete

WORKING_DIR = "./cache"

@asset
def raw_text() -> str:
    """Load the text document to index."""
    with open("text.txt", "r", encoding="utf-8") as f:
        return f.read()

@asset
def pathrag_index(raw_text: str) -> str:
    """Insert the document into the knowledge base."""
    rag = PathRAG(working_dir=WORKING_DIR, llm_model_func=gpt_4o_mini_complete)
    rag.insert(raw_text)
    return "indexed"

@asset
def pathrag_answer(pathrag_index: str) -> str:
    """Query the knowledge base after indexing."""
    rag = PathRAG(working_dir=WORKING_DIR, llm_model_func=gpt_4o_mini_complete)
    return rag.query("What is PathRAG?", QueryParam(mode="hybrid"))


defs = Definitions(
    assets=[raw_text, pathrag_index, pathrag_answer],
    resources={
        "io_manager": S3PickleIOManager(
            s3_resource=S3Resource(), s3_bucket="your-bucket", s3_prefix="pathrag"
        ),
        "s3": S3Resource(),
    },
)
