import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class SyftrRetrieverFunctionConfig(FunctionBaseConfig, name="syftr_retriever"):
    """
    Syftr Retriever Function Configuration
    This function retrieves relevant documents from the FinanceBench dataset
    using a sparse index and returns the top-k documents based on the input query.
    It uses the BM25 algorithm for retrieval and supports custom LLM configurations.
    The function is designed to be used within the AIQ framework and can be
    registered as a workflow function.
    """

    # Add your custom configuration parameters here
    llm_for_tokenization: str = Field(
        default="gpt-4o-mini",
        description="LLM whose tokenizer will be used for splitting text",
    )
    top_k: int = Field(
        default=5,
        description="Number of top documents to retrieve from the sparse index",
    )


@register_function(config_type=SyftrRetrieverFunctionConfig)
async def syftr_retriever_function(
    config: SyftrRetrieverFunctionConfig, builder: Builder
):
    import Stemmer
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.schema import TransformComponent

    from syftr.storage import FinanceBenchHF
    from syftr.llm import get_tokenizer

    documents = FinanceBenchHF().iter_grounding_data(partition="test")
    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        tokenizer=get_tokenizer(config.llm_for_tokenization),
    )
    transforms: list[TransformComponent] = [splitter]

    # Build sparse index
    pipeline = IngestionPipeline(transformations=transforms)
    nodes = pipeline.run(documents=documents, show_progress=True)
    docstore = StorageContext.from_defaults().docstore
    docstore.add_documents(nodes)
    retriever = BM25Retriever.from_defaults(
        nodes=list(docstore.docs.values()),
        similarity_top_k=config.top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    # Implement your function logic here
    async def _response_fn(input_message: str) -> str:
        # Process the input_message and generate output
        nodes = await retriever.aretrieve(input_message)
        nodes_text = [node.text for node in nodes]
        return " ".join(nodes_text)

    try:
        yield FunctionInfo.create(single_fn=_response_fn)
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up syftr_retriever workflow.")
