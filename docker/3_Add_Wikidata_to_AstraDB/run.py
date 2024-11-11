import sys
import requests
import time
import os
import pickle

sys.path.append("../src")

from wikidataDB import Session, WikidataID, WikidataEntity  # type: ignore
from wikidataEmbed import WikidataTextifier, JinaAIEmbeddings  # type: ignore

from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from astrapy.info import CollectionVectorServiceOptions
from transformers import AutoTokenizer
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy.sql import func


load_dotenv()


# Load the environment variables
MODEL = os.getenv("MODEL", "nvidia")
SAMPLE = os.getenv("SAMPLE", "false").lower() == "true"
SAMPLE_SIZE = os.getenv("SAMPLE_SIZE", 1000)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 500))
OFFSET = int(os.getenv("OFFSET", 0))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Error if the collection name is not provided
if not COLLECTION_NAME:
    raise ValueError("The COLLECTION_NAME environment variable is required")

# Load the AstraDB environment variables
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

# Initialize the textifier
textifier = WikidataTextifier(with_claim_aliases=False, with_property_aliases=False)
sample_pkl_path = "../data/Evaluation Data/Sample IDs (EN).pkl"

# Initialize the graph store and tokenizer
graph_store = None
tokenizer = None
max_token_size = None

# Initialize the graph store
if MODEL == "nvidia":
    print("Using the NVIDIA model")

    # Load the embeddings
    tokenizer = AutoTokenizer.from_pretrained(
        "intfloat/e5-large-unsupervised",
        trust_remote_code=True,
        clean_up_tokenization_spaces=False,
        max_length=max_token_size,
        truncation=True,
    )
    max_token_size = 500

    # Initialize the collection vector service options
    collection_vector_service_options = CollectionVectorServiceOptions(
        provider="nvidia", model_name="NV-Embed-QA"
    )

    # Initialize the graph store
    graph_store = AstraDBVectorStore(
        collection_name=COLLECTION_NAME,
        collection_vector_service_options=collection_vector_service_options,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        namespace=ASTRA_DB_KEYSPACE,
    )
else:
    print("Using the Jina model")

    # Load the embeddings
    embeddings = JinaAIEmbeddings(embedding_dim=1024)
    tokenizer = embeddings.tokenizer
    max_token_size = 1024

    # Initialize the graph store
    graph_store = AstraDBVectorStore(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        namespace=ASTRA_DB_KEYSPACE,
    )

# Assume we aren't sampling
sample_ids: list[int] = []
with Session() as session:
    sample_ids_count = session.query(func.count(WikidataEntity.id)).scalar()

# Load the Sample IDs
if SAMPLE:
    # Check if the file exists
    if not os.path.exists(sample_pkl_path):
        # Load the Sample IDs
        with Session() as session:
            # Modify the query to fetch random 100 IDs
            sample_ids_query = (
                session.query(WikidataEntity.id)
                .join(WikidataID, WikidataEntity.id == WikidataID.id)
                .filter(WikidataID.in_wikipedia)
                .limit(SAMPLE_SIZE)  # Limit to SAMPLE_SIZE results
            )

            # Fetch and append IDs to the list
            for entity in sample_ids_query:
                sample_ids.append(entity.id)
    else:
        # Load the Sample IDs
        with open(sample_pkl_path, "rb") as f:
            sample_ids = pickle.load(f)

    sample_ids_count = len(sample_ids)

if __name__ == "__main__":
    with tqdm(total=sample_ids_count) as progressbar:
        with Session() as session:
            # Conditionally apply filter based on sample_ids
            query = (
                session.query(WikidataEntity)
                .join(WikidataID, WikidataEntity.id == WikidataID.id)
                .filter(WikidataID.in_wikipedia)
            )

            # If sample_ids is non-empty, apply the filter
            if SAMPLE:
                query = query.filter(WikidataEntity.id.in_(sample_ids))

            # Apply offset and batch size as required
            entities = query.offset(OFFSET).yield_per(BATCH_SIZE)
            progressbar.update(OFFSET)

            # Initialize the batch
            doc_batch = []
            ids_batch = []

            for entity in entities:
                progressbar.update(1)

                # Chunk the text and add to the batch
                chunks = textifier.chunk_text(
                    entity, tokenizer, max_length=max_token_size
                )

                # Add the chunks to the batch
                for i, chunk in enumerate(chunks):

                    # Create the document
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "QID": entity.id,
                            "ChunkID": i + 1,
                            "aliases": entity.aliases,
                            "label": entity.label,
                            "description": entity.description,
                        },
                    )

                    # Add the document to the batch
                    doc_batch.append(doc)
                    ids_batch.append(f"{entity.id}_{i + 1}")

                    # If the batch is full, add it to the graph store
                    if len(doc_batch) < BATCH_SIZE:
                        continue

                    # Update the progress bar
                    tqdm.write(
                        progressbar.format_meter(
                            progressbar.n,
                            progressbar.total,
                            progressbar.format_dict["elapsed"],
                        )
                    )  # tqdm is not working in docker compose. This is the alternative
                    try:
                        graph_store.add_documents(doc_batch, ids=ids_batch)

                        doc_batch = []
                        ids_batch = []
                    except Exception as e:
                        print(e)
                        while True:
                            try:
                                # Check for internet connection
                                response = requests.get(
                                    "https://www.google.com", timeout=5
                                )
                                if response.status_code == 200:
                                    break
                            except Exception as e:
                                print("Waiting for internet connection...")
                                time.sleep(5)

            # Add the remaining documents
            if len(doc_batch) > 0:
                graph_store.add_documents(doc_batch, ids=ids_batch)
