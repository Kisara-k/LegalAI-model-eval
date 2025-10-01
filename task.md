Task:
Evaluate approaches to building a RAG pipeline.

Dataset:
data/acts_with_metadata.tsv
Contains chunked legal acts (content field), with extracted metadata fields (short_title, keywords, section_title, summary).

Approach:
Join the 4 metadata fields into one structured text field.
Design 10 legal queries.
Retrieve relavant documents for each query
Seperately for content field and joined metadata field (2 lists).

Retriever:
Get the 2 lists for each retriever seperately
BM25
FAISS

Test FAISS seperately with 3 embedding models, one of which needs to be Legal BERT, and others need to be cutting edge models that are well suited for this task. (search the web)

For FAISS, create a seperate notebook that will build the faiss indices for content and metadata in a GPU environment. Testing will be done on CPU, loading these built FAISS indexes.

