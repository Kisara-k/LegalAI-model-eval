"""
Legal queries for RAG evaluation.
Each query represents a realistic legal information need.
"""

LEGAL_QUERIES = [
    {
        "id": 1,
        "query": "What are the procedures for presidential elections when the office becomes vacant?",
        "category": "Constitutional Law",
        "keywords": ["presidential election", "vacancy", "procedure", "parliament"],
    },
    {
        "id": 2,
        "query": "What are the tax obligations and payment requirements for businesses with quarterly turnover?",
        "category": "Tax Law",
        "keywords": ["tax obligations", "quarterly turnover", "payment", "business"],
    },
    {
        "id": 3,
        "query": "What are the penalty provisions for bribery and undue influence in elections?",
        "category": "Electoral Law",
        "keywords": ["penalty", "bribery", "undue influence", "election offences"],
    },
    {
        "id": 4,
        "query": "What is the process for filing appeals to the Board of Review and Court of Appeal?",
        "category": "Procedural Law",
        "keywords": ["appeals", "Board of Review", "Court of Appeal", "procedure"],
    },
    {
        "id": 5,
        "query": "What are the voting procedures and ballot requirements for parliamentary elections?",
        "category": "Electoral Law",
        "keywords": ["voting procedure", "ballot", "preference voting", "election"],
    },
    {
        "id": 6,
        "query": "How are taxes assessed and collected, and what powers do assessors have?",
        "category": "Tax Law",
        "keywords": ["tax assessment", "collection", "assessor powers", "returns"],
    },
    {
        "id": 7,
        "query": "What are the establishment provisions and governance structure of the Defence Academy?",
        "category": "Administrative Law",
        "keywords": ["Defence Academy", "establishment", "Board of Management", "Commandant"],
    },
    {
        "id": 8,
        "query": "What are the import duty requirements and customs procedures for articles manufactured abroad?",
        "category": "Customs Law",
        "keywords": ["import duty", "customs", "manufactured abroad", "levy"],
    },
    {
        "id": 9,
        "query": "What is the nomination process and requirements for presidential candidates?",
        "category": "Constitutional Law",
        "keywords": ["nomination", "presidential candidates", "written consent", "seconding"],
    },
    {
        "id": 10,
        "query": "What are the powers and functions of the Commissioner-General in tax administration?",
        "category": "Tax Law",
        "keywords": ["Commissioner-General", "powers", "tax administration", "delegation"],
    },
]


def get_query_by_id(query_id: int) -> dict:
    """Get a specific query by ID."""
    for query in LEGAL_QUERIES:
        if query["id"] == query_id:
            return query
    raise ValueError(f"Query with ID {query_id} not found")


def get_all_queries() -> list:
    """Get all queries as a list."""
    return LEGAL_QUERIES


def get_queries_by_category(category: str) -> list:
    """Get all queries for a specific category."""
    return [q for q in LEGAL_QUERIES if q["category"] == category]


def get_query_texts() -> list:
    """Get just the query text strings."""
    return [q["query"] for q in LEGAL_QUERIES]
