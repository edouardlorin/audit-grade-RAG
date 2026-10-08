from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

print(client.__version__)

"""
search_result = client.query_points(
    collection_name="standards_v1",
    query="ignition",
    with_payload=False,
    limit=3
).points

print(search_result)

"""