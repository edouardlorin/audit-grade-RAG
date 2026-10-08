import os, random
from qdrant_client import QdrantClient
c = QdrantClient(url='http://127.0.0.1:6333', api_key=os.environ['QDRANT_API_KEY'])
pts, off = [], None
while True:
    p, off = c.scroll('standards_v1', limit=512, offset=off, with_payload=True)
    if not p: break
    pts += [x.payload for x in p if x.payload.get('page_start')]
    if off is None: break
for d in random.sample(pts, 3):
    print(f">>> OPEN THE PDF AT PAGE {d['pdf_page_start']}")
    print(f"    citation claims : {d['clause_no']} , printed p. {d['page_start']}")
    print(f"    text should read: {d['body'][:170]}...")
