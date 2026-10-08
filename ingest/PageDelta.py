import os, random
from qdrant_client import QdrantClient
c = QdrantClient(url='http://127.0.0.1:6333', api_key=os.environ['QDRANT_API_KEY'])
pts, off = [], None
while True:
    p, off = c.scroll('standards_v1', limit=512, offset=off, with_payload=True)
    if not p: break
    pts += [x.payload for x in p]
    if off is None: break
for d in random.sample(pts, 5):
    print({d['clause_no']:<18} printed p.{str(d['page_start']):<6} PDF p.{d['pdf_page_start']:<6} delta={None if d['page_start'] is None else d['page_start']-d['pdf_page_start']})