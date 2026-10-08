import os
from qdrant_client import QdrantClient
c = QdrantClient(url="http://127.0.0.1:6333", api_key=os.environ["QDRANT_API_KEY"])
pts, off = [], None
while True:
    p, off = c.scroll("standards_v1", limit=512, offset=off, with_payload=True)
    if not p: break
    pts += [x.payload for x in p if x.payload["clause_no"].lower().startswith("Appendix")]
    if off is None: break

print(pts)

for d in sorted(pts, key=lambda x: x["part"]):
    print("  part {d['part']}/{d['n_parts']}  p{d['page_start']}  {len(d['body']):>5} chars")
    print("     {d['body'][:100]}")
    