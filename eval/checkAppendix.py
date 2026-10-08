import os, collections
from qdrant_client import QdrantClient
c = QdrantClient(url='http://127.0.0.1:6333', api_key=os.environ['QDRANT_API_KEY'])
n, chars, off = collections.Counter(), collections.Counter(), None

while True:
    pts, off = c.scroll('standards_v1', limit=512, offset=off, with_payload=True)
    if not pts: break
    for p in pts:
        cl = p.payload['clause_no']
        if cl.lower().startswith('appendix'):
            n[cl] += 1
            chars[cl] += len(p.payload.get('body',''))
    if off is None: break

for k in sorted(n):
    print(f'  {k:<14} {n[k]:>4} chunks  {chars[k]:>8} chars  '
          f'{chars[k]//max(n[k],1):>5} avg')
