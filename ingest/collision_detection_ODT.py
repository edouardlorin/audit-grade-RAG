import collections, yaml, pathlib
from parse import parse_pdf
from chunk import chunk_document

m = yaml.safe_load(open(r'C:\rag\corpus\registry.yaml', encoding='utf-8'))['documents'][0]
p = str(pathlib.Path(r'C:\rag\corpus') / m['file'])
cs = chunk_document(parse_pdf(p, page_offset=m.get('page_offset'), skip_tables=True), m, 'x')
seen = collections.defaultdict(list)
for c in cs: seen[c.chunk_id].append(c)
d = {k:v for k,v in seen.items() if len(v)>1}
print('colliding ids:', len(d))
for v in list(d.values())[:12]:
    print(f'  {v[0].clause_no!r:<20} part {v[0].part} x{len(v)}  pdf pages {[c.pdf_page_start for c in v]}')