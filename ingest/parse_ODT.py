from parse import parse_pdf

b = parse_pdf(r'C:\rag\corpus\raw\CS-25 (Amendment 27).pdf')
h = [x for x in b if x.kind=='heading']
print('blocks', len(b), '| headings', len(h), '| tables',
      sum(1 for x in b if x.kind=='table'))
for x in h[:15]: print(f'  p{x.pdf_page:>4}  L{x.level}  {x.clause_no:<18} {x.clause_title[:52]}')
print('  ...')
for x in h[-10:]: print(f'  p{x.pdf_page:>4}  L{x.level}  {x.clause_no:<18} {x.clause_title[:52]}')
