import fitz
d = fitz.open(r'C:\rag\corpus\raw\CS-25 (Amendment 27).pdf')
print('total PDF pages:', len(d))
for i in (200, 600, 900, 1312):
    txt = d[i].get_text('text').strip().splitlines()
    print(f'--- pdf page {i+1}')
    print('   foot:', repr(' '.join(txt[-3:])[-110:]))