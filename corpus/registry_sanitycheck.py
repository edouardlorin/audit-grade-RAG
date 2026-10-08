import yaml, pathlib
r = yaml.safe_load(open(r'C:\rag\corpus\registry.yaml', encoding='utf-8'))
for d in r['documents']:
    p = pathlib.Path(r'C:\rag\corpus') / d['file']
    print(d['doc_id'], d['edition'], '| file exists:', p.exists())