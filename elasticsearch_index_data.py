import sys
sys.path.append('/home/mohamed/PycharmProjects/nordlys')
from nordlys.core.retrieval.elastic import Elastic
import os
import json

wikitables_folder='wikitables'
file_to_index='mcon_predictions.json'

path=os.path.join(wikitables_folder,file_to_index)
index_name = "mcon_index"

with open(path) as f:
    docs = json.load(f)

mappings = {
    "attributes": Elastic.analyzed_field(),
    "pgTitle": Elastic.analyzed_field(),
    "secondTitle": Elastic.analyzed_field(),
    "caption": Elastic.analyzed_field(),
    "data": Elastic.analyzed_field(),
    "attributes++": Elastic.analyzed_field(),
    "attributes+": Elastic.analyzed_field(),
    "description+": Elastic.analyzed_field(),
    "values+": Elastic.analyzed_field(),
    "description+attributes": Elastic.analyzed_field(),
    "all": Elastic.analyzed_field(),
}

elastic = Elastic(index_name)
elastic.create_index(mappings, model='BM25',force=True)
elastic.add_docs_bulk(docs)
print("index has been built")