import os
from flask import Flask, render_template, jsonify, abort, request

from search_backend.language_models.language_models import NGramModel
from search_backend.wiki_index import WikiIndex

app = Flask(__name__)
DIR = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = 'data/python-3.6.3-docs-text'
INIT = 0
DUMP_PATH = os.path.join(DIR, '../enwiki-20171020-pages-articles1.xml-p10p30302')

if INIT:
    text_index = WikiIndex(DUMP_PATH).build_index_file()
    print('Index ready')
else:
    text_index = WikiIndex(DUMP_PATH)
language_model_en = NGramModel(3, 'en').process_dir(DATA_PATH)


@app.route("/")
def root():
    return render_template('index.html')


@app.route('/search', methods=["POST"])
def search():
    json = request.get_json()
    if not json or json.get('query') is None:
        abort(400)

    query = str(json['query']).strip()
    print(query)
    suggested_query = language_model_en.spell_check(query)

    if query.startswith(("'", '"')) and query.endswith(("'", '"')):
        found_docs = text_index.search_query(query)
    elif ' AND ' in query or ' OR ' in query:
        found_docs = text_index.search_query(query)
    else:
        found_docs = text_index.search_query(query)

    return jsonify(
        results=found_docs,
        suggested_query=suggested_query
    )


if __name__ == "__main__":

    app.run()
