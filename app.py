import os
from flask import Flask, render_template, jsonify, abort, request
from flask import url_for

import search_backend
from search_backend.language_models import NGramModel
from search_backend.text_index import TextIndex

app = Flask(__name__)

DATA_PATH = 'data/python-3.6.3-docs-text'
INIT = 0

if INIT:
    text_index = TextIndex().build_index(DATA_PATH)
else:
    text_index = TextIndex.load_index()
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
    suggested_query = language_model_en.spell_check(query)

    if query.startswith(("'", '"')) and query.endswith(("'", '"')):
        found_docs = text_index.process_query(query)
    elif ' AND ' or ' OR ' in query:
        found_docs = text_index.process_conditional_query(query)
    else:
        found_docs = text_index.process_query(query)

    return jsonify(
        results=found_docs,
        suggested_query=suggested_query
    )


if __name__ == "__main__":

    app.run()
