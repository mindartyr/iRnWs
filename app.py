import os
from flask import Flask, render_template, jsonify, abort, request
import search_backend
from search_backend.text_index import TextIndex

app = Flask(__name__)

DATA_PATH = 'data/python-3.6.3-docs-text'

text_index = TextIndex().build_index(DATA_PATH)


@app.route("/")
def root():
    return render_template('index.html')


@app.route('/search', methods=["POST"])
def search():
    json = request.get_json()
    if not json or json.get('query') is None:
        abort(400)

    query = str(json['query']).strip()

    return jsonify(
        results=search_backend.search(query)
    )


if __name__ == "__main__":
    app.run()
