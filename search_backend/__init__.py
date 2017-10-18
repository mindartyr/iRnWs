def search(query_string):
    result_list = []

    for i in range(10):
        result_list.append({
            'title': 'Dummy title for result #{} of query “{}”'.format(i + 1, query_string),
            'snippet': 'Dummy snippet',
            'href': 'http://www.example.com'
        })

    return result_list
