


@app.route('/content-based', methods=['GET'])
def content_based():
    book_id = int(request.args.get('book_id'))
    recommendations = get_content_recommendations(book_id)
    return jsonify(recommendations.to_dict(orient='records'))

@app.route('/collaborative', methods=['GET'])
def collaborative():
    user_id = int(request.args.get('user_id'))
    recommendations = get_collaborative_recommendations(user_id)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
