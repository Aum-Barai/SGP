# __init__.py - Initialize the Flask app

from flask import Flask, render_template, request, jsonify
from .llm_services import analyze_sentiment, get_sentiment_explanation


def create_app():
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        sentiment = None
        tweet = ""
        if request.method == "POST":
            tweet = request.form.get("tweet", "").strip()
            if tweet:
                sentiment = analyze_sentiment(tweet)
        return render_template("index.html", tweet=tweet, sentiment=sentiment)

    @app.route("/explain", methods=["POST"])
    def explain():
        tweet = request.json.get("tweet", "").strip()
        sentiment = request.json.get("sentiment", "").strip()
        if tweet and sentiment:
            explanation = get_sentiment_explanation(tweet, sentiment)
            return jsonify({"explanation": explanation})
        return jsonify({"explanation": "Unable to generate explanation."}), 400

    return app
