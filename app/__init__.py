# __init__.py - Initialize the Flask app

from flask import Flask, render_template, request, jsonify
from .llm_services import analyze_sentiment, get_sentiment_explanation, get_analyzer_stats


def create_app():
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        sentiment = None
        tweet = ""
        stats = get_analyzer_stats()
        if request.method == "POST":
            tweet = request.form.get("tweet", "").strip()
            if tweet:
                sentiment = analyze_sentiment(tweet)
                stats = get_analyzer_stats()  # Get updated stats
        return render_template("index.html", tweet=tweet, sentiment=sentiment, stats=stats)

    @app.route("/explain", methods=["POST"])
    def explain():
        tweet = request.json.get("tweet", "").strip()
        sentiment = request.json.get("sentiment", "").strip()
        if tweet and sentiment:
            explanation = get_sentiment_explanation(tweet, sentiment)
            stats = get_analyzer_stats()
            return jsonify({
                "explanation": explanation,
                "stats": stats
            })
        return jsonify({"explanation": "Unable to generate explanation.", "stats": get_analyzer_stats()}), 400

    @app.route("/stats", methods=["GET"])
    def stats():
        """Get current analyzer statistics."""
        return jsonify(get_analyzer_stats())

    return app
