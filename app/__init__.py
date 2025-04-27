# __init__.py - Initialize the Flask app

import re
from flask import Flask, render_template, request, jsonify
from .llm_services import (
    analyze_sentiment,
    get_sentiment_explanation,
    get_analyzer_stats,
    get_available_models,
    update_analyzer_model,
)
import logging

logger = logging.getLogger(__name__)


def sanitize_input(text):
    """
    Sanitize user input to prevent prompt injection attacks.

    Args:
        text (str): User input text

    Returns:
        str: Sanitized text
    """
    if not text:
        return ""

    # Remove potentially dangerous patterns:
    # 1. Remove any code fence markers that could break out of prompts
    text = re.sub(r"```.*?```", "[code block]", text, flags=re.DOTALL)
    text = re.sub(r"`.*?`", "[code]", text, flags=re.DOTALL)

    # 2. Remove any special control sequences or prompt markers
    text = text.replace("\\n", " ")
    text = re.sub(r"<\|.*?\|>", "", text)  # Remove model control tokens

    # 3. Limit length to avoid excessively long inputs (e.g., max 500 chars)
    if len(text) > 500:
        text = text[:497] + "..."

    return text


def create_app():
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        sentiment = None
        tweet = ""
        stats = get_analyzer_stats()
        models = get_available_models()
        selected_model = stats["model_info"]["name"]
        temperature = stats["model_info"]["temperature"]

        if request.method == "POST":
            # Sanitize the input
            tweet = sanitize_input(request.form.get("tweet", "").strip())

            # Handle model and temperature from form
            model_name = request.form.get("model_name", "").strip()
            temp_str = request.form.get("temperature", "")

            # Only update if values are provided and different from current
            if model_name and model_name != selected_model:
                try:
                    models = get_available_models()
                    model_names = [model["name"] for model in models]
                    if model_name in model_names:
                        selected_model = model_name
                except Exception:
                    pass  # Keep the existing model if there's an error

            # Handle temperature if provided
            if temp_str:
                try:
                    temp = float(temp_str)
                    if 0 <= temp <= 1:
                        temperature = temp
                except ValueError:
                    pass  # Keep the existing temperature if invalid

            # Update the model if needed
            if (
                selected_model != stats["model_info"]["name"]
                or temperature != stats["model_info"]["temperature"]
            ):
                update_analyzer_model(selected_model, temperature)

            # Analyze the tweet if provided
            if tweet:
                sentiment = analyze_sentiment(tweet)
                stats = get_analyzer_stats()  # Get updated stats
            else:
                stats = get_analyzer_stats()  # Just get stats without analyzing

        return render_template(
            "index.html",
            tweet=tweet,
            sentiment=sentiment,
            stats=stats,
            models=models,
            selected_model=selected_model,
            temperature=temperature,
        )

    @app.route("/explain", methods=["POST"])
    def explain():
        # Extract the request data
        request_data = request.get_json()
        if not request_data:
            return (
                jsonify(
                    {
                        "explanation": None,
                        "message": "Invalid request format: No JSON data provided",
                        "stats": get_analyzer_stats(),
                    }
                ),
                400,
            )

        # Sanitize inputs
        tweet = sanitize_input(request_data.get("tweet", "").strip())
        sentiment = sanitize_input(request_data.get("sentiment", "").strip())

        # Validate inputs
        if not tweet:
            return (
                jsonify(
                    {
                        "explanation": None,
                        "message": "No tweet text provided",
                        "stats": get_analyzer_stats(),
                    }
                ),
                400,
            )

        if not sentiment:
            return (
                jsonify(
                    {
                        "explanation": None,
                        "message": "No sentiment value provided",
                        "stats": get_analyzer_stats(),
                    }
                ),
                400,
            )

        try:
            # Get the explanation
            explanation = get_sentiment_explanation(tweet, sentiment)
            stats = get_analyzer_stats()

            # Check for empty explanation
            if not explanation or explanation == "Unable to generate explanation.":
                return (
                    jsonify(
                        {
                            "explanation": None,
                            "message": "Could not generate explanation for this sentiment",
                            "stats": stats,
                        }
                    ),
                    400,
                )

            return jsonify({"explanation": explanation, "stats": stats})

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return (
                jsonify(
                    {
                        "explanation": None,
                        "message": f"Server error: {str(e)}",
                        "stats": get_analyzer_stats(),
                    }
                ),
                500,
            )

    @app.route("/stats", methods=["GET"])
    def stats():
        """Get current analyzer statistics."""
        return jsonify(get_analyzer_stats())

    @app.route("/models", methods=["GET"])
    def models():
        """Get available models."""
        models = get_available_models()
        return jsonify({"models": models})

    @app.route("/update_model", methods=["POST"])
    def update_model():
        """Update the model being used."""
        model_name = request.json.get("model_name", "").strip()

        # Validate that a model name was provided
        if not model_name:
            return jsonify({"success": False, "message": "Model name is required"}), 400

        # Validate temperature is between 0 and 1
        try:
            temp = float(request.json.get("temperature", 0))
            if not 0 <= temp <= 1:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": "Temperature must be between 0 and 1",
                        }
                    ),
                    400,
                )
            temperature = temp
        except ValueError:
            return (
                jsonify({"success": False, "message": "Temperature must be a number"}),
                400,
            )

        # Check if the model exists in the available models list
        models = get_available_models()
        model_names = [model["name"] for model in models]
        if model_name not in model_names:
            return (
                jsonify(
                    {"success": False, "message": f"Model '{model_name}' not found"}
                ),
                404,
            )

        # Update the model if it exists
        success = update_analyzer_model(model_name, temperature)
        if success:
            return jsonify(
                {"success": True, "message": f"Model updated to {model_name}"}
            )
        else:
            return jsonify({"success": False, "message": "Failed to update model"}), 500

    return app
