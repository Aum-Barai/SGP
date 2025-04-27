import logging
import time
import subprocess
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up caching
set_llm_cache(InMemoryCache())

# Valid sentiment categories
VALID_SENTIMENTS = frozenset(
    {
        "Positive",
        "Negative",
        "Neutral",
        "Sarcastic",
        "Funny",
        "Meme",
        "Excited",
        "Angry",
        "Sad",
        "Anxious",
        "Surprised",
        "Confused",
        "Hopeful",
        "Proud",
        "Grateful",
    }
)


@dataclass
class ModelMetrics:
    """Class to store model performance metrics."""

    response_time: float  # in seconds
    success: bool
    input_tokens: int = 0  # Number of input tokens
    output_tokens: int = 0  # Number of output tokens
    total_tokens: int = 0  # Total tokens used
    model_name: str = "llama3.2:latest"
    temperature: float = 0
    error_message: Optional[str] = None
    cache_hit: bool = False
    timestamp: str = str(datetime.now())


@dataclass
class AnalyzerStats:
    """Class to store analyzer statistics."""

    total_requests: int = 0
    successful_requests: int = 0
    cache_hits: int = 0
    avg_response_time: float = 0.0
    last_response_time: float = 0.0
    error_rate: float = 0.0
    total_tokens_used: int = 0
    avg_tokens_per_request: float = 0.0
    metrics_history: List[Dict] = None  # Store historical metrics for charts

    def __post_init__(self):
        if self.metrics_history is None:
            self.metrics_history = []


class SentimentAnalyzer:
    """Class to handle sentiment analysis operations."""

    def __init__(self, model_name: str = "llama3.2:latest", temperature: float = 0):
        """
        Initialize the sentiment analyzer.

        Args:
            model_name (str): Name of the Ollama model to use
            temperature (float): Temperature for text generation (0-1)
        """
        try:
            # First, verify the Ollama server is running by sending a simple test request
            self._check_ollama_connection()

            self.model_name = model_name
            self.temperature = temperature
            self.llm = OllamaLLM(
                model=model_name,
                temperature=temperature,
                stop=["\n"],  # Stop on newlines for cleaner outputs
                timeout=20,  # 20 second timeout
                retry_on_failure=True,  # Auto-retry on temporary failures
            )

            # Test the model with a simple input to make sure it's available
            self._test_model_availability()

        except ConnectionError as e:
            logger.error(f"Cannot connect to Ollama server: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {str(e)}")
            raise

    def _check_ollama_connection(self):
        """
        Check if Ollama server is running.

        Raises:
            ConnectionError: If the Ollama server is not running or unreachable
        """
        import requests

        try:
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(
                    f"Ollama server returned status code {response.status_code}"
                )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")

    def _test_model_availability(self):
        """
        Test if the selected model is available on the Ollama server.

        Raises:
            ValueError: If the model is not available
        """
        try:
            # Test with a very simple prompt
            self.llm.invoke("test")
        except Exception as e:
            error_message = str(e).lower()
            if "not found" in error_message or "no such file" in error_message:
                raise ValueError(
                    f"Model '{self.model_name}' not found on Ollama server. Please download it first."
                )
            raise

        # Initialize metrics
        self.stats = AnalyzerStats()
        self.recent_metrics: list[ModelMetrics] = []
        self.max_metrics_history = 100

        # Create prompt templates
        self.sentiment_template = PromptTemplate.from_template(
            """Task: Classify the sentiment of the following tweet into exactly one category.
Categories: Positive, Negative, Neutral, Sarcastic, Funny, Meme, Excited, Angry, Sad, Anxious, Surprised, Confused, Hopeful, Proud, or Grateful
Rules:
- Return ONLY the category name, nothing else
- Choose the most appropriate category based on the tweet's content, tone, and emotional expression
- Consider both explicit statements and implicit emotional cues
- Be consistent in classification

Tweet: "{tweet}"
Category:"""
        )

        self.explanation_template = PromptTemplate.from_template(
            """Tweet: "{tweet}"
Sentiment: {sentiment}

Task: Explain in 2-3 short sentences why this tweet was classified with the above sentiment.
Be specific about the words, tone, or context that led to this classification.
Focus on:
- Key words or phrases that indicate the sentiment
- Overall tone and context
- Any special patterns or characteristics

Response:"""
        )

        # Create chains with error handling
        self.sentiment_chain = (
            {"tweet": RunnablePassthrough()}
            | self.sentiment_template
            | self.llm
            | StrOutputParser()
        )

        self.explanation_chain = (
            {"tweet": lambda x: x["tweet"], "sentiment": lambda x: x["sentiment"]}
            | self.explanation_template
            | self.llm
            | StrOutputParser()
        )

    def _estimate_tokens(self, text: str) -> int:
        """Better estimation of token count.

        This function provides a more accurate token count estimation
        than simply counting words by using character-level heuristics.

        Args:
            text (str): Text to estimate tokens for

        Returns:
            int: Estimated token count
        """
        if not text:
            return 0

        # Try to use tiktoken if available (better accuracy)
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except (ImportError, Exception):
            # Fall back to character-based heuristic (closer to GPT tokenization than word count)
            # Average ratio of tokens to characters is ~0.25 for English text
            char_count = len(text)
            # Add extra tokens for code or special characters
            special_char_count = sum(
                1 for c in text if not c.isalnum() and not c.isspace()
            )
            return max(1, int((char_count * 0.25) + (special_char_count * 0.1)))

    def _update_metrics(self, metrics: ModelMetrics) -> None:
        """Update analyzer statistics with new metrics."""
        self.stats.total_requests += 1
        self.stats.successful_requests += int(metrics.success)
        self.stats.cache_hits += int(metrics.cache_hit)
        self.stats.total_tokens_used += metrics.total_tokens

        # Update response time metrics
        self.stats.last_response_time = metrics.response_time
        total_time = (
            self.stats.avg_response_time * (self.stats.total_requests - 1)
            + metrics.response_time
        )
        self.stats.avg_response_time = total_time / self.stats.total_requests

        # Update token metrics
        self.stats.avg_tokens_per_request = (
            self.stats.total_tokens_used / self.stats.total_requests
        )

        # Update error rate
        self.stats.error_rate = 1 - (
            self.stats.successful_requests / self.stats.total_requests
        )

        # Store metrics history for charts
        history_entry = {
            "timestamp": metrics.timestamp,
            "response_time": metrics.response_time,
            "total_tokens": metrics.total_tokens,
            "success": metrics.success,
            "cache_hit": metrics.cache_hit,
        }
        self.stats.metrics_history.append(history_entry)

        # Store recent metrics
        self.recent_metrics.append(metrics)
        if len(self.recent_metrics) > self.max_metrics_history:
            self.recent_metrics.pop(0)
        if len(self.stats.metrics_history) > self.max_metrics_history:
            self.stats.metrics_history.pop(0)

    def get_stats(self) -> Dict:
        """Get current analyzer statistics."""
        return {
            **asdict(self.stats),
            "recent_metrics": [
                asdict(m) for m in self.recent_metrics[-5:]
            ],  # Last 5 requests
            "model_info": {"name": self.model_name, "temperature": self.temperature},
        }

    def analyze_sentiment(self, tweet_text: str) -> Dict[str, Optional[str]]:
        """
        Analyze the sentiment of a tweet using zero-shot classification.

        Args:
            tweet_text (str): The tweet text to analyze

        Returns:
            Dict[str, Optional[str]]: Dictionary containing sentiment and any error message
        """
        start_time = time.time()
        metrics = None
        cache_hit = False

        if not tweet_text.strip():
            metrics = ModelMetrics(
                response_time=0,
                success=True,
                cache_hit=True,
                model_name=self.model_name,
                temperature=self.temperature,
            )
            self._update_metrics(metrics)
            return {"sentiment": "Neutral", "error": "Empty tweet provided"}

        try:
            # Estimate input tokens
            input_tokens = self._estimate_tokens(tweet_text)

            # Create a cache key for checking
            cache_key = f"{self.model_name}:{self.temperature}:{hash(self.sentiment_template.template)}:{hash(tweet_text)}"

            # For very fast responses, we'll consider them cache hits
            # This fallback method is used since direct cache access might not be possible in all environments

            # Get sentiment using the chain
            sentiment_start_time = time.time()
            sentiment = self.sentiment_chain.invoke(tweet_text)
            sentiment_end_time = time.time()

            # If response time is very quick, it's likely a cache hit
            sentiment_response_time = sentiment_end_time - sentiment_start_time
            if sentiment_response_time < 0.05:  # 50ms threshold for assuming cache hit
                cache_hit = True

            cleaned_sentiment = sentiment.strip()

            # Calculate metrics
            response_time = time.time() - start_time
            output_tokens = self._estimate_tokens(cleaned_sentiment)
            total_tokens = input_tokens + output_tokens

            metrics = ModelMetrics(
                response_time=response_time,
                success=cleaned_sentiment in VALID_SENTIMENTS,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model_name=self.model_name,
                temperature=self.temperature,
                cache_hit=cache_hit,
            )

            if cleaned_sentiment not in VALID_SENTIMENTS:
                metrics.success = False
                metrics.error_message = f"Invalid sentiment: {cleaned_sentiment}"
                self._update_metrics(metrics)
                return {
                    "sentiment": "Neutral",
                    "error": f"Received invalid sentiment: {cleaned_sentiment}",
                    "metrics": asdict(metrics),
                }

            self._update_metrics(metrics)
            return {
                "sentiment": cleaned_sentiment,
                "error": None,
                "metrics": asdict(metrics),
            }

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Error analyzing sentiment: {str(e)}")
            metrics = ModelMetrics(
                response_time=response_time,
                success=False,
                error_message=str(e),
                cache_hit=False,
                model_name=self.model_name,
                temperature=self.temperature,
            )
            self._update_metrics(metrics)
            return {"sentiment": "Error", "error": str(e), "metrics": asdict(metrics)}

    def get_explanation(
        self, tweet_text: str, sentiment: str
    ) -> Dict[str, Optional[str]]:
        """
        Get an explanation for why the given sentiment was chosen.

        Args:
            tweet_text (str): The original tweet text
            sentiment (str): The sentiment that was chosen

        Returns:
            Dict[str, Optional[str]]: Dictionary containing explanation and any error message
        """
        start_time = time.time()
        metrics = None

        if not tweet_text.strip() or not sentiment.strip():
            metrics = ModelMetrics(
                response_time=0,
                success=True,
                cache_hit=True,
                model_name=self.model_name,
                temperature=self.temperature,
            )
            self._update_metrics(metrics)
            return {"explanation": None, "error": "Empty tweet or sentiment provided"}

        if sentiment not in VALID_SENTIMENTS:
            metrics = ModelMetrics(
                response_time=0,
                success=False,
                error_message=f"Invalid sentiment: {sentiment}",
                cache_hit=True,
                model_name=self.model_name,
                temperature=self.temperature,
            )
            self._update_metrics(metrics)
            return {
                "explanation": None,
                "error": f"Invalid sentiment category: {sentiment}",
                "metrics": asdict(metrics),
            }

        try:
            # Estimate input tokens
            input_tokens = self._estimate_tokens(tweet_text) + self._estimate_tokens(
                sentiment
            )

            explanation = self.explanation_chain.invoke(
                {"tweet": tweet_text, "sentiment": sentiment}
            )

            # Calculate metrics
            response_time = time.time() - start_time
            output_tokens = self._estimate_tokens(explanation)
            total_tokens = input_tokens + output_tokens
            cache_hit = response_time < 0.1

            metrics = ModelMetrics(
                response_time=response_time,
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model_name=self.model_name,
                temperature=self.temperature,
                cache_hit=cache_hit,
            )
            self._update_metrics(metrics)

            return {
                "explanation": explanation.strip(),
                "error": None,
                "metrics": asdict(metrics),
            }

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Error getting explanation: {str(e)}")
            metrics = ModelMetrics(
                response_time=response_time,
                success=False,
                error_message=str(e),
                cache_hit=False,
                model_name=self.model_name,
                temperature=self.temperature,
            )
            self._update_metrics(metrics)
            return {"explanation": None, "error": str(e), "metrics": asdict(metrics)}


def get_available_models() -> List[Dict]:
    """Get a list of available Ollama models.

    This function tries multiple approaches to get the model list:
    1. Try using the Ollama API directly
    2. Try parsing the CLI output
    3. Fall back to a configurable default list

    Returns:
        List[Dict]: List of dictionaries containing model information
    """
    models = []

    # First, try using the Ollama API directly
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "models" in data:
                for model in data["models"]:
                    models.append(
                        {
                            "name": model.get("name", ""),
                            "size": f"{model.get('size', 0) // (1024*1024)} MB",
                            "display": f"{model.get('name', '')} ({model.get('size', 0) // (1024*1024)} MB)",
                        }
                    )
    except Exception as e:
        logger.warning(f"Failed to get models from Ollama API: {str(e)}")

    # If API didn't work, try CLI approach
    if not models:
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, check=True
            )

            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:  # Make sure we have at least the header and one model
                # Skip header line and process each model line
                for line in lines[1:]:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            # Format: NAME ID SIZE MODIFIED
                            name = parts[0]
                            size = (
                                parts[2] + " " + parts[3]
                                if len(parts) > 3
                                else parts[2]
                            )

                            # Don't add header row or empty entries
                            if name.lower() != "name" and name:
                                models.append(
                                    {
                                        "name": name,
                                        "size": size,
                                        "display": f"{name} ({size})",
                                    }
                                )
        except Exception as e:
            logger.warning(f"Failed to get models from Ollama CLI: {str(e)}")

    # If still no models, use default hardcoded list
    if not models:
        # Load from environment or config if available
        import os

        default_models_str = os.environ.get("OLLAMA_MODELS", "")

        if default_models_str:
            # Parse from environment variable in format "model1,model2,model3"
            for model_name in default_models_str.split(","):
                if model_name.strip():
                    models.append(
                        {
                            "name": model_name.strip(),
                            "size": "Unknown",
                            "display": f"{model_name.strip()} (Unknown)",
                        }
                    )
        else:
            # Hardcoded fallback list
            models = [
                {
                    "name": "llama3.2:latest",
                    "size": "2.0 GB",
                    "display": "llama3.2:latest (2.0 GB)",
                },
                {
                    "name": "qwen2.5-coder:1.5b",
                    "size": "986 MB",
                    "display": "qwen2.5-coder:1.5b (986 MB)",
                },
                {
                    "name": "llava:13b",
                    "size": "8.0 GB",
                    "display": "llava:13b (8.0 GB)",
                },
                {
                    "name": "deepseek-r1:8b",
                    "size": "4.9 GB",
                    "display": "deepseek-r1:8b (4.9 GB)",
                },
            ]

    return models


# Create a lock for thread safety when updating the analyzer
analyzer_lock = threading.Lock()

# Create a global instance of the analyzer with default model
analyzer = SentimentAnalyzer()


# Function to update the analyzer with a new model
def update_analyzer_model(model_name: str, temperature: float = 0) -> bool:
    """Update the analyzer with a new model.

    Thread-safe function to update the global analyzer instance with a new model.

    Args:
        model_name (str): Name of the Ollama model to use
        temperature (float): Temperature for text generation (0-1)

    Returns:
        bool: True if successful, False otherwise
    """
    global analyzer

    # Create a new analyzer instance first to avoid affecting the current one if there's an error
    try:
        new_analyzer = SentimentAnalyzer(model_name=model_name, temperature=temperature)

        # If successful, update the global instance with thread safety
        with analyzer_lock:
            analyzer = new_analyzer
        return True
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        return False


# Expose the functions with the same interface as before
def analyze_sentiment(tweet_text: str) -> str:
    """Backward-compatible function for sentiment analysis."""
    result = analyzer.analyze_sentiment(tweet_text)
    return result["sentiment"]


def get_sentiment_explanation(tweet_text: str, sentiment: str) -> str:
    """
    Get explanation for a sentiment classification.

    Args:
        tweet_text (str): The tweet to explain
        sentiment (str): The sentiment to explain

    Returns:
        str: The explanation

    Raises:
        ValueError: If the input is invalid or explanation cannot be generated
    """
    if not tweet_text or not tweet_text.strip():
        raise ValueError("Tweet text cannot be empty")

    if not sentiment or not sentiment.strip():
        raise ValueError("Sentiment cannot be empty")

    result = analyzer.get_explanation(tweet_text, sentiment)

    # Check for errors
    if result.get("error"):
        logger.error(f"Error getting explanation: {result['error']}")
        raise ValueError(result["error"])

    # Check for missing explanation
    if not result.get("explanation"):
        logger.error("No explanation was generated")
        raise ValueError("Unable to generate explanation")

    return result["explanation"]


def get_analyzer_stats() -> Dict:
    """Get current analyzer statistics."""
    return analyzer.get_stats()
