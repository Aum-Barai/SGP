import logging
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

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
            self.model_name = model_name
            self.temperature = temperature
            self.llm = OllamaLLM(
                model=model_name,
                temperature=temperature,
                stop=["\n"],  # Stop on newlines for cleaner outputs
                timeout=20,  # 20 second timeout
                retry_on_failure=True,  # Auto-retry on temporary failures
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {str(e)}")
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
        """Rough estimation of token count."""
        return len(text.split())

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

            # Get sentiment using the chain
            sentiment = self.sentiment_chain.invoke(tweet_text)
            cleaned_sentiment = sentiment.strip()

            # Calculate metrics
            response_time = time.time() - start_time
            output_tokens = self._estimate_tokens(cleaned_sentiment)
            total_tokens = input_tokens + output_tokens
            cache_hit = response_time < 0.1  # Assuming sub-100ms is a cache hit

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


# Create a global instance of the analyzer
analyzer = SentimentAnalyzer()


# Expose the functions with the same interface as before
def analyze_sentiment(tweet_text: str) -> str:
    """Backward-compatible function for sentiment analysis."""
    result = analyzer.analyze_sentiment(tweet_text)
    return result["sentiment"]


def get_sentiment_explanation(tweet_text: str, sentiment: str) -> str:
    """Backward-compatible function for getting explanations."""
    result = analyzer.get_explanation(tweet_text, sentiment)
    return result["explanation"] or "Unable to generate explanation."


def get_analyzer_stats() -> Dict:
    """Get current analyzer statistics."""
    return analyzer.get_stats()
