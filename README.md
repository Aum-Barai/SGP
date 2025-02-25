<!-- @format -->

# Tweet Sentiment Analyzer

A Flask web application that performs zero-shot sentiment analysis on tweets using the Ollama LLM model. The application provides sentiment classification and explanations for the classifications.

## Features

- Tweet sentiment analysis using zero-shot classification
- Six sentiment categories: Positive, Negative, Neutral, Sarcastic, Funny, or Meme
- Detailed explanations for sentiment classifications
- Modern, responsive UI with Twitter-like character counter
- Real-time sentiment analysis using Ollama's LLM model

## Prerequisites

- Python 3.8 or higher
- Ollama installed with llama3.2:latest model
- pip (Python package installer)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/tweet-sentiment-analyzer.git
cd tweet-sentiment-analyzer
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running with the llama3.2:latest model:

```bash
ollama run llama3.2:latest
```

## Usage

1. Start the Flask application:

```bash
python run.py
```

2. Open your web browser and navigate to:

```
http://localhost:5000
```

3. Enter a tweet in the text area and click "Analyze Sentiment"
4. View the sentiment result and click "Explain Why" for a detailed explanation

## Project Structure

```
tweet-sentiment-analyzer/
├── app/
│   ├── __init__.py          # Flask application factory
│   ├── llm_services.py      # Sentiment analysis services
│   └── templates/
│       └── index.html       # Main application template
├── requirements.txt         # Project dependencies
├── run.py                  # Application entry point
└── README.md              # Project documentation
```

## API Endpoints

- `GET /` - Main page with the sentiment analysis form
- `POST /` - Endpoint for submitting tweets for analysis
- `POST /explain` - Get explanation for a sentiment classification

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
