<!-- @format -->

# Instructions for Tweet Sentiment Analyzer

## Setting up the Development Environment

1.  **Prerequisites**:

    - Ensure you have Python 3.8 or higher installed.
    - Install Ollama and make sure the `llama3.2:latest` model is downloaded and running.
    - Have `pip` installed for Python package management.

2.  **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/tweet-sentiment-analyzer.git
    cd tweet-sentiment-analyzer
    ```

    _(Replace `https://github.com/yourusername/tweet-sentiment-analyzer.git` with the actual repository URL)_

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Ollama**:
    Make sure Ollama is running in the background with the `llama3.2:latest` model loaded:
    ```bash
    ollama run llama3.2:latest
    ```

## Running the Application

1.  **Start the Flask Application**:

    ```bash
    python run.py
    ```

2.  **Access the Application**:
    Open your web browser and go to `http://localhost:5000`.

## Project Structure Overview

- `app/`: Contains the core application logic.
  - `__init__.py`: Flask app initialization and route definitions.
  - `llm_services.py`: Sentiment analysis logic and interaction with Ollama.
  - `templates/`: HTML templates for the user interface.
- `requirements.txt`: Python dependencies.
- `run.py`: Entry point to start the Flask app.
- `README.md`: Project documentation.
- `LICENSE`: Project license.

## Key Functionality

- **Sentiment Analysis**: Enter a tweet in the text area on the homepage and click "Analyze Sentiment" to get the sentiment classification.
- **Explanation**: After getting a sentiment, click "Explain Why" to get a short explanation for the classification.
- **Statistics**: The homepage displays real-time statistics about the analyzer's performance.

## Contributing

If you want to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a branch for your feature or bug fix: `git checkout -b feature/your-feature-name`.
3.  Make your changes and commit them: `git commit -m 'Add your feature'`.
4.  Push to your branch: `git push origin feature/your-feature-name`.
5.  Open a pull request to the main repository.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
