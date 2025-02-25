@echo off
REM ========================================================
REM Create optimal file structure for the Flask MVP project
REM ========================================================

REM Create main subdirectories
mkdir app
mkdir tests

REM Create subdirectories inside the "app" folder
cd app
mkdir templates
mkdir static
mkdir static\css
mkdir static\js
mkdir models

REM Create initial Python module files with basic placeholder text
echo # __init__.py - Initialize the Flask app > __init__.py
echo # routes.py - Define your URL endpoints > routes.py
echo # sentiment_model.py - Placeholder for ML model code > models\sentiment_model.py

REM Return to the project root
cd ..

REM Create project-level files
echo # Configuration settings for Flask > config.py
echo from app import app > run.py
echo Flask==2.0.1 > requirements.txt
echo # Project README > README.md

REM Create a sample test file in the tests folder
cd tests
echo # test_routes.py - Write your tests here > test_routes.py

REM Return to the project root
cd ..

echo.
echo Project structure created successfully!
pause
