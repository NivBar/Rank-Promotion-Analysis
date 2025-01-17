# Project: Rank Promotion analysis

## Overview
This repository contains scripts for a chat-based competition system utilizing OpenAI models for text generation and ranking improvements. The project includes utilities for bot follow-ups, ranking analysis, and dataset processing.

## File Breakdown

### 1. `competition_chatgpt_google.py`
- **Purpose**: Handles chat-based competition logic, message processing, and ranking calculations.
- **Inputs**: Query and ranking data.
- **Outputs**: Ranked responses from models.

### 2. `config.py`
- **Purpose**: Contains configuration settings for the model, including temperature, max tokens, and active bots.
- **Inputs**: Predefined settings and external API key.
- **Outputs**: Configurable parameters used across scripts.

### 3. `create_bot_followup_file.py`
- **Purpose**: Generates bot follow-up data based on queries and ranking results.
- **Inputs**: `greg_data.csv` or `tommy_data.csv`.
- **Outputs**: Processed follow-up files in CSV format.

### 4. `create_greg_data.py`
- **Purpose**: Processes TREC-formatted documents and converts them into structured datasets for ranking analysis.
- **Inputs**: Raw ranking data in text format.
- **Outputs**: `greg_data.csv` with structured ranking data.

### 5. `ranking_stats.py`
- **Purpose**: Computes ranking statistics, including cosine similarity and ranking changes over iterations.
- **Inputs**: Processed ranking data.
- **Outputs**: Statistical metrics for ranking evaluation.

### 6. `text_validation.py`
- **Purpose**: Ensures valid sentence structure and processes text data before further ranking.
- **Inputs**: Bot-generated texts.
- **Outputs**: Validated texts.

### 7. `utils.py`
- **Purpose**: Provides utility functions for data handling, initial document creation, and prompt fine-tuning.
- **Inputs**: Query descriptions and ranking data.
- **Outputs**: Structured query-response data.

## Installation & Usage

### Prerequisites
- Python 3.x
- Install required dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Scripts
Each script serves a specific purpose. Example:
```bash
python competition_chatgpt_google.py
```
Modify configuration settings in `config.py` as needed.

## Dependencies
- `openai`
- `pandas`
- `nltk`
- `tqdm`
- `torch`
- `transformers`

Ensure all required datasets (`greg_data.csv`, `tommy_data.csv`, etc.) are available before execution.

## Notes
- The project involves ranking text responses based on search engine optimization techniques.
- Ensure `API_key.py` (if required) is correctly configured for OpenAI API usage.