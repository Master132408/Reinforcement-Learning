# Reinforcement Learning for Hallucination Prevention

## Project Description
This project applies reinforcement learning (specifically a contextual bandit approach using LinUCB) to mitigate hallucinations in generative language models when answering medical questions. By dynamically selecting between models like Google's Gemini 2.0 Flash, Phi 3 Mini, and Qwen 4 based on the question and category, we aim to improve the factual consistency of outputs.

## Team Members
- **Agaaz Singhal**
- **Krishnav Mahansaria**
- **Mudit Surana**

## Technologies & Libraries Used
- **Python 3.x**
- **APIs**:
  - [Hugging Face Inference API](https://huggingface.co/inference-api)
  - [Google Gemini API](https://ai.google.dev/)
- **Libraries**:
  - `requests`
  - `json`, `sys`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `sentence-transformers`

## Project Structure
- `ask_medical_question.py`: Script that lets users input a medical question and selects the best model based on category.
- `contextual_bandit_llm_selector.py`: Implements the LinUCB algorithm for contextual bandit training and evaluation.
- `dataset.jsonl`: Input dataset used for training, formatted in JSON Lines.
- `requirements.txt`: List of all required Python packages.

## Dataset Format
Each entry in `dataset.jsonl` should include:
```json
{
  "input": "What is the first-line treatment for asthma?",
  "output": "Inhaled corticosteroids.",
  "subject_name": "Medicine",
  "gemini_ans": "...",
  "phi3mini_ans": "...",
  "qwen4b_ans": "..."
}
