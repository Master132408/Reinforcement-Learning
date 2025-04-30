
import sys
import json
import requests

category_model_mapping = {
    "Anaesthesia": "gemini_ans",
    "Anatomy": "gemini_ans",
    "Biochemistry": "gemini_ans",
    "Dental": "gemini_ans",
    "ENT": "gemini_ans",
    "Forensic Medicine": "gemini_ans",
    "Gynaecology & Obstetrics": "gemini_ans",
    "Medicine": "phi3mini_ans",
    "Microbiology": "gemini_ans",
    "Ophthalmology": "gemini_ans",
    "Orthopaedics": "gemini_ans",
    "Pathology": "gemini_ans",
    "Pediatrics": "gemini_ans",
    "Pharmacology": "gemini_ans",
    "Physiology": "gemini_ans",
    "Psychiatry": "gemini_ans",
    "Radiology": "gemini_ans",
    "Skin": "phi3mini_ans",
    "Social & Preventive Medicine": "gemini_ans",
    "Surgery": "gemini_ans",
    "Unknown": "phi3mini_ans"
}
# Map *local* names ➜ real Hugging Face repo IDs
HF_MODEL_MAP = {
    "phi3mini_ans": "microsoft/Phi-3-mini-4k-instruct",
    "qwen4b_ans"  : "Qwen/Qwen1.5-4B-Chat",
}
HF_API_TOKEN="--------------"
def ask_hf(local_name: str, prompt: str, max_tokens: int = 256) -> str:
    repo_id = HF_MODEL_MAP.get(local_name)
    if repo_id is None:
        raise ValueError(f"No HF mapping for model '{local_name}'")

    url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens}
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()                       # will now be 200

    return resp.json()[0]["generated_text"]


# Function to call the Gemini model
def ask_gemini(prompt: str) -> str:
    GEMINI_API_KEY = "----------------"  
    GEMINI_FLASH_URL = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.0-flash:generateContent"
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }

    r = requests.post(
        GEMINI_FLASH_URL,
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

# Function to ask the user for the medical question and category
def ask_medical_question():
    print("Welcome to the Medical Question Answering System!")

    # Step 1: Ask the user for the medical question.
    question = input("Please enter your medical question: ")

    # Step 2: Ask the user for the category of the question.
    category = input("Please enter the category of your medical question (e.g., 'Anaesthesia', 'Medicine', etc.): ")

    # Step 3: Match the category to the best model
    best_model = category_model_mapping.get(category, "phi3mini_ans")  # Default model for "Unknown" or unmatched categories

    # Step 4: Call the corresponding LLM API based on the category.
    if best_model == "gemini_ans":
        answer = ask_gemini(question)
    elif best_model == "phi3mini_ans":
        answer = ask_hf("phi3mini_ans", question)
    elif best_model == "qwen4b_ans":
        answer = ask_hf("qwen4b_ans", question)
    else:
        answer = "Model not found."

    # Display the result
    print(f"\nModel used: {best_model}")
    print(f"Answer: {answer}")

# Main execution point
if __name__ == "__main__":
    ask_medical_question()
