# get_gpt_tip.py

import os
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")
api_key = os.getenv("OPENROUTER_API_KEY")

def get_feedback_from_gpt(emotion):
    system_prompt = "You are a professional public speaking coach giving advice based on audience facial emotion. The topic being presented is Hawaiian beach. Give me suggestions based on the audience reachtion specific to the topic. For example, if the expression is Fear, mention an interesting fact about Hawaiian beach that can help the audience feel more comfortable. Give like two or three responses."
    # system_prompt = "You are a professional public speaking coach giving advice based on audience facial emotion. Give like two or three responses."
    user_prompt = f"The audience looks {emotion.lower()}. What should I say or do in my presentation?"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://your-site.com",  # optional
        "X-Title": "Emotion-Based Feedback"
    }

    json_data = {
        "model": "openai/gpt-3.5-turbo",  # or another OpenRouter-supported model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=json_data)
    data = response.json()
    return data['choices'][0]['message']['content']

# Test
if __name__ == "__main__":
    emotion = input("Enter detected emotion: ")
    tip = get_feedback_from_gpt(emotion)
    print("\nGPT Tip:\n", tip)
