import cv2
import mss
import numpy as np
import os
import requests
import textwrap
from deepface import DeepFace
from dotenv import load_dotenv
from emotion_feedback_gui import EmotionFeedbackGUI

# Load GPT API key
load_dotenv(".env.local")
api_key = os.getenv("OPENROUTER_API_KEY")

# Setup GUI
gui = EmotionFeedbackGUI()

def get_feedback_from_gpt(emotion):
    system_prompt = (
        "You are a professional public speaking coach. "
        "Give a short tip (1-2 sentences) based on the audience's facial emotion. "
        "Be supportive and give suggestions if the audience seems unengaged. "
        f"The topic being presented is Hawaiian beach. If the audience looks {emotion}, tailor advice to that."
    )
    user_prompt = f"The audience looks {emotion.lower()}. What should I say or do next in my presentation?"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://your-site.com",
        "X-Title": "Live Emotion Feedback"
    }

    json_data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=json_data)
        res.raise_for_status()
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        print("GPT error:", e)
        return "Unable to fetch suggestion."

def main():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Full screen

        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                print(f"\nDetected Emotion: {emotion}")
                suggestion = get_feedback_from_gpt(emotion)
                print(f"GPT Suggestion:\n{textwrap.fill(suggestion, width=80)}\n")
                
                # FIXED: Run GUI update safely in main thread
                gui.root.after(0, gui.update_feedback, emotion, suggestion)

            except Exception as e:
                print(f"DeepFace failed:", e)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    gui.root.after(100, main)  # Schedule emotion loop
    gui.run()
