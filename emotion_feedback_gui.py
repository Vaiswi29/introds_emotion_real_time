import tkinter as tk

class EmotionFeedbackGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Live Emotion Feedback")
        self.label = tk.Label(self.root, text="Emotion: None", font=("Arial", 16))
        self.label.pack()
        self.feedback_box = tk.Text(self.root, wrap=tk.WORD, width=50, height=10)
        self.feedback_box.pack()

    def update_feedback(self, emotion, feedback):
        self.label.config(text=f"Emotion: {emotion}")
        self.feedback_box.delete("1.0", tk.END)
        self.feedback_box.insert(tk.END, feedback)

    def run(self):
        self.root.mainloop()
