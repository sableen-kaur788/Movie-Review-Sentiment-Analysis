import tkinter as tk
from tkinter import messagebox, Text, Label, Button, PhotoImage, END, WORD, GROOVE, RAISED
import joblib

# Load model & vectorizer
log_reg_model = joblib.load('logistic_regression_model (1).pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer (1).pkl')

# Predict function
def predict_sentiment():
    review = text_area.get("1.0", END).strip()
    if not review:
        messagebox.showwarning("Warning", "Please enter a movie review!")
        return
    
    review_tfidf = tfidf_vectorizer.transform([review])
    prediction = log_reg_model.predict(review_tfidf)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    if sentiment == "Positive":
        result_label.config(text=f"Predicted Sentiment: {sentiment} 😊", fg="#2E8B57")  # Dark Green
    else:
        result_label.config(text=f"Predicted Sentiment: {sentiment} 😞", fg="#B22222")  # Dark Red

# Tkinter window
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

root.geometry(f"{screen_width}x{screen_height}+0+0")
root.resizable(False, False)
root.title("Movie Sentiment Analyzer")
root.configure(bg="#FDE2F3")  

# App icon
logo = PhotoImage(file="1.png")  
root.iconphoto(False, logo)

# Title 
title_label = Label(root, text="🎬 MOVIE SENTIMENT ANALYZER", 
                    bg="#E5BEEC", fg="#1C1C1C",   
                    font="arial 28 bold", padx=10, pady=20, border=4, relief=GROOVE)
title_label.pack(pady=20)

# Text area 
text_area = Text(root, font="arial 22", bg="#FFF8F3", fg="#1C1C1C",   
                 relief=GROOVE, wrap=WORD, padx=15, pady=15)
text_area.place(x=screen_width * 0.05, y=screen_height * 0.18, 
                width=screen_width * 0.9, height=screen_height * 0.45)

# Predict Button 
predict_button = Button(root, text=" PREDICT SENTIMENT", bg="#917FB3", fg="white",
                        width=20, borderwidth=4, font="arial 24 bold", relief=RAISED,
                        activebackground="#76549A", activeforeground="white",
                        command=predict_sentiment)
predict_button.place(x=screen_width * 0.35, y=screen_height * 0.67)

# Result Label - centered under button
result_label = Label(root, text="", font="arial 24 bold", bg="#FDE2F3", fg="#1C1C1C")
result_label.place(relx=0.5, rely=0.82, anchor="center") 

root.mainloop()
