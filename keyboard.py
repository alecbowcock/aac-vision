import customtkinter as ctk
from tkinter import Label, Frame
import torch
import string
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the tokenizer and model
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').eval()

# Define the number of top predictions to consider
top_k = 10

NUM_PREDICTION_BTNS = 6

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return tokens[:top_clean]


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=5):
    print(text_sentence)

    input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = bart_model(input_ids)[0]
    bart = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    return bart

# Function to get model predictions
def get_model_predictions(text):
    # Add a mask token at the end for prediction
    input_text = ' '.join(text.split())
    input_text += ' <mask>'
    predictions = get_all_predictions(input_text, NUM_PREDICTION_BTNS)
    return predictions

# Function to handle button click
def on_key_press(char):
    # Retrieve the current text and ensure it's in uppercase
    current_text = typed_text.cget("text").upper()

    if char == "space":
        # Append a space to the current text
        current_text += " "
    elif char == "back":
        # Remove the last character from the current text
        current_text = current_text[:-1]
    else:
        # Add the new character in uppercase to the current text
        current_text += char.upper()

    # Update the CTkLabel's text to display in uppercase
    typed_text.configure(text=current_text)

    if char == "space":
        predictions = get_model_predictions(current_text.lower())
        print("Predictions:", '\n'.join(predictions))  # Add this line to check predictions
        for i, prediction in enumerate(predictions):
            if i < len(prediction_buttons):
                prediction_buttons[i].configure(text = prediction.upper())
            else:
                break

# Function to close the application
def close_application():
    root.destroy()

# Initialize main application window with customtkinter
root = ctk.CTk()
root.title("Full Screen Keyboard")
root.configure(bg_color='#2c3e50')

# Set full screen
root.attributes("-fullscreen", True)

# Frame for the text area and predictions
text_frame = Frame(root, bg='#2c3e50')
text_frame.pack(side="top", fill="x")

# Increase the font size for the message bar
message_font_size = 72  # Adjust this size as needed
typed_text = ctk.CTkLabel(text_frame, text="", font=("Times", message_font_size), fg_color='white')
typed_text.pack(side="top", fill="x", padx=20, pady=50)  # Adjust padding as needed to increase siz

# Frame for prediction buttons
prediction_frame = Frame(text_frame, bg='#2c3e50')
prediction_frame.pack(side="top", pady=(0, 20))

# Define the size for the prediction buttons
prediction_button_width = 200  # Adjust the width as needed
prediction_button_height = 120  # Adjust the height as needed
prediction_button_font_size = 36  # Adjust the font size as needed
prediction_button_padx = 3    # Space between prediction buttons horizontally
prediction_button_pady = 0    # Space between prediction buttons vertically

# Create prediction buttons
prediction_buttons = []
for i in range(NUM_PREDICTION_BTNS):
    button = ctk.CTkButton(prediction_frame, text="", font=("Times", prediction_button_font_size), fg_color='#34495e', text_color='white', width=prediction_button_width, height=prediction_button_height, corner_radius=10)
    button.pack(side="left", padx=prediction_button_padx, pady=prediction_button_pady)
    prediction_buttons.append(button)

# Frame for keyboard buttons
keyboard_frame = Frame(root, bg='#2c3e50')
keyboard_frame.pack(side="bottom", fill="both", expand=True)

# Define keyboard layout
# Define the size and padding for the buttons
button_width = 120  # Increase the width
button_height = 120  # Increase the height
button_padx = 3    # Increase the horizontal padding
# button_pady = 1    # Increase the vertical padding

# Define keyboard layout
keys = [
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm', 'back'],
    ['space']
]

# Create buttons for keyboard
for i, row in enumerate(keys):
    frame_row = ctk.CTkFrame(keyboard_frame, fg_color='#2c3e50', corner_radius=10)
    frame_row.pack(side="top", fill="x", expand=True)
    for char in row:
        if char == "space":
            # Create the spacebar with a width that spans multiple keys
            button = ctk.CTkButton(frame_row, text="SPACE", command=lambda c=char: on_key_press(c), font=("Times", 36, "bold"), fg_color='#34495e', text_color='white', width=button_width * 5, height=button_height, corner_radius=10)
            button.pack(side="left", padx=button_padx, fill="both", expand=True)
            # button.pack(side="left", padx=button_padx, pady=button_pady, fill="both", expand=True)
        else:
            # Normal keys
            text = char if char != "back" else "⬅"  # You can use an arrow symbol or an image
            button = ctk.CTkButton(frame_row, text=text.upper(), command=lambda c=char: on_key_press(c), font=("Times", 36, "bold"), fg_color='#34495e', text_color='white', width=button_width, height=button_height, corner_radius=10)
            button.pack(side="left", padx=button_padx, fill="both", expand=True)
            # button.pack(side="left", padx=button_padx, pady=button_pady, fill="both", expand=True)

# Create an exit button in the top-right corner of the window
exit_button = ctk.CTkButton(root, text="X", command=close_application, font=("Times", 12), fg_color='red', text_color='white', corner_radius=5)
exit_button.place(relx=1.0, rely=0.0, x=-10, y=10, anchor="ne")  # Adjust x and y as needed

root.mainloop()
