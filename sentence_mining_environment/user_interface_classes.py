import tkinter as tk
from tkinter import filedialog
import os
import json

JSON_FILENAME = "word_sentence_data.json"

def load_wsj():
    """
    Load the Word-Sentence JSON (WSJ) from disk if it exists.
    Otherwise return an empty dictionary.
    """
    if os.path.exists(JSON_FILENAME):
        with open(JSON_FILENAME, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_wsj(wsj_data):
    """
    Save the WSJ dictionary to disk as a JSON file.
    """
    with open(JSON_FILENAME, "w", encoding="utf-8") as f:
        json.dump(wsj_data, f, indent=2, ensure_ascii=False)

def split_into_sentences(text):
    """
    A simple utility function to split text into sentences.
    This is a naive approach, splitting on '.' and line breaks.
    Feel free to improve this to handle punctuation more robustly.
    """
    # Replace newlines with spaces (for simpler splitting).
    text = text.replace('\n', ' ')
    # Split on periods. You can adjust or add other punctuation if desired.
    raw_sentences = text.split('.')

    # Clean up extra spaces, then add '.' back to each sentence if it’s not empty
    sentences = []
    for raw in raw_sentences:
        sentence = raw.strip()
        if sentence:
            sentence_with_period = sentence + '.'
            sentences.append(sentence_with_period)
    return sentences

class WordSentenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Word-Sentence JSON App")

        # WSJ data loaded from disk
        self.wsj_data = load_wsj()

        # We'll store the user-acquired text (from file or pasted) here
        self.user_text = ""

        # Mapping from a unique tag to (word, sentence)
        # This helps us know which word/sentence was clicked in the Text widget.
        self.tag_word_map = {}

        # Create the start screen
        self.start_frame = tk.Frame(self.root)
        self.start_frame.pack(padx=20, pady=20)

        upload_button = tk.Button(
            self.start_frame, 
            text="Upload a File", 
            command=self.upload_file
        )
        upload_button.pack(side="left", padx=10)

        paste_button = tk.Button(
            self.start_frame, 
            text="Paste Text", 
            command=self.open_paste_window
        )
        paste_button.pack(side="left", padx=10)

    def upload_file(self):
        """
        Opens a file dialog so the user can select a text file.
        """
        filepath = filedialog.askopenfilename(
            title="Select a text file",
            filetypes=[("Text Files", "*.txt")]
        )
        if filepath:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    self.user_text = f.read()
                # Once text is acquired, move to display screen
                self.show_text_display()
            except Exception as e:
                print(f"Error reading file: {e}")

    def open_paste_window(self):
        """
        Opens a new window that allows the user to paste text.
        """
        paste_window = tk.Toplevel(self.root)
        paste_window.title("Paste your text")

        # A Text widget to paste or type text (with word wrap)
        text_widget = tk.Text(paste_window, width=50, height=15, wrap=tk.WORD)
        text_widget.pack(padx=10, pady=10)

        # Done button
        done_button = tk.Button(
            paste_window, 
            text="Done", 
            command=lambda: self.handle_pasted_text(text_widget, paste_window)
        )
        done_button.pack(pady=5)

    def handle_pasted_text(self, text_widget, window):
        """
        Reads the text from the Text widget, stores it, and closes the paste window.
        Then transitions to the display screen.
        """
        self.user_text = text_widget.get("1.0", tk.END).strip()
        window.destroy()
        self.show_text_display()

    def show_text_display(self):
        """
        Once we have user_text from either upload or paste, create a new screen
        that shows all text in a Text widget with word wrapping.
        Each word is tagged to enable click handling.
        """
        # Destroy the start frame if it exists
        if self.start_frame is not None:
            self.start_frame.destroy()
            self.start_frame = None

        # Create a new frame for displaying text
        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Create a Text widget with word wrap
        self.text_display = tk.Text(
            self.display_frame,
            wrap='word',
            font=("Helvetica", 16)  # Adjust font size as desired
        )
        self.text_display.pack(fill='both', expand=True)

        sentences = split_into_sentences(self.user_text)

        tag_index = 0  # We'll assign unique tags as we go
        for sentence in sentences:
            # Split sentence into words (naively on whitespace)
            words = sentence.split()
            for word in words:
                # Create a unique tag for this word
                tag = f"tag_{tag_index}"
                tag_index += 1

                # Store the (word, sentence) in a dictionary for later lookup
                self.tag_word_map[tag] = (word, sentence)

                # Insert the word and a trailing space
                # We associate it with the same tag
                self.text_display.insert('end', word + " ", tag)

                # Configure the tag to look clickable (blue, underlined, etc. if you want)
                self.text_display.tag_config(tag, foreground='blue', underline=False)

                # Bind click on this tag to the handler
                self.text_display.tag_bind(tag, "<Button-1>", self.on_word_click)

            # After each sentence, insert a blank line
            self.text_display.insert('end', "\n\n")

    def on_word_click(self, event):
        """
        Called when a user clicks on a tagged word in the Text widget.
        We find which tag was clicked, retrieve (word, sentence),
        update the WSJ, change the word color to red, and save.
        """
        # Get mouse coordinates within the text widget
        x, y = event.x, event.y
        # Find the index (position) in the text where the user clicked
        index = event.widget.index(f"@{x},{y}")
        # Retrieve all tags at that position (there might be multiple)
        tags_at_index = event.widget.tag_names(index)

        for tag in tags_at_index:
            if tag in self.tag_word_map:
                # We found the tag that holds (word, sentence)
                word, sentence = self.tag_word_map[tag]

                word = word.lower()

                # Update WSJ
                if word not in self.wsj_data:
                    self.wsj_data[word] = []
                if sentence not in self.wsj_data[word]:
                    self.wsj_data[word].append(sentence)

                # Save the WSJ to disk
                save_wsj(self.wsj_data)

                # Give user feedback: change the word color to red
                event.widget.tag_config(tag, foreground='red')

                # For demonstration in the console
                print(f"Clicked '{word}'. Updated WSJ data for this word.")

def main():
    root = tk.Tk()
    app = WordSentenceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
