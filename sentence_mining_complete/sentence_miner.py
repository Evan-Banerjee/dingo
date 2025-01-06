import tkinter as tk
from tkinter import filedialog, ttk
import os
import json

# ------------------
# For demonstration, we show how you *could* integrate MarianMT from Hugging Face
# to translate sentences into English without any paid API.
# To actually use it, you need to:
#   pip install transformers sentencepiece sacremoses
# and download models that handle your source language -> English
#
# Example model names (you can search on Hugging Face):
#   English -> English is trivial, no translation needed.
#   Spanish -> English: "Helsinki-NLP/opus-mt-es-en"
#   Turkish -> English: "Helsinki-NLP/opus-mt-tr-en"
#   Greek   -> English: "Helsinki-NLP/opus-mt-el-en"
#   Chinese -> English: "Helsinki-NLP/opus-mt-zh-en" 
#     (though you may want to check for a simplified vs. traditional model)
#
# For a single universal translator, you might try "Helsinki-NLP/opus-mt-mul-en"
# but not all languages are equally well-supported. 
#
# This example uses naive checks and one model name per language to illustrate usage.

try:
    from transformers import MarianMTModel, MarianTokenizer
except ImportError:
    MarianMTModel = None
    MarianTokenizer = None
    print("Warning: Hugging Face Transformers not installed. Translation won't work.")


# For Chinese segmentation using THULAC:
#   pip install thulac
# Then you can import it as:
try:
    import thulac
except ImportError:
    thulac = None
    print("Warning: THULAC is not installed. Chinese segmentation won't work.")


# Map each language to a separate JSON file
LANGUAGE_TO_JSON = {
    "English":           "word_sentence_data_en.json",
    "Spanish":           "word_sentence_data_es.json",
    "Turkish":           "word_sentence_data_tr.json",
    "Simplified Chinese": "word_sentence_data_zh_hans.json",
    "Traditional Chinese": "word_sentence_data_zh_hant.json",
    "Greek":             "word_sentence_data_el.json",
}

# Map each language to an appropriate MarianMT model (or None if no translation needed)
LANGUAGE_TO_MODEL = {
    "English":           None,  # No translation needed from English -> English
    "Spanish":           "Helsinki-NLP/opus-mt-es-en",
    "Turkish":           "Helsinki-NLP/opus-mt-tr-en",
    "Simplified Chinese":"Helsinki-NLP/opus-mt-zh-en",
    "Traditional Chinese":"Helsinki-NLP/opus-mt-zh-en",
    "Greek":             "Helsinki-NLP/opus-mt-el-en",
}

# Some purely grammatical words that might yield "no direct translation" in Chinese
# (This is a very naive example. In practice, you'd have a more nuanced approach.)
CHINESE_FUNCTION_WORDS = {"的", "了", "吗", "呢", "吧", "啊", "嘛", "着", "过"}


def load_wsj(language):
    """
    Load the Word-Sentence JSON (WSJ) from disk if it exists for a given language.
    Otherwise return an empty dictionary.
    """
    json_filename = LANGUAGE_TO_JSON[language]
    if os.path.exists(json_filename):
        with open(json_filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_wsj(language, wsj_data):
    """
    Save the WSJ dictionary to disk as a JSON file, depending on the chosen language.
    """
    json_filename = LANGUAGE_TO_JSON[language]
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(wsj_data, f, indent=2, ensure_ascii=False)


def load_translation_model(language):
    """
    Loads a MarianMT model + tokenizer for the given language -> English (if needed).
    Returns (model, tokenizer) or (None, None) if no translation is needed / possible.
    """
    if not MarianMTModel or not MarianTokenizer:
        return (None, None)
    model_name = LANGUAGE_TO_MODEL.get(language, None)
    if model_name is None:
        return (None, None)
    
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return (model, tokenizer)
    except Exception as e:
        print(f"Could not load model {model_name} for {language}: {e}")
        return (None, None)


def translate_sentence_to_en(sentence, model, tokenizer):
    """
    Translate a sentence (in source language) to English using MarianMT.
    Returns the translated string, or the original if no translation was possible.
    """
    if not model or not tokenizer:
        # No model available or loaded, just return original
        return sentence

    # MarianMT standard usage
    batch = tokenizer([sentence], return_tensors="pt", truncation=True)
    gen = model.generate(**batch)
    translation = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return translation.strip()


def segment_with_thulac(sentence):
    """
    Uses THULAC to segment a Chinese sentence into a list of words.
    If THULAC is not available, it raises an exception or you can handle a fallback.
    """
    if not thulac:
        raise ImportError("THULAC is not installed or not imported.")
    # Ideally, you'd create a single THULAC() instance at the start for performance,
    # but for demonstration, we do it here:
    thu = thulac.thulac(seg_only=True)
    segmented_pairs = thu.cut(sentence)
    # segmented_pairs is a list of (word, tag) pairs
    segmented_words = [wp[0] for wp in segmented_pairs]
    return segmented_words


def naive_segment(sentence, language):
    """
    Given a sentence and language, returns a list of words.
    For Chinese, calls THULAC (if available). For others, uses whitespace splitting.
    """
    # Handle Chinese specially
    if language in ["Simplified Chinese", "Traditional Chinese"]:
        # Attempt to segment with THULAC
        try:
            return segment_with_thulac(sentence)
        except:
            # fallback: naive split by whitespace
            return sentence.split()
    else:
        # Simple whitespace-based tokenization for demonstration
        return sentence.split()


def is_grammatical_particle(word, language):
    """
    Very naive check to see if 'word' might be a purely grammatical particle
    that has no direct English translation. 
    Only specialized for Chinese here, as an example.
    """
    if language in ["Simplified Chinese", "Traditional Chinese"]:
        # Check if it's in our small function-word set
        if word in CHINESE_FUNCTION_WORDS:
            return True
    # For other languages, you might expand this logic
    return False


class WordSentenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Language Word-Sentence JSON App")

        # Step 1: Language selection
        self.language_frame = tk.Frame(self.root)
        self.language_frame.pack(padx=20, pady=20)

        self.language_label = tk.Label(self.language_frame, text="Select the language:")
        self.language_label.pack(side="left", padx=5)

        # We'll use a ttk.Combobox with the 6 languages
        self.language_var = tk.StringVar()
        self.language_combobox = ttk.Combobox(
            self.language_frame,
            textvariable=self.language_var,
            values=list(LANGUAGE_TO_JSON.keys()),  # 6 languages
            width=25
        )
        # Enable typing to filter options
        self.language_combobox["state"] = "normal"
        self.language_combobox.set("English")  # default
        self.language_combobox.pack(side="left", padx=5)

        self.language_button = tk.Button(
            self.language_frame,
            text="Confirm",
            command=self.confirm_language
        )
        self.language_button.pack(side="left", padx=5)

        # Frame references
        self.start_frame = None
        self.display_frame = None

        # Data
        self.selected_language = None
        self.user_text = ""
        self.wsj_data = {}  # Word-Sentence JSON for the chosen language

        # Text widget references
        self.text_display = None
        self.tag_word_map = {}

        # Translation models (per language)
        self.loaded_models = {}

    def confirm_language(self):
        """
        Called when user confirms the language from the combobox.
        Loads the JSON data for that language and transitions to the next screen.
        """
        lang = self.language_var.get()
        if lang not in LANGUAGE_TO_JSON:
            tk.messagebox.showerror("Error", "Please select a valid language.")
            return

        self.selected_language = lang

        # Load the WSJ from disk
        self.wsj_data = load_wsj(self.selected_language)

        # Possibly pre-load the translation model for this language
        if self.selected_language not in self.loaded_models:
            model, tokenizer = load_translation_model(self.selected_language)
            self.loaded_models[self.selected_language] = (model, tokenizer)

        # Destroy the language_frame
        self.language_frame.destroy()

        # Now show the start screen for uploading/pasting text
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

        text_widget = tk.Text(paste_window, width=50, height=15, wrap=tk.WORD)
        text_widget.pack(padx=10, pady=10)

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
        Display the user-acquired text in a Tk Text widget.
        Each word is tagged to enable click/unclick handling.
        """
        if self.start_frame is not None:
            self.start_frame.destroy()
            self.start_frame = None

        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.text_display = tk.Text(
            self.display_frame,
            wrap='word',
            font=("Helvetica", 16)
        )
        self.text_display.pack(fill='both', expand=True)

        # We'll split the entire text into sentences, naive approach (split on '.')
        # You might want to refine this approach for advanced punctuation handling.
        sentences = self.split_into_sentences(self.user_text)

        self.tag_word_map = {}  # reset
        tag_index = 0

        for sentence in sentences:
            # Segment the sentence into words based on the chosen language
            words = naive_segment(sentence, self.selected_language)
            
            for word in words:
                tag = f"tag_{tag_index}"
                tag_index += 1

                # We'll store a dictionary with all relevant info:
                #   { "word": <word>, "sentence": <original sentence> }
                # We'll do the translation logic when the user clicks.
                self.tag_word_map[tag] = {
                    "word": word,
                    "sentence": sentence
                }

                self.text_display.insert('end', word + " ", tag)

                # If this word is already in the JSON with this sentence, color it red
                if self.word_in_json(word, sentence):
                    self.text_display.tag_config(tag, foreground='red')
                else:
                    self.text_display.tag_config(tag, foreground='blue')

                # Bind click
                self.text_display.tag_bind(tag, "<Button-1>", self.on_word_click)

            # Insert a couple of newlines after each sentence
            self.text_display.insert('end', "\n\n")

    def word_in_json(self, word, sentence):
        """
        Check if this word is already in self.wsj_data, and if the exact sentence
        is in the corresponding list (the first element of the tuple).
        """
        if word not in self.wsj_data:
            return False
        # Each entry in wsj_data[word] is a tuple of (orig_sent, translated_sent, word_translation)
        for tpl in self.wsj_data[word]:
            orig_sentence = tpl[0]
            if orig_sentence == sentence:
                return True
        return False

    def on_word_click(self, event):
        """
        Called when a user clicks on a tagged word in the Text widget.
        If the word-sentence pair is not in the JSON, add it (and color red).
        If it is already there, remove it (and color back to blue).
        """
        x, y = event.x, event.y
        index = event.widget.index(f"@{x},{y}")
        tags_at_index = event.widget.tag_names(index)

        for tag in tags_at_index:
            if tag in self.tag_word_map:
                info = self.tag_word_map[tag]
                word = info["word"]
                sentence = info["sentence"]

                if not self.word_in_json(word, sentence):
                    # It's not in JSON, so add it
                    self.add_sentence_to_json(word, sentence)
                    # Update color to red
                    event.widget.tag_config(tag, foreground='red')
                else:
                    # It's already in JSON, so remove it
                    self.remove_sentence_from_json(word, sentence)
                    # Update color to blue
                    event.widget.tag_config(tag, foreground='blue')

                break  # We found our tag, no need to check others

    def add_sentence_to_json(self, word, sentence):
        """
        Adds (original sentence, translated sentence, word-in-context translation)
        to the WSJ data for 'word'. Then saves to disk.
        """
        if word not in self.wsj_data:
            self.wsj_data[word] = []

        # We do the translations now:
        model, tokenizer = self.loaded_models.get(self.selected_language, (None, None))

        # Translate the entire sentence to English
        translated_sentence = translate_sentence_to_en(sentence, model, tokenizer)

        # Translate the word in the context of the sentence
        word_translation = self.get_word_translation(word, sentence, translated_sentence)

        # Append the tuple
        self.wsj_data[word].append((
            sentence,
            translated_sentence,
            word_translation
        ))

        # Save
        save_wsj(self.selected_language, self.wsj_data)

    def remove_sentence_from_json(self, word, sentence):
        """
        Removes the tuple for (sentence, ..., ...) from the WSJ data for 'word'.
        If that word's list becomes empty, remove the word entirely.
        Then save to disk.
        """
        tuples_for_word = self.wsj_data.get(word, [])
        new_list = []
        removed_something = False
        for tpl in tuples_for_word:
            orig_sentence = tpl[0]
            if orig_sentence == sentence:
                removed_something = True
                # skip adding it
            else:
                new_list.append(tpl)

        if removed_something:
            if len(new_list) == 0:
                # remove the word entirely
                del self.wsj_data[word]
            else:
                self.wsj_data[word] = new_list

            save_wsj(self.selected_language, self.wsj_data)

    def get_word_translation(self, word, sentence, translated_sentence):
        """
        Returns the English translation of 'word' in the context of 'sentence'.
        If 'word' is recognized as a purely grammatical particle, returns "no direct translation".
        Otherwise, tries to do a best-effort alignment (here we do a naive approach).
        """
        # If grammar-only, return no direct translation
        if is_grammatical_particle(word, self.selected_language):
            return "no direct translation"

        # A real alignment approach might use attention weights or a dictionary.
        # For demonstration, we do something naive:
        # We'll attempt to guess that the 'word' might appear in the translated_sentence, 
        # or just return an entire (word + " => " + translated_sentence) as a hack.

        # For English source, there's no actual "translation" needed:
        if self.selected_language == "English":
            return word

        # Very naive guess: Return the entire translated sentence, or you might 
        # do something more advanced. 
        # Alternatively, we might just say something like:
        return self.naive_word_alignment(word, sentence, translated_sentence)

    def naive_word_alignment(self, word, sentence, translated_sentence):
        """
        A placeholder for a naive approach to figure out a single-word translation
        from the translated sentence. 
        """
        # For demonstration, we simply say:
        # "Word '...' from <language> => [some guess from translated_sentence]"
        # A real solution would require more advanced alignment. 
        return f"(best guess) {translated_sentence}"

    def split_into_sentences(self, text):
        """
        A naive approach: split on '.' and strip. 
        Return a list of sentences that end with '.' 
        (except if the sentence is empty).
        """
        text = text.replace('\n', ' ')
        raw_sentences = text.split('.')
        sentences = []
        for raw in raw_sentences:
            s = raw.strip()
            if s:
                sentences.append(s + '.')
        return sentences


def main():
    root = tk.Tk()
    app = WordSentenceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
