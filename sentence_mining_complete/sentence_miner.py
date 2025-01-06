import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import json
import re
import difflib  # NEW: We'll use difflib to compare translations

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
try:
    import thulac
except ImportError:
    thulac = None
    print("Warning: THULAC is not installed. Chinese segmentation won't work.")


# Map each language to a separate JSON file
LANGUAGE_TO_JSON = {
    "English":            "word_sentence_data_en.json",
    "Spanish":            "word_sentence_data_es.json",
    "Turkish":            "word_sentence_data_tr.json",
    "Simplified Chinese": "word_sentence_data_zh_hans.json",
    "Traditional Chinese":"word_sentence_data_zh_hant.json",
    "Greek":              "word_sentence_data_el.json",
}

# Map each language to an appropriate MarianMT model (or None if no translation needed)
LANGUAGE_TO_MODEL = {
    "English":            None,  # No translation needed from English -> English
    "Spanish":            "Helsinki-NLP/opus-mt-es-en",
    "Turkish":            "Helsinki-NLP/opus-mt-tr-en",
    "Simplified Chinese": "Helsinki-NLP/opus-mt-zh-en",
    "Traditional Chinese":"Helsinki-NLP/opus-mt-zh-en",
    # Feel free to switch to a smaller Greek model if the big one is problematic:
    # "Greek":            "Helsinki-NLP/opus-mt-el-en",
    "Greek":             "Helsinki-NLP/opus-mt-tc-big-el-en",
}

# Some purely grammatical words that might yield "no direct translation" in Chinese
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
        return sentence  # fallback: no translation

    batch = tokenizer([sentence], return_tensors="pt", truncation=True)
    gen = model.generate(**batch)
    translation = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return translation.strip()


def segment_with_thulac(sentence):
    """
    Uses THULAC to segment a Chinese sentence into a list of words.
    If THULAC is not available, raises an exception or you can handle a fallback.
    """
    if not thulac:
        raise ImportError("THULAC is not installed or not imported.")
    thu = thulac.thulac(seg_only=True)
    segmented_pairs = thu.cut(sentence)
    return [wp[0] for wp in segmented_pairs]


def naive_segment(sentence, language):
    """
    Given a sentence and language, returns a list of words.
    For Chinese, calls THULAC (if available). For others, uses whitespace splitting.
    """
    if language in ["Simplified Chinese", "Traditional Chinese"]:
        # Attempt to segment with THULAC
        try:
            return segment_with_thulac(sentence)
        except:
            # fallback: naive split by whitespace
            return sentence.split()
    else:
        # Simple whitespace-based tokenization
        return sentence.split()


def is_grammatical_particle(word, language):
    """
    Very naive check to see if 'word' might be a purely grammatical particle
    that has no direct English translation. Only specialized for Chinese here.
    """
    if language in ["Simplified Chinese", "Traditional Chinese"]:
        if word in CHINESE_FUNCTION_WORDS:
            return True
    return False


class WordSentenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Language Word-Sentence JSON App")

        # Language selection frame
        self.language_frame = tk.Frame(self.root)
        self.language_frame.pack(padx=20, pady=20)

        self.language_label = tk.Label(self.language_frame, text="Select the language:")
        self.language_label.pack(side="left", padx=5)

        self.language_var = tk.StringVar()
        self.language_combobox = ttk.Combobox(
            self.language_frame,
            textvariable=self.language_var,
            values=list(LANGUAGE_TO_JSON.keys()),
            width=25
        )
        self.language_combobox["state"] = "normal"
        self.language_combobox.set("English")  # default
        self.language_combobox.pack(side="left", padx=5)

        self.language_button = tk.Button(
            self.language_frame,
            text="Confirm",
            command=self.confirm_language
        )
        self.language_button.pack(side="left", padx=5)

        # Frames
        self.start_frame = None
        self.display_frame = None

        # Data
        self.selected_language = None
        self.user_text = ""
        self.wsj_data = {}

        # Text widget references
        self.text_display = None
        self.tag_word_map = {}

        # Cache for translation models
        self.loaded_models = {}

    def confirm_language(self):
        lang = self.language_var.get()
        if lang not in LANGUAGE_TO_JSON:
            messagebox.showerror("Error", "Please select a valid language.")
            return

        self.selected_language = lang
        self.wsj_data = load_wsj(self.selected_language)

        # Possibly pre-load the translation model for this language
        if self.selected_language not in self.loaded_models:
            model, tokenizer = load_translation_model(self.selected_language)
            self.loaded_models[self.selected_language] = (model, tokenizer)

        self.language_frame.destroy()

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
        filepath = filedialog.askopenfilename(
            title="Select a text file",
            filetypes=[("Text Files", "*.txt")]
        )
        if filepath:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    self.user_text = f.read()
                self.show_text_display()
            except Exception as e:
                print(f"Error reading file: {e}")

    def open_paste_window(self):
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
        self.user_text = text_widget.get("1.0", tk.END).strip()
        window.destroy()
        self.show_text_display()

    def show_text_display(self):
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

        # Split user text into sentences
        sentences = self.split_into_sentences(self.user_text)

        self.tag_word_map = {}
        tag_index = 0

        for sentence in sentences:
            words = naive_segment(sentence, self.selected_language)

            for word in words:
                tag = f"tag_{tag_index}"
                tag_index += 1

                self.tag_word_map[tag] = {
                    "word": word,
                    "sentence": sentence
                }

                self.text_display.insert('end', word + " ", tag)

                # Color depends on if it's already in the JSON
                if self.word_in_json(word, sentence):
                    self.text_display.tag_config(tag, foreground='red')
                else:
                    self.text_display.tag_config(tag, foreground='blue')

                self.text_display.tag_bind(tag, "<Button-1>", self.on_word_click)

            self.text_display.insert('end', "\n\n")

    def word_in_json(self, word, sentence):
        if word not in self.wsj_data:
            return False
        for tpl in self.wsj_data[word]:
            if tpl[0] == sentence:
                return True
        return False

    def on_word_click(self, event):
        x, y = event.x, event.y
        index = event.widget.index(f"@{x},{y}")
        tags_at_index = event.widget.tag_names(index)

        for tag in tags_at_index:
            if tag in self.tag_word_map:
                info = self.tag_word_map[tag]
                word = info["word"]
                sentence = info["sentence"]

                if not self.word_in_json(word, sentence):
                    self.add_sentence_to_json(word, sentence)
                    event.widget.tag_config(tag, foreground='red')
                else:
                    self.remove_sentence_from_json(word, sentence)
                    event.widget.tag_config(tag, foreground='blue')
                break

    def add_sentence_to_json(self, word, sentence):
        if word not in self.wsj_data:
            self.wsj_data[word] = []

        # Translate the entire sentence
        model, tokenizer = self.loaded_models.get(self.selected_language, (None, None))
        translated_sentence = translate_sentence_to_en(sentence, model, tokenizer)

        # Find a contextual meaning for the word
        word_translation = self.get_word_translation(word, sentence, translated_sentence)

        # Append (orig_sentence, translated_sentence, word_translation)
        self.wsj_data[word].append((sentence, translated_sentence, word_translation))
        save_wsj(self.selected_language, self.wsj_data)

    def remove_sentence_from_json(self, word, sentence):
        tuples_for_word = self.wsj_data.get(word, [])
        new_list = []
        removed_something = False
        for tpl in tuples_for_word:
            if tpl[0] == sentence:
                removed_something = True
            else:
                new_list.append(tpl)

        if removed_something:
            if len(new_list) == 0:
                del self.wsj_data[word]
            else:
                self.wsj_data[word] = new_list
            save_wsj(self.selected_language, self.wsj_data)

    def get_word_translation(self, word, sentence, translated_sentence):
        """
        Returns the English translation of 'word' in context.
        1) If 'word' is a grammatical particle (Chinese), return "no direct translation".
        2) Otherwise, do a naive approach:
           - Translate the full sentence -> T1
           - Translate the sentence with 'word' removed -> T2
           - Compare T1 and T2 in English to guess the difference
           - If difference is empty, fall back to single-word translation
        """
        if is_grammatical_particle(word, self.selected_language):
            return "no direct translation"

        # No model loaded? Just return the entire sentence
        model, tokenizer = self.loaded_models.get(self.selected_language, (None, None))
        if not model or not tokenizer:
            return f"(best guess) {translated_sentence}"

        # 1) Full sentence translation (we already have 'translated_sentence' as T1)
        T1 = translated_sentence
        T1_tokens = T1.split()

        # 2) Remove the word from the source sentence
        #    We'll do a naive removal of all occurrences of 'word' as a whole token (regex).
        #    This can fail if punctuation is attached, but let's keep it simple.
        sentence_minus_word = self.remove_word_naively(sentence, word)

        # 3) Translate T2
        T2 = translate_sentence_to_en(sentence_minus_word, model, tokenizer)
        T2_tokens = T2.split()

        # 4) Compare T1_tokens vs T2_tokens for differences
        #    We'll gather the tokens that appear in T1 but not in T2, or changed.
        diff = self.compare_token_lists(T1_tokens, T2_tokens)

        # If 'diff' is empty, we fallback to single-word translation
        if not diff:
            single_word_translation = translate_sentence_to_en(word, model, tokenizer)
            if single_word_translation.strip():
                return single_word_translation
            else:
                return f"(best guess) {T1}"
        else:
            # Join the diff tokens to form a guess
            return " ".join(diff)

    def remove_word_naively(self, sentence, word):
        """
        Remove the given word (as a whole token) from 'sentence' using a naive regex.
        For example, "Hola a todos" removing "todos" -> "Hola a"
        """
        pattern = r'\b' + re.escape(word) + r'\b'
        # We'll remove all occurrences. This might remove multiple matches if they exist.
        return re.sub(pattern, '', sentence)

    def compare_token_lists(self, tokens1, tokens2):
        """
        Use difflib.SequenceMatcher to find which tokens changed from tokens1 -> tokens2.
        We'll return a list of tokens that are in tokens1 but not in tokens2.
        """
        matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
        diff_tokens = []
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            # If it's a 'delete' or 'replace', tokens1[i1:i2] were removed/changed
            if op in ['delete', 'replace']:
                diff_tokens.extend(tokens1[i1:i2])
        return diff_tokens

    def split_into_sentences(self, text):
        """
        UPDATED: Treat every piece of punctuation like a period (including Chinese punctuation).
        We'll replace any punctuation with a single '.' and then split on '.'.
        Finally, we re-append '.' to each chunk for uniformity.
        """
        # Replace newlines with space
        text = text.replace('\n', ' ')
        # We treat a broad range of punctuation (English + Chinese) as periods:
        #   .  ,  ?  !  ;  :  "  '  ，  。  ！  ？  …  ：  etc.
        # You can extend this pattern if needed.
        punctuation_pattern = r'[.,!?;:"\'，。！？…：]+'
        # Replace all these with a single '.'
        text = re.sub(punctuation_pattern, '.', text)

        # Now split on '.' 
        raw_sentences = text.split('.')

        sentences = []
        for raw in raw_sentences:
            s = raw.strip()
            if s:
                # Re-append a period
                sentences.append(s + '.')

        return sentences


def main():
    root = tk.Tk()
    app = WordSentenceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
