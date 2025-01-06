from transformers import MarianMTModel, MarianTokenizer

# Pre-load models for all languages
LANGUAGES = {
    "Spanish": "Helsinki-NLP/opus-mt-es-en",
    "Turkish": "Helsinki-NLP/opus-mt-tr-en",
    "Simplified Chinese": "Helsinki-NLP/opus-mt-zh-en",
    "Greek": "Helsinki-NLP/opus-mt-tc-big-el-en",
}

if __name__ == '__main__':
    for lang, model_name in LANGUAGES.items():
        print(f"Downloading and caching model for {lang}: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)  # Load tokenizer
        model = MarianMTModel.from_pretrained(model_name)  # Load model
        print(f"Model for {lang} loaded successfully.\n")
