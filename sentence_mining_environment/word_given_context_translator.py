# import jieba
# from transformers import MarianMTModel, MarianTokenizer

# def translate_and_get_contextual_meaning(sentence, target_word):
#     # Load MarianMT model and tokenizer
#     model_name = "Helsinki-NLP/opus-mt-zh-en"
#     tokenizer = MarianTokenizer.from_pretrained(model_name)
#     model = MarianMTModel.from_pretrained(model_name)

#     # Perform word segmentation
#     segmented_sentence = list(jieba.cut(sentence))
#     print(f"Segmented Sentence (Initial): {segmented_sentence}")

#     # Ensure target word exists in segmentation
#     if target_word not in segmented_sentence:
#         # Fall back to character-level segmentation
#         print(f"Target word '{target_word}' not found. Falling back to character-level segmentation.")
#         segmented_sentence = list(sentence)
#         print(f"Segmented Sentence (Fallback): {segmented_sentence}")

#     # Check again for the target word
#     if target_word not in segmented_sentence:
#         raise ValueError(f"Word '{target_word}' not found in the segmented sentence: {segmented_sentence}")

#     # Translate the full sentence
#     inputs = tokenizer(sentence, return_tensors="pt", padding=True)
#     translated = model.generate(**inputs)
#     translated_sentence = tokenizer.decode(translated[0], skip_special_tokens=True)
#     translated_tokens = translated_sentence.split()

#     print(f"Original Sentence: {sentence}")
#     print(f"Translated Sentence: {translated_sentence}")
#     print(f"Tokenized Translated Sentence: {translated_tokens}")

#     # Align Chinese tokens with translated tokens
#     chinese_to_english_map = {
#         " ".join(segmented_sentence): translated_sentence
#     }

#     # Retrieve the contextual meaning of the target word
#     if target_word in segmented_sentence:
#         # Check if the word maps directly to a translated token
#         contextual_meaning = " ".join([token for token in translated_tokens if target_word in segmented_sentence])
#     else:
#         contextual_meaning = "Unknown"

#     print(f"Contextual meaning of '{target_word}': {contextual_meaning}")
#     return translated_sentence, contextual_meaning

# if __name__ == "__main__":
#     sentence = "我会吃水果"  # Input sentence
#     target_word = "果"  # Word to analyze

#     try:
#         translate_and_get_contextual_meaning(sentence, target_word)
#     except ValueError as e:
#         print(e)

import thulac

def segment_with_thulac(sentence):
    # Initialize THULAC in segmentation-only mode
    thu = thulac.thulac(seg_only=True)  # Disable POS tagging
    segmented_words = [word for word, _ in thu.cut(sentence)]
    return segmented_words

if __name__ == "__main__":
    sentence = "汉语是汉藏语系中最大的一支语族"
    segmented_words = segment_with_thulac(sentence)
    print(f"Segmented Words (THULAC): {segmented_words}")

