import nbformat as nbf

def create_stt_nb():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Module 1: Speech to Text (STT)\\nDemonstrating how to convert voice to text using `SpeechRecognition`."),
        nbf.v4.new_code_cell("import speech_recognition as sr\\n\\ndef test_stt(audio_path, lang='en-US'):\\n    r = sr.Recognizer()\\n    with sr.AudioFile(audio_path) as source:\\n        audio = r.record(source)\\n        return r.recognize_google(audio, language=lang)\\n\\nprint('STT Logic Ready! Import `core.py` to use the main class.')")
    ]
    with open('STT_Module.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_grammar_nb():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Module 2: Grammar Correction\\nDemonstrating how to fix grammatical errors using `language-tool-python`."),
        nbf.v4.new_code_cell("from transformers import pipeline\\n\\ntry:\\n    corrector = pipeline('text2text-generation', model='vennify/t5-base-grammar-correction')\\n    text = 'A sentence with a error in the hitchy.'\\n    result = corrector('grammar: ' + text, max_length=128)\\n    corrected = result[0]['generated_text']\\n    print(f'Original: {text}')\\n    print(f'Corrected: {corrected}')\\nexcept Exception as e:\\n    print(f'Model Error: {e}')\\n")
    ]
    with open('Grammar_Module.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_tts_nb():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Module 3: Translation and Text-to-Speech\\nDemonstrating GoogleTranslator and gTTS audio generation."),
        nbf.v4.new_code_cell("from deep_translator import GoogleTranslator\\nfrom gtts import gTTS\\nimport IPython.display as ipd\\n\\ntext = 'Hello world'\\ntranslated = GoogleTranslator(source='auto', target='fr').translate(text)\\ntts = gTTS(translated, lang='fr')\\ntts.save('test.mp3')\\n\\nprint(f'Translated: {translated}')\\nipd.Audio('test.mp3')")
    ]
    with open('Translation_TTS_Module.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_stt_nb()
    create_grammar_nb()
    create_tts_nb()
    print("Notebooks created successfully!")
