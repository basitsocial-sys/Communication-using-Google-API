import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
from transformers import pipeline
import os
import tempfile

class STTModule:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def convert_audio_to_text(self, audio_path, is_urdu=False):
        """Converts spoken audio to text."""
        lang = "ur-PK" if is_urdu else "en-US"
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language=lang)
                return text
        except sr.UnknownValueError:
            return "Error: Could not understand audio."
        except sr.RequestError as e:
            return f"Error: STT service unavailable ({e})"
        except Exception as e:
            return f"Error reading audio file: {e}"


class GrammarModule:
    def __init__(self):
        # Using a T5 Neural Network for state-of-the-art Grammar Correction
        self.corrector = None
        try:
            self.corrector = pipeline(
                "text2text-generation", 
                model="vennify/t5-base-grammar-correction",
                device=-1
            )
        except Exception as e:
            print(f"Warning: Grammar model unavailable: {e}")

    def correct_text(self, text):
        """Checks and corrects English grammar using ML."""
        if not text or "Error:" in text:
            return text
        if self.corrector is None:
            return text + " (Model Limit Reached / Offline)"
        
        try:
            # The model requires the prefix "grammar: " to function correctly
            result = self.corrector("grammar: " + text, max_length=128)
            corrected = result[0]['generated_text']
            return corrected
        except Exception as e:
            return text + " (Correction Failed)"


class TranslationTTSModule:
    def __init__(self):
        self.supported_langs = {
            "French": "fr",
            "Arabic": "ar",
            "Urdu": "ur",
            "Spanish": "es",
            "German": "de",
            "Chinese": "zh-CN",
            "English": "en"
        }

    def get_supported_languages(self):
        return list(self.supported_langs.keys())

    def translate_and_speak(self, text, target_language_name):
        """Translates text and generates a TTS audio file."""
        if not text or "Error:" in text:
            return "", None
            
        target_code = self.supported_langs.get(target_language_name, "en")
        
        # 1. Translate
        try:
            translator = GoogleTranslator(source='auto', target=target_code)
            translated_text = translator.translate(text)
        except Exception as e:
            return f"Translation Error: {e}", None
            
        # 2. TTS
        try:
            tts = gTTS(text=translated_text, lang=target_code)
            # Save to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file.close() # close it so gTTS can write
            tts.save(temp_file.name)
            return translated_text, temp_file.name
        except Exception as e:
            return translated_text, None


class QAModule:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        try:
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
            model_name = "deepset/roberta-base-squad2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: QA model unavailable: {e}")

    def answer_question(self, question, context):
        if not question or not context:
            return "Please provide both context and question."
        if self.model is None or self.tokenizer is None:
            return "QA Model Offline. (Check console for download progress)"
        try:
            import torch
            inputs = self.tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()
            
            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
            answer = self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
            return answer
        except Exception as e:
            return f"Error finding answer: {e}"
