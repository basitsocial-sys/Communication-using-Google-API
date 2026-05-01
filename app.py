import gradio as gr
from core import STTModule, GrammarModule, TranslationTTSModule, QAModule

stt = STTModule()
grammar = GrammarModule()
tts_trans = TranslationTTSModule()
qa_system = QAModule()

def process_pipeline(audio_file, text_input, input_lang, target_lang, use_grammar):
    is_urdu = (input_lang == "Urdu")
    
    # 1. Determine input (Text vs Audio)
    if text_input and text_input.strip():
        transcribed_text = text_input
    elif audio_file:
        transcribed_text = stt.convert_audio_to_text(audio_file, is_urdu=is_urdu)
    else:
        return "Please record audio or type some text.", "", "", None
    
    if "Error" in transcribed_text:
        return transcribed_text, "", "", None
        
    # 2. Grammar Correction (only if English or expected)
    corrected_text = transcribed_text
    if use_grammar and not is_urdu:
        corrected_text = grammar.correct_text(transcribed_text)
        
    # 3. Translation & TTS
    final_translation, audio_path = tts_trans.translate_and_speak(corrected_text, target_lang)
    return transcribed_text, corrected_text, final_translation, audio_path

def qa_pipeline_process(context, question, target_lang):
    eng_answer = qa_system.answer_question(question, context)
    if eng_answer.startswith("Please") or eng_answer.startswith("Error") or eng_answer.startswith("QA Model"):
        return eng_answer, ""
    
    final_translation, _ = tts_trans.translate_and_speak(eng_answer, target_lang)
    return eng_answer, final_translation

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

.gradio-container {
    background-color: #F2F4F8 !important;  
    color: #232F3E !important;             
    font-family: 'Open Sans', 'Helvetica Neue', sans-serif !important;
}
/* AWS Header Style */
h1, h2, h3 {
    color: #232F3E !important;
    font-weight: 700 !important;
}
h1 {
    text-align: left;
    background-color: #232F3E;
    color: #FFFFFF !important;
    padding: 20px;
    border-radius: 8px 8px 0 0;
    margin-bottom: 0px !important;
}
p.subtitle {
    background-color: #232F3E;
    color: #EAEDED !important;
    padding: 0 20px 20px 20px;
    border-radius: 0 0 8px 8px;
    margin-bottom: 20px !important;
    margin-top: 0px !important;
    font-size: 15px;
}
/* Cards (Boxes) */
.gr-box, fieldset {
    background-color: #FFFFFF !important;
    border: 1px solid #D5DBDB !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 1px 0 rgba(0,28,36,0.3) !important;
}
/* Primary Button (AWS Orange) */
button.primary {
    background-color: #FF9900 !important;
    color: #232F3E !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 2px !important;
    transition: background-color 0.2s;
}
button.primary:hover {
    background-color: #EC7211 !important;
}
/* Text inputs and labels */
textarea, input, span.label {
    color: #16191F !important;
    font-size: 14px !important;
}
footer { display: none !important; }
"""

# UI Construction
with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as demo:
    gr.Markdown("<h1> Multilingual ML Core Engine</h1>")
    gr.Markdown("<p class='subtitle'>Securely ingest, transcribe, correct, and translate text and speech using custom robust endpoints.</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input Source")
            input_text = gr.Textbox(label="Text Input", placeholder="Type here...", lines=3)
            input_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="OR Audio Input")
            
            gr.Markdown("### 2. Configuration")
            input_lang = gr.Radio(["English", "Urdu"], value="English", label="Source Language")
            use_grammar = gr.Checkbox(value=True, label="Apply Grammar Correction")
            target_lang = gr.Dropdown(tts_trans.get_supported_languages(), value="French", label="Target Language")
            
            submit_btn = gr.Button("Analyze and Translate", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### Outputs")
            raw_text = gr.Textbox(label="Raw Transcription", lines=2)
            corr_text = gr.Textbox(label="Corrected Text", lines=2)
            trans_text = gr.Textbox(label="Translated Text", lines=2)
            out_audio = gr.Audio(label="Translated Speech (TTS)", interactive=False)
            
    submit_btn.click(
        fn=process_pipeline,
        inputs=[input_audio, input_text, input_lang, target_lang, use_grammar],
        outputs=[raw_text, corr_text, trans_text, out_audio]
    )

    gr.Markdown("<hr>")
    gr.Markdown("<h2>Amazon AWS Q&A Knowledge Engine</h2>")
    with gr.Row():
        with gr.Column(scale=1):
            qa_context = gr.Textbox(label="Context / Passage (English)", lines=4)
            qa_question = gr.Textbox(label="Question (English)", lines=2)
            qa_lang = gr.Dropdown(tts_trans.get_supported_languages(), value="French", label="Target Answer Language")
            qa_btn = gr.Button("Extract & Translate Answer", variant="primary", size="lg")
        with gr.Column(scale=1):
            qa_eng_answer = gr.Textbox(label="Extracted Answer (English)", lines=2)
            qa_trans_answer = gr.Textbox(label="Translated Answer", lines=2)
            
    qa_btn.click(
        fn=qa_pipeline_process,
        inputs=[qa_context, qa_question, qa_lang],
        outputs=[qa_eng_answer, qa_trans_answer]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
