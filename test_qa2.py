from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import traceback
try:
    print("Loading model...")
    model_name = "deepset/roberta-base-squad2"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Getting pipeline...")
    p = pipeline('question-answering', model=model, tokenizer=tokenizer)
    print("Testing pipeline...")
    res = p(question="What is the capital of France?", context="The capital of France is Paris.")
    print("Success:", res)
except Exception as e:
    traceback.print_exc()
