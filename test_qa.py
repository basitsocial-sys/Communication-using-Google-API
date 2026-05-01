from transformers import pipeline
try:
    p = pipeline('question-answering', model='deepset/roberta-base-squad2', device=-1)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
