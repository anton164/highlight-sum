from yaml import load
from question_generation.pipelines import pipeline
from datasets import load_dataset

xum = load_dataset("xsum")

nlp = pipeline("question-generation")

text_input = xum["test"][0]["document"]

print("Generating questions for", text_input)

print(
    nlp(
        text_input
    )
)
