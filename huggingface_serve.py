import argparse

import gradio as gr
from transformers import pipeline


class ModelInterface:

    def __init__(self, model_name):
        self.classifier = pipeline('sentiment-analysis', model=model_name)

    def infer(self, text):
        result = self.classifier(text)[0]
        return result['label'], result['score']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default="nlptown/bert-base-multilingual-uncased-sentiment")
    args = parser.parse_args()
    model = ModelInterface(args.model_name)

    iface = gr.Interface(
        fn=model.infer,
        inputs=gr.inputs.Textbox(lines=2, placeholder="Text here...", label='text'),
        outputs=[
            gr.outputs.Textbox(type="auto", label="stars"),
            gr.outputs.Textbox(type="number", label="confidence"),
        ],
        title='sentiment analysis demo',
        theme='huggingface',
        interpretation='default',
    )
    iface.launch(share=True)


if __name__ == '__main__':
    main()