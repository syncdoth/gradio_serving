import gradio as gr
from pororo import Pororo


class TranslateInterface:

    def __init__(self):
        self.mt = Pororo(task="translation",
                         lang="multi",
                         model='transformer.large.multi.fast.mtpg')

    def infer(self, text, src, tgt, top_p=0.9, temperature=1.0):
        if top_p == 0:  # turn off
            top_p = -1
        result = self.mt.predict(text=text, src=src, tgt=tgt, top_p=top_p, temperature=temperature)
        return result


def main():
    model = TranslateInterface()

    iface = gr.Interface(
        fn=model.infer,
        inputs=[
            gr.inputs.Textbox(lines=2, placeholder="Text here...", label='text'),
            gr.inputs.Dropdown(['en', 'ko'],
                               type="value",
                               default='ko',
                               label='source language'),
            gr.inputs.Dropdown(['en', 'ko'],
                               type="value",
                               default='en',
                               label='target language'),
            gr.inputs.Slider(minimum=0, maximum=1., step=None, default=0, label='top p sampling (0 means to turn off)'),
            gr.inputs.Slider(minimum=0, maximum=1., step=None, default=1., label='sampling temperature'),
        ],
        outputs=[
            gr.outputs.Textbox(type="auto", label="trainslated"),
        ],
        title='Pororo MT Demo',
        theme='huggingface',
    )
    iface.launch(server_name="0.0.0.0")


if __name__ == '__main__':
    main()
