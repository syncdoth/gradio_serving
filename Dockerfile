FROM huggingface/transformers-pytorch-gpu:latest

RUN pip install gradio Jinja2

EXPOSE 7860

CMD ["python", "huggingface_serve.py"]
