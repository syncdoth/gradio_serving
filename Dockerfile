FROM huggingface/transformers-pytorch-gpu:latest

RUN pip install gradio

EXPOSE 80

CMD ["python", "huggingface_serve.py"]
