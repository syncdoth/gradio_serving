FROM huggingface/transformers-pytorch-cpu:latest

RUN pip install gradio Jinja2

COPY . /app

WORKDIR /app

EXPOSE 7860

CMD ["python", "huggingface_serve.py"]
