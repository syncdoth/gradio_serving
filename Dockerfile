FROM python:3.8
RUN pip install torch transformers gradio Jinja2

COPY . /app
WORKDIR /app

EXPOSE 7860

CMD ["python", "huggingface_serve.py"]
