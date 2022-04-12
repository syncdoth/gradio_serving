FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel
RUN pip install transformers gradio Jinja2

COPY . /app
WORKDIR /app

EXPOSE 7860

CMD ["python", "huggingface_serve.py"]
