FROM python:3.12.1

RUN pip install uv

WORKDIR /app

COPY .python-version pyproject.toml uv.lock ./

RUN uv sync --locked

COPY app/app.py app/model.bin ./

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696", "--reload"]