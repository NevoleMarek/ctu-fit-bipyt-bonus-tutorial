FROM python:3.11-buster

RUN pip install poetry==1.8.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock* README.md /app/

# Install dependencies, ignore the current project package
RUN poetry check --lock && poetry install --no-root --only main && rm -rf $POETRY_CACHE_DIR

# Copy the rest of the project files
COPY . /app

CMD ["poetry", "run", "fastapi", "run", "modelapp/main.py", "--port", "8000"]

