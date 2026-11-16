FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt

RUN uv venv /opt/.venv \
    && uv pip install --no-cache --torch-backend cpu -r /tmp/requirements.txt


WORKDIR /opt/project
