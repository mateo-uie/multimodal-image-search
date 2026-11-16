IMAGE := p4-clip
SCRIPT ?= src/entorno_test.py

build:
	docker build -t $(IMAGE) .

run:
	docker run -it \
		-v ./:/opt/project \
		-v p4-cache:/root/.cache \
		--rm \
		$(IMAGE) python $(SCRIPT)

shell:
	docker run -it \
		-v ./:/opt/project \
		-v p4-cache:/root/.cache \
		--rm \
		$(IMAGE) /bin/bash

web:
	docker run -it \
		-v ./:/opt/project \
		-v p4-cache:/root/.cache \
		-p 8000:8000 \
		--rm \
		$(IMAGE) python src/web_app.py
