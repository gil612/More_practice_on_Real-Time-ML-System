run-live-dev:
	copy live.dev.env .env
	poetry run python src/main.py

run-historical-dev:
	copy historical.dev.env .env
	poetry run python src/main.py

build:
	docker build -t topic-to-feature-store .

run: build
	docker run \
	--network=redpanda_network \
	--env-file live.prod.env \
	--env-file hopsworks_credentials.env \
	topic-to-feature-store



