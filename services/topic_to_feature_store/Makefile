run-live-dev:
	copy live.dev.env .env
	poetry run python src/main.py

run-historical-dev:
	copy historical.dev.env .env
	poetry run python src/main.py
	
build:
	docker build -t topic_to_feature_store .

run-live: build
	docker run \
		--network=redpanda_network \
		--env-file live.prod.env \
		--env-file credentials.env \
		topic_to_feature_store