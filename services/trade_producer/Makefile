run-live-dev:
	copy live.dev.env .env
	poetry run python src/main.py

run-historical-dev:
	copy historical.dev.env .env
	poetry run python src/main.py

build:
	docker build -t trade_producer .

run-live: build
	docker run \
		--network=redpanda_network \
		--env-file live.prod.env \
		trade_producer

run-historical: build
	docker run \
		--network=redpanda_network \
		--env-file historical.prod.env \
		trade_producer
