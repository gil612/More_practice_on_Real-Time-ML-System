run-dev:
#   copy dev.env .env
	poetry run python src/main.py

# build:
# 	docker build -t trade_producer .

# run: build
# 	docker compose -f ../../docker-compose/redpanda.yml up -d
# 	timeout /t 10 /nobreak > NUL
# 	docker run \
# 		--network=redpanda-dev-cluster_redpanda_network \
# 		--env-file prod.env \
# 		trade_producer