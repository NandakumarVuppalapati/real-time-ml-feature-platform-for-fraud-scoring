.PHONY: up down logs demo train offline-features test lint sample-data clean

up:            ## Build and start the core stack (Kafka, Redis, MLflow, Flink, Feast, API)
	docker compose up --build -d
	@echo "API:          http://localhost:8000/docs"
	@echo "MLflow UI:    http://localhost:5001"
	@echo "Flink UI:     http://localhost:8081"

down:           ## Stop and remove all containers
	docker compose down

logs:            ## Tail logs for every service
	docker compose logs -f

demo:            ## Start replaying the transaction dataset into Kafka
	docker compose --profile demo up producer

offline-features: ## Recompute batch (Spark) features from raw/sample data
	docker compose run --rm training python prepare_offline_features.py

train:           ## Train + register a new model version, alias it @champion
	docker compose run --rm training python train.py

sample-data:      ## Regenerate the small synthetic sample dataset
	python3 data/generate_sample.py

test:             ## Run the test suite
	pytest tests/ -v

lint:             ## Static checks
	ruff check . || true

clean:            ## Stop containers and remove volumes (wipes Redis/MLflow/Feast state)
	docker compose down -v
