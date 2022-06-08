build:
	docker build -t hnet .
up:
	docker-compose up -d
down:
	docker-compose down
run-hnet:
	docker-compose up -d && docker exec -it hnet python3 source/main.py
