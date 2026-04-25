PYTHONPATH=.

endpoint:
	uv run python -m api.main

ui:
	uv run python -m app_ui.app

jupyter:
	uv run jupyter lab --port 9999

oracle:
	docker compose up oracle -d