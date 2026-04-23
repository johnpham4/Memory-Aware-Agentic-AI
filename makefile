endpoint:
	uv run python -m api.main

ui:
	uv run python -m app_ui.app

exports:
	uv run python -m scripts.export_conversations \
	--session-id f1d51814-10f4-4fc0-a532-11bfbfb39a34 \
	--output conversation.json