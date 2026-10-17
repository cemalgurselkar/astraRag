"""Real API runtime used by black-box startup-failure tests.

Unlike fake_app, this module wires the production factory so lifespan
exercises the actual AstraRAG bootstrap (settings, corpus, models).
"""

from astrarag.api.app import create_app

app = create_app()
