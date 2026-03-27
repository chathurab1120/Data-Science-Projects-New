PYTHON := .venv/Scripts/python

.PHONY: install test run-gradio run-streamlit

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

run-gradio:
	$(PYTHON) app/gradio_app.py

run-streamlit:
	$(PYTHON) -m streamlit run app/streamlit_app.py
