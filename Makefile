install:
	pip install -r requirements.txt

test:
	pytest src/tests

run:
	python -m jupyterlab notebooks/demo_notebook.ipynb
