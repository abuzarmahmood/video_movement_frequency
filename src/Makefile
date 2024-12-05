.PHONY: setup clean

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

setup: $(VENV)/bin/activate install-packages install-requirements

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install-requirements: $(VENV)/bin/activate
	$(PIP) install -r ../requirements.txt

install-packages:
	cat ../packages.txt | xargs sudo apt-get install -y

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
