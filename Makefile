.PHONY: setup clean 

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

setup: $(VENV)/bin/activate install-packages install-requirements install-dev-requirements

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install-requirements: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

install-packages:
	@for pkg in $$(cat packages.txt); do \
		if ! dpkg -l $$pkg >/dev/null 2>&1; then \
			sudo apt-get install -y $$pkg; \
		fi \
	done

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
