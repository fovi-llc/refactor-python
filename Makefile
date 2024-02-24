# define the name of the virtual environment directory
VENV := .venv

# default target, when make executed without arguments
all: venv

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)

# venv is a shortcut target
venv: $(VENV)/bin/activate

install: venv
	./$(VENV)/bin/python3 -m pip install git+https://github.com/paul-gauthier/aider.git
	./$(VENV)/bin/python3 -m pip install -r requirements.txt

clean:
	rm -rf $(VENV) statement.py

statement.py:
	# Example from https://github.com/hamedsh/refactoring_book.  It is the initial pre-refactoring version of the statement.py file.
	wget https://raw.githubusercontent.com/hamedsh/refactoring_book/2e82aca59df22cc97016f8c39fc094689396c022/chapter_1/statement.py

# You'll need to set the OPENAI_API_KEY environment variable first (see https://platform.openai.com/api-keys)
test: venv statement.py
	./$(VENV)/bin/python3 extract.py statement.py
