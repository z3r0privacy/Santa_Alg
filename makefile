VENV=venv
BIN=$(VENV)/bin
NICE=nice -n 19

MET=sim

run:
	$(NICE) $(BIN)/python run.py $(MET) $(ARGS)

run-file:
	$(NICE) $(BIN)/python run.py $(MET) --from-file=$(FILE)

venv:
	virtualenv -p python3 $(VENV)

freeze: venv
	$(BIN)/pip freeze | grep -v "pkg-resources" > requirements.txt

install: venv
	$(BIN)/pip install -r requirements.txt
	mkdir -p solutions/

explore:
	$(BIN)/ipython -m explore -i

uninstall:
	rm -rf $(VENV)

clean:
	find . -type f -name '*.pyc' -exec rm -f {} +
	find . -type d -name '__pycache__' -exec rm -rf {} +

