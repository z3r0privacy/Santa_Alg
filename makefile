VENV=venv
BIN=$(VENV)/bin
NICE=nice -n 19

MET=sim
RND=42
ARGS=

run:
	$(NICE) $(BIN)/python run.py $(MET) --random-seed=$(RND) $(ARGS)

run-file:
	$(NICE) $(BIN)/python run.py $(MET) --random-seed=$(RND) --from-file=$(FILE) $(ARGS)

venv:
	virtualenv -p python3 $(VENV)

freeze: venv
	$(BIN)/pip freeze | grep -v "pkg-resources" > requirements.txt

install: venv
	$(BIN)/pip install -r requirements.txt
	mkdir -p solutions/ old-solutions/ checkpoints/ old-checkpoints/

explore:
	$(BIN)/ipython -m explore -i

uninstall:
	rm -rf $(VENV)

archive:
	mv solutions/*.csv old-solutions/
	for sol in old-solutions/*.csv; do gzip $$sol; done
	mv checkpoints/*.csv old-checkpoints/
	for chk in old-checkpoints/*.csv; do gzip $$chk; done

clean:
	find . -type f -name '*.pyc' -exec rm -f {} +
	find . -type d -name '__pycache__' -exec rm -rf {} +

