#################################################################################
## Install packages commands											     	#
#################################################################################


.PHONY: create_venv activate_venv compile install install-dev

# Detect the operating system
OS := $(shell uname -s)

ifeq ($(OS), Darwin) # macOS
    INSTALL_CMD=curl -LsSf https://astral.sh/uv/install.sh | sh
    ACTIVATE_CMD=source .venv/bin/activate
endif
ifeq ($(OS), Linux)
    INSTALL_CMD=curl -LsSf https://astral.sh/uv/install.sh | sh
    ACTIVATE_CMD=source .venv/bin/activate
endif
ifeq ($(OS), Windows_NT)
    INSTALL_CMD=powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ACTIVATE_CMD=.venv\Scripts\activate
endif


## Install uv, create a virtual environment
create_venv:
	@which uv > /dev/null 2>&1 || (echo "Installing uv..." && $(INSTALL_CMD))
	uv venv

## Activate virtual environment
activate:
	@echo "Activating virtual environment..."
	@$(ACTIVATE_CMD)

## Compile all the pinned requirements*.txt files from the unpinned requirements*.in files
compile:
	uv pip install --upgrade uv
	rm -f requirements/*.txt
	uv pip compile -p3.12 requirements/requirements.in --emit-index-url  --output-file=requirements/requirements.txt
	uv pip compile -p3.12 requirements/requirements-dev.in --output-file=requirements/requirements-dev.txt


## Install required packages
install:
	uv pip install --upgrade uv
	uv pip install -r requirements/requirements.txt

## Install required and development packages
install-dev:
	uv pip install --upgrade uv
	uv pip install -r requirements/requirements.txt \
	               -r requirements/requirements-dev.txt
	pre-commit install


##  Sync pinned dependencies with your virtual environment
sync:
	pip install --upgrade uv
	uv pip sync requirements/requirements.txt



###############################################################
# TRAINING COMMANDS                                                    #
###############################################################

## clean artifacts
clean:
	@echo ">>> cleaning files"
	rm ./models/*.joblib || true

## Preprocess dataset
preprocess-dataset:
	@echo ">>> generating dataset"
	# TODO: add commands here

## train the model
train:
	@echo ">>> training model"
	# TODO: add commands here


## serve trained model with a REST API
serve:
	@echo ">>> serving the trained model"
	# TODO: add commands here

## install dependencies -> clean artifacts -> generate dataset -> train -> serve
run-pipeline: install clean preprocess-dataset train serve

## linting and code style
lint:
	@echo ">>> linting and code style"
	ruff check

## create coverage report
coverage:
	@echo ">>> running coverage pytest"
	pytest  --cov=src tests/

## run unit tests in the current virtual environment
test:
	@echo ">>> running unit tests with the existing environment"
	pytest


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
