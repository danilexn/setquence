# makefile inspired from the huggingface promptsource repository
# https://github.com/bigscience-workshop/promptsource/blob/main/Makefile
#
# run this Makefile on individual scripts,
# > make check_file=biodatasets/<dataset_name>/<dataset_name>.py

.PHONY: quality

setquence_dir := setquence

# Format source code automatically (all files)

quality:
	black --line-length 119 --target-version py38 $(check_file)
	isort $(check_file) --line-length 119
	flake8 $(check_file) --max-line-length 119 --ignore=E203,E231

quality_all:
	black --check --line-length 119 --target-version py38 $(setquence_dir)
	isort --check-only $(setquence_dir) --line-length 119
	flake8 $(setquence_dir) --max-line-length 119 --ignore=E203,E231

modify_all:
	black --line-length 119 --target-version py38 $(setquence_dir)
	isort $(setquence_dir) --line-length 119
	flake8 $(setquence_dir) --max-line-length 119 --ignore=E203,E231
