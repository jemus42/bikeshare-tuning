.PHONY: all report

all:
	Rscript -e "targets::tar_make()"

report:
	quarto render tuning-report.qmd
