TEX=pdflatex
BIB=bibtex
MAIN=main

.PHONY: all clean

all: final_report.pdf

final_report.pdf: $(MAIN).tex references.bib tables/*.tex figures/*
	$(TEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(BIB) $(MAIN) || true
	$(TEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(TEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	cp $(MAIN).pdf final_report.pdf

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).pdf final_report.pdf
