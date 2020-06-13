#!/bin/sh

pdflatex -shell-escape slides.tex && \
  bibtex slides && \
  pdflatex -shell-escape slides.tex && \
  pdflatex -shell-escape slides.tex

