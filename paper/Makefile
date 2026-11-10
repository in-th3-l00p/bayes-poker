NAME := paper

.PHONY: all clean

all: $(NAME).pdf

$(NAME).pdf: $(NAME).tex refs.bib
	tectonic $(NAME).tex

clean:
	rm -f $(NAME).pdf $(NAME).aux $(NAME).bbl $(NAME).blg $(NAME).log \
	      $(NAME).out $(NAME).toc $(NAME).synctex.gz
