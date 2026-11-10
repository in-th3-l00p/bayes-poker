all: raport.pdf

figures/calibration.pdf data: bayes_poker.py
	python3 bayes_poker.py | tee results.txt

raport.pdf: raport.tex figures/calibration.pdf
	tectonic raport.tex

zip: raport.pdf
	rm -f submission.zip
	zip submission.zip raport.pdf bayes_poker.py

clean:
	rm -f raport.pdf submission.zip
