
latex -interaction=nonstopmode RomainMencattiniMemoire.tex
bibtex RomainMencattiniMemoire.aux

latex -interaction=nonstopmode RomainMencattiniMemoire.tex
latex -interaction=nonstopmode RomainMencattiniMemoire.tex
sleep 1
pdflatex RomainMencattiniMemoire.tex
