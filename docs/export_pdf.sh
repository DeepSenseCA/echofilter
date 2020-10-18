#!/bin/bash

# Exit when any command fails
set -e

# Build usage guide
#------------------
# Remove previous export
rm -vf -r _build_guide
# sphinx-build does not give precedence to a master_doc argument on the
# command line over the arugment in conf.py, so we have to edit the conf.py
sed -i 's/master_doc = "index"/master_doc = "guide"/' conf.py
sed -i 's/" Documentation"/" Usage Guide"/' conf.py
# Export rST to LaTeX with sphinx
sphinx-build -b latex -D master_doc='guide' -D latex_show_urls='footnote' . ./_build_guide guide.rst
# Remove docstring formatting indicators, which aren't stripped by
# sphinx-argparse
sed -in 's+^\\item\s*\[{[Rd]|}\]+\\item\[\]+' _build_guide/Echofilter.tex
# Build PDF from LaTeX using the make file provided by sphinx
make -C _build_guide
# Restore conf.py to processing index.rst instead of guide.rst
sed -i 's/master_doc = "guide"/master_doc = "index"/' conf.py
sed -i 's/" Usage Guide"/" Documentation"/' conf.py

# Build full documentation
#-------------------------
# Remove previous export
rm -vf -r _build_pdf
# Make a copy of index.rst with indices removed from TOC
cp index.rst index_pdf.rst
sed -i '/py-modindex/d' ./index_pdf.rst
sed -i '/genindex/d' ./index_pdf.rst
# sphinx-build does not give precedence to a master_doc argument on the
# command line over the arugment in conf.py, so we have to edit the conf.py
sed -i 's/master_doc = "index"/master_doc = "index_pdf"/' conf.py
# Export rST to LaTeX with sphinx
sphinx-build -b latex . ./_build_pdf index.rst
# Remove docstring formatting indicators, which aren't stripped by
# sphinx-argparse
sed -in 's+^\\item\s*\[{[Rd]|}\]+\\item\[\]+' _build_pdf/Echofilter.tex
# Build PDF from LaTeX using the make file provided by sphinx
make -C _build_pdf
# Restore conf.py to processing index.rst instead of index_pdf.rst
sed -i 's/master_doc = "index_pdf"/master_doc = "index"/' conf.py
