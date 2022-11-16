#!/bin/bash

# Exit when any command fails
set -e

# Go to the directory containing the script
cd "${0%/*}"

# Build usage guide
#------------------
# Remove previous export
rm -vf -r _build_guide
# Move guide into source directory
cp guide.rst source/
# sphinx-build does not give precedence to a master_doc argument on the
# command line over the arugment in conf.py, so we have to edit the conf.py
sed -i 's/master_doc = "index"/master_doc = "guide"/' source/conf.py
sed -i 's/" Documentation"/" Usage Guide"/' source/conf.py
# Export rST to LaTeX with sphinx
sphinx-build -b latex -D master_doc='guide' -D latex_show_urls='footnote' source ./_build_guide source/guide.rst
# Remove docstring formatting indicators, which aren't stripped by
# sphinx-argparse
sed -in 's+^\\item\s*\[{[Rd]|}\]+\\item\[\]+' _build_guide/Echofilter.tex
# Build PDF from LaTeX using the make file provided by sphinx
make -C _build_guide
# Restore conf.py to processing index.rst instead of guide.rst
sed -i 's/master_doc = "guide"/master_doc = "index"/' source/conf.py
sed -i 's/" Usage Guide"/" Documentation"/' source/conf.py
rm source/guide.rst
mv _build_guide/Echofilter.pdf _build_guide/Echofilter_Usage-Guide.pdf

# Build full documentation
#-------------------------
# Remove previous export
rm -vf -r _build_pdf
# Make a copy of index.rst with indices removed from TOC
cp source/index.rst source/index_pdf.rst
sed -i '/py-modindex/d' source/index_pdf.rst
sed -i '/genindex/d' source/index_pdf.rst
# sphinx-build does not give precedence to a master_doc argument on the
# command line over the arugment in conf.py, so we have to edit the conf.py
sed -i 's/master_doc = "index"/master_doc = "index_pdf"/' source/conf.py
# Export rST to LaTeX with sphinx
sphinx-build -b latex source ./_build_pdf
# Remove docstring formatting indicators, which aren't stripped by
# sphinx-argparse
sed -in 's+^\\item\s*\[{[Rd]|}\]+\\item\[\]+' _build_pdf/Echofilter.tex
# Build PDF from LaTeX using the make file provided by sphinx
make -C _build_pdf
# Restore conf.py to processing index.rst instead of index_pdf.rst
sed -i 's/master_doc = "index_pdf"/master_doc = "index"/' source/conf.py
rm source/index_pdf.rst
