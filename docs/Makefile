# Makefile for Sphinx documentation
#
# You can set these variables from the command line.
SPHINXBUILD   ?= sphinx-build
SPHINXOPTS    ?= --keep-going
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make help" is always printed when you enter `make` without arguments
.PHONY: help
help:
	@echo "Sphinx documentation build targets:"
	@echo ""
	@echo "  make html       - Build HTML documentation"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make linkcheck  - Check for broken links"
	@echo "  make spelling   - Check spelling"
	@echo "  make serve      - Serve documentation locally on port 8000"
	@echo ""
	@echo "Variables:"
	@echo "  SPHINXBUILD   = $(SPHINXBUILD)"
	@echo "  SPHINXOPTS    = $(SPHINXOPTS)"
	@echo "  SOURCEDIR     = $(SOURCEDIR)"
	@echo "  BUILDDIR      = $(BUILDDIR)"

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Removing build artifacts..."
	@rm -rf $(BUILDDIR)
	@echo "Build artifacts removed successfully."

# Build HTML documentation
.PHONY: html
html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@if [ $$? -eq 0 ]; then \
		echo ""; \
		echo "Build finished successfully! The HTML pages are in $(BUILDDIR)/html"; \
	else \
		echo "Build failed!"; \
		exit 1; \
	fi

# Check for broken links
.PHONY: linkcheck
linkcheck:
	@echo "Checking for broken links..."
	@$(SPHINXBUILD) -b linkcheck $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/linkcheck
	@if [ $$? -eq 0 ]; then \
		echo ""; \
		echo "Link check finished! Results are in $(BUILDDIR)/linkcheck"; \
	else \
		echo "Link check failed!"; \
		exit 1; \
	fi

# Check spelling
.PHONY: spelling
spelling:
	@echo "Checking spelling..."
	@$(SPHINXBUILD) -b spelling $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/spelling
	@if [ $$? -eq 0 ]; then \
		echo ""; \
		echo "Spelling check finished! Results are in $(BUILDDIR)/spelling"; \
	else \
		echo "Spelling check failed!"; \
		exit 1; \
	fi

# Serve documentation locally
.PHONY: serve
serve: html
	@echo "Serving documentation on http://localhost:8000"
	@echo "Press Ctrl+C to stop the server"
	@cd $(BUILDDIR)/html && python -m http.server 8000

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -b $@ $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/$@
