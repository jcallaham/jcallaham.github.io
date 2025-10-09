# Build with Jekyll

JEKYLL = bundle exec jekyll
JEKYLL_OPTS = --incremental --trace

serve:
	$(JEKYLL) serve $(JEKYLL_OPTS) --drafts