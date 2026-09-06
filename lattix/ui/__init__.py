"""The browser UI (``lattix ui``): a local standard-library HTTP server (:mod:`lattix.ui.server`) serving one
offline page (``static/index.html``) that reads a deck, translates it, draws the beam line before and after
(:mod:`lattix.ui.model`), aligns and diffs the two element by element (:mod:`lattix.ui.compare`) and runs the
engines on demand (:mod:`lattix.ui.jobs`, :mod:`lattix.oracles.validate`)."""
