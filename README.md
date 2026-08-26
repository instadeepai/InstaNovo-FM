# InstaNovo-FM project page

This branch holds **only the built site**. The project code lives on `main`, and so do the
scripts that generate everything here — see `tools/projectpage/` on `main`.

## Layout

```
index.html              thin shell: a short summary and the explorer, embedded
umap/index.html         the explorer itself, also usable standalone / full screen
umap/data/              column-sharded point set + manifest.json
static/css|js|images    stylesheets, vendored Plotly, viewer, favicon, social card
.nojekyll               serve the tree as-is
```

## Serve it locally

```bash
python3 -m http.server 8000     # from this directory
```

Then open <http://localhost:8000/>. Every path in the site is **relative**, so the same tree
works at a domain root, in a repository subpath, and on the random subdomain that GitHub
assigns to a privately published site. If you add a link, keep it relative — a leading `/`
will break at least one of those three.

## Rebuild

From a checkout of `main`, with this worktree at `$GH`:

```bash
python3 tools/projectpage/build_fig3_data.py \
    --prototype ~/Downloads/umap/figure3_umap_explorer_3d.html \
    --out "$GH/umap" \
    --provenance ~/Downloads/umap/umap_3d_coords_for_kostas/provenance.json
python3 tools/projectpage/extract_viewer.py \
    --prototype ~/Downloads/umap/figure3_umap_explorer_3d.html --site "$GH"
python3 tools/projectpage/check_distance.py \
    --tex ../mass_spectrometry_foundation_model_manuscript/main.tex \
    --html "$GH/index.html"
```

`check_distance.py` guards the one editorial constraint that matters here: the page must read as
a summary of the manuscript, not a copy of it, because the work has not been published yet. It
fails if a long verbatim passage, the abstract, a figure legend or a figure image reappears.

## Publishing

Settings → Pages → *Deploy from a branch* → `gh-pages` / `(root)`, then set
**Visibility → Private** so only people with read access to this repository can reach it.
That option needs an organisation on GitHub Enterprise Cloud, which `instadeepai` has.
A privately published site is served from a unique random subdomain shown on that same
settings page, and changes take up to ten minutes to appear.

## Why the landing page is only a shell

The explorer is the deliverable for this version, so `index.html` is deliberately thin: a wordmark,
a summary of at most five lines, three links, and the explorer in an iframe taking the rest of the
viewport. The explorer detects being framed (`window.self !== window.top`) and sets
`data-framed="true"` on its root, which hides its own back-link and title so the identity is not
stated twice.

The explorer remains a standalone page. `umap/` works on its own, is what the "Full screen" link
opens, and is the URL to share directly.
