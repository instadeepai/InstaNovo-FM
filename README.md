# InstaNovo-FM project page

This branch holds **only the built site**. The project code lives on `main`, and so do the
scripts that generate everything here — see `tools/projectpage/` on `main`.

## Layout

```
index.html              the project page
umap/index.html         the interactive UMAP explorer
umap/data/              column-sharded point set + manifest.json
static/css|js|images    stylesheets, vendored Plotly, viewer, figure renders
static/pdfs/            figure sources, behind each "full resolution" link
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
tools/projectpage/convert_figures.sh                       # figure PDFs -> WebP + PDF copies
python3 tools/projectpage/build_fig3_data.py \
    --prototype ~/Downloads/umap/figure3_umap_explorer_3d.html \
    --out "$GH/umap" \
    --provenance ~/Downloads/umap/umap_3d_coords_for_kostas/provenance.json
python3 tools/projectpage/extract_viewer.py \
    --prototype ~/Downloads/umap/figure3_umap_explorer_3d.html --site "$GH"
python3 tools/projectpage/check_content.py \
    --tex ../mass_spectrometry_foundation_model_manuscript/main.tex \
    --html "$GH/index.html"
```

## Publishing

Settings → Pages → *Deploy from a branch* → `gh-pages` / `(root)`, then set
**Visibility → Private** so only people with read access to this repository can reach it.
That option needs an organisation on GitHub Enterprise Cloud, which `instadeepai` has.
A privately published site is served from a unique random subdomain shown on that same
settings page, and changes take up to ten minutes to appear.
