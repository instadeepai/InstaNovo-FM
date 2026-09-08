# InstaNovo-FM project page

This branch holds **only the built site**. The project code lives on `main`, and so do the
scripts that generate everything here — see `tools/projectpage/` on `main`.

## Layout

```
index.html              the UMAP explorer -- what the site serves at its root
data/                   column-sharded point set + manifest.json
selftest.html           the explorer's own diagnostic page
landing.html            the project landing page: still served, but nothing links to
                        it -- kept so the root can be swapped back
umap/index.html         stub redirecting to ../ , so old /umap/ links still work
static/css|js|images    stylesheets, vendored Plotly, viewer, favicon, social card
.nojekyll               serve the tree as-is
```

The root deliberately serves the **explorer**, not a landing page. See
[Swapping the root back](#swapping-the-root-back).

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

## Swapping the root back

The landing page is still in the tree, so putting it back at the root is three moves
and three path rewrites -- the reverse of what the switch did:

```bash
git mv index.html umap/index.html      # explorer back under umap/ (replaces the stub)
git mv selftest.html umap/selftest.html
git mv data umap/data
git mv landing.html index.html

sed -i 's|"static/|"../static/|g' umap/index.html umap/selftest.html
sed -i 's|href="https://github.com/instadeepai/InstaNovo-FM"|href="../"|' umap/index.html
sed -i 's|href="./"|href="umap/"|g' index.html
```

Or recover the whole site as it stood before the switch, exactly:

```bash
git checkout -B gh-pages gh-pages-landing-2026-09-07
```

That tag is the last commit with the landing page at the root. Either way, check the
result with the local server above: every path in the tree is relative, and the
explorer resolves its data as `data/` relative to whatever document loads it, so
moving the explorer means moving `data/` with it.

## Why the landing page is only a shell

The explorer is the deliverable for this version, so `landing.html` is deliberately thin: a
wordmark, a summary of at most five lines, three links to the explorer, and the figures. It is no
longer what the site serves at its root, and it no longer embeds the explorer in an iframe -- it
links to it. The explorer still detects being framed (`window.self !== window.top`) and sets
`data-framed="true"` on its root, which hides its own back-link and title; that path is unused
while nothing frames it, and costs nothing to keep.

The URL to share is the site root.
