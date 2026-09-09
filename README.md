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


## The embeddings behind these maps

Both explorers carry an `embeddings ↗` link in the header, to
[`InstaDeepAI/InstaNovo-FM-embeddings`](https://huggingface.co/datasets/InstaDeepAI/InstaNovo-FM-embeddings)
— the 768-d vectors these layouts project, published as a `100k` config (the Figure 3
point set, with its exact coordinates) and a `1M` config. The explorers ship only
coordinates and display columns, so anyone wanting the representation itself needs the
dataset.

## `preview/` — the 1,000,000-point build

`preview/` is a second explorer over 1,000,000 LCFM test spectra, served at
`/InstaNovo-FM/preview/`. The site root is unchanged and remains the published
100,000-point view.

It exists to compare two candidate UMAP layouts of the same spectra, switchable from
the header:

| layout | what it is |
|---|---|
| `fig100k` — **loads first** | the layout published as Figure 3, exactly as it appears in the preprint. The manuscript never mentions the million-point version, so this is what a reader arriving from the paper sees |
| `transform` | the published layout extended to the full million: Figure 3's spectra keep their exact coordinates and the other 900,003 are placed into that same space by `umap-learn`'s `.transform()` |
| `native` | a `cuml.manifold.UMAP` fit over all 1,000,000 embeddings — a different embedding, not a denser Figure 3 |

Neither 1M layout appears in the manuscript, and their tooltips say so.

The two 1M layouts agree globally (Procrustes r = 0.858) and disagree locally — about
11% neighbourhood overlap — so they are genuinely different embeddings, not one fit at
two resolutions.

Every spectrum is drawn; nothing is sampled. `preview/selftest.html` asserts that, along
with nine other invariants, against whatever payload is in `preview/data/`.

Regenerate the payload from the main branch:

    scripts/explorer/build_explorer_data.py \
        --metadata embeddings.h5 \
        --layouts umap_1m_layouts.parquet \
        --layout native=native_x,native_y,native3_x,native3_y,native3_z \
        --layout transform=transform_x,transform_y,transform3_x,transform3_y,transform3_z \
        --layouts published_100k_anchor.parquet --layout fig100k=umap_x,umap_y \
        --order-layout native --default-layout fig100k --out preview/data

### Why this branch has no history

The payload is ~104 MB of binaries and git keeps every version forever, so a few
rebuilds would leave everyone cloning several hundred megabytes of superseded data.
The branch is therefore replaced by a single orphan commit each time the preview data
changes. The state served before the first such rewrite is tagged
`gh-pages-pre-preview-2026-09-09`, and the earlier landing-page state is tagged
`gh-pages-landing-2026-09-07`.
