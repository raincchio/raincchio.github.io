# raincchio.github.io

Personal homepage of Xing Chen — https://raincchio.github.io/

Static HTML/CSS served by GitHub Pages (`.nojekyll` disables Jekyll).
Content lives in `data/*.json`; the HTML pages are generated from it and
committed, so Pages needs no build step.

## Editing (local admin)

```bash
python3 admin.py          # then open http://localhost:8000/admin
```

Tabs for each module (动态 / 论文 / 工作经历 / 学习经历 / 博客 / 站点信息);
add, edit, reorder, delete entries and hit 保存并重建 — the JSON is written
and all pages regenerate. Preview at http://localhost:8000/. When done,
commit and push (`git add -A && git commit && git push`).

Prefer hand-editing? Change `data/*.json` directly and run `python3 build.py`.

### Blog posts

Each post is a zip uploaded from the 博客 tab: one Markdown file plus the
images it references by relative path (e.g. `![](fig1.png)`). Title and date
come from front matter (`--- title: … / date: … ---`) or the first `#`
heading, and stay editable in the tab. Uploading with the same slug replaces
the post; source unpacks to `posts/<slug>/index.md`, output is generated at
`blog/<slug>/`. Markdown subset: headings, emphasis, links, images, code
(fenced + inline), lists, blockquotes, tables, hr.

## Structure

- `data/` — all content, one JSON per module (single source of truth)
- `build.py` — regenerates the pages below from `data/` (stdlib only)
- `admin.py` — local editing UI + preview server (stdlib only, binds 127.0.0.1)
- `index.html` — home: motto + latest 3 updates (auto-picked by date) + links
- `posts/` — blog sources: `posts/<slug>/index.md` + images (from uploaded zips)
- `blog/` — generated post list plus one directory per post (page + copied images)
- `publications/` — full publication list, grouped by year
- `experience/`, `education/` — timelines
- `style.css` — shared stylesheet (light/dark via `prefers-color-scheme`), hand-maintained
- `drafts/` — unpublished notes, not linked from the site

Don't edit the generated `index.html` files by hand — the next build
overwrites them. Layout changes go in `build.py`, styling in `style.css`.
