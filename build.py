#!/usr/bin/env python3
"""Generate the static site from data/*.json.

Usage: python3 build.py
Only the Python standard library is required.

Generated files: index.html, blog/ (index + one page per post),
publications/index.html, experience/index.html, education/index.html.
Blog pages other than those listed in data/posts.json are removed on build.
"""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

NAV = [
    ("Blog", "/blog/"),
    ("Publications", "/publications/"),
    ("Experience", "/experience/"),
    ("Education", "/education/"),
]

HOME_UPDATE_COUNT = 6


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def esc(s):
    return html.escape(s or "", quote=True)


def date_key(s):
    """Parse the first date in a string ('2026.8', '2026-09-01', '2019.9 – 2025.6')
    into a sortable (year, month, day) tuple; the range form yields its start."""
    m = re.search(r"(\d{4})(?:[.\-/](\d{1,2}))?(?:[.\-/](\d{1,2}))?", s or "")
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))


def fmt_date(s):
    y, m, _ = date_key(s)
    return f"{y}.{m:02d}" if m else (str(y) if y else "")


def collect_updates(updates, pubs, posts, experience):
    """Merge manual updates with entries auto-derived from the other modules."""
    items = [{"key": date_key(u["date"]), "date": fmt_date(u["date"]) or u["date"], "text": u["text"]}
             for u in updates]
    for p in pubs:
        venue = f" ({esc(p['venue'])})" if p.get("venue") else ""
        items.append({
            "key": date_key(p.get("year", "")),
            "date": fmt_date(p.get("year", "")),
            "text": f'Paper: <a href="/publications/">“{esc(p["title"])}”</a>{venue}.',
        })
    for p in posts:
        items.append({
            "key": date_key(p.get("date", "")),
            "date": fmt_date(p.get("date", "")),
            "text": f'Blog post: <a href="/blog/{esc(p["slug"])}.html">{esc(p["title"])}</a>.',
        })
    for e in experience:
        items.append({
            "key": date_key(e.get("date", "")),
            "date": fmt_date(e.get("date", "")),
            "text": f'Joined <a href="/experience/">{esc(e["org"])}</a> as {esc(e["title"])}.',
        })
    items.sort(key=lambda i: i["key"], reverse=True)
    return items[:HOME_UPDATE_COUNT]


def page(site, title, active, body):
    nav_links = "\n        ".join(
        f'<a href="{url}"{" class=\"active\"" if label == active else ""}>{label}</a>'
        for label, url in NAV
    )
    full_title = esc(title) if title else esc(site["name"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{full_title}</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <header class="site-header">
    <nav>
      <a class="brand" href="/">{esc(site["name"])}</a>
      <div class="nav-links">
        {nav_links}
      </div>
    </nav>
  </header>

  <main>
{body}
  </main>

  <footer class="site-footer">
    <p>© 2026 {esc(site["name"])} · <a href="{esc(site["source_url"])}">Source</a></p>
  </footer>
</body>
</html>
"""


def render_home(site, latest):
    links = "\n        ".join(
        f'<li><a href="{esc(l["url"])}">{esc(l["label"])}</a></li>' for l in site["links"]
    )
    items = "\n        ".join(
        f'<li>\n          <span class="date">{esc(u["date"])}</span>\n'
        f'          <span>{u["text"]}</span>\n        </li>'
        for u in latest
    )
    avatar = ""
    if site.get("avatar"):
        avatar = f'\n      <img class="avatar" src="{esc(site["avatar"])}" alt="Portrait of {esc(site["name"])}">'
    body = f"""    <section class="hero home-hero">
      <div class="hero-text">
        <h1>{esc(site["name"])}</h1>
        <p class="tagline">{esc(site["tagline"])}</p>
        <p class="bio goal">“{esc(site["motto"])}”</p>
        <ul class="links">
          {links}
        </ul>
      </div>{avatar}
    </section>

    <section id="updates">
      <h2>Latest Updates</h2>
      <ul class="news-list">
        {items}
      </ul>
    </section>"""
    return page(site, "", "", body)


def render_publications(site, pubs):
    def authors_html(a):
        return esc(a).replace(esc(site["highlight"]), f"<strong>{esc(site['highlight'])}</strong>")

    years = sorted({str(date_key(p.get("year", ""))[0]) for p in pubs}, reverse=True)
    groups = []
    for year in years:
        in_year = sorted(
            (p for p in pubs if str(date_key(p.get("year", ""))[0]) == year),
            key=lambda p: date_key(p.get("year", "")),
            reverse=True,
        )
        entries = "\n".join(
            f"""        <li>
          <p class="pub-title">{esc(p["title"])}</p>
          <p class="pub-authors">{authors_html(p["authors"])}</p>
          <p class="pub-venue">{esc(p["venue"])}</p>
        </li>"""
            for p in in_year
        )
        groups.append(
            f"""    <section class="year-group">
      <h2>{esc(year) if year != "0" else "Other"}</h2>
      <ol class="pub-list">
{entries}
      </ol>
    </section>"""
        )
    scholar = next((l["url"] for l in site["links"] if "scholar" in l["url"]), "")
    body = f"""    <section class="hero">
      <h1>Publications</h1>
      <p class="section-note">Also on <a href="{esc(scholar)}">Google Scholar</a>. Author lists as indexed by Scholar.</p>
    </section>

""" + "\n\n".join(groups)
    return page(site, f"Publications · {site['name']}", "Publications", body)


def render_timeline(site, title, entries):
    items = "\n".join(
        f"""        <li>
          <p class="entry-date">{esc(e["date"])}</p>
          <p class="entry-title">{esc(e["title"])} <span class="entry-org">· {esc(e["org"])}</span></p>
          <p class="entry-desc">{e["desc"]}</p>
        </li>"""
        for e in entries
    )
    body = f"""    <section class="hero">
      <h1>{title}</h1>
    </section>

    <section>
      <ul class="timeline">
{items}
      </ul>
    </section>"""
    return page(site, f"{title} · {site['name']}", title, body)


def render_blog_index(site, posts):
    if posts:
        rows = "\n".join(
            f'        <li><span class="date">{esc(p["date"])}</span>'
            f'<a href="/blog/{esc(p["slug"])}.html">{esc(p["title"])}</a></li>'
            for p in sorted(posts, key=lambda p: p.get("date", ""), reverse=True)
        )
        listing = f'      <ul class="post-list">\n{rows}\n      </ul>'
    else:
        listing = '      <div class="empty-state">\n        <p>Posts coming soon.</p>\n      </div>'
    body = f"""    <section class="hero">
      <h1>Blog</h1>
      <p class="bio">{esc(site["blog_intro"])}</p>
{listing}
    </section>"""
    return page(site, f"Blog · {site['name']}", "Blog", body)


def render_post(site, post):
    body = f"""    <article class="hero post">
      <h1>{esc(post["title"])}</h1>
      <p class="post-meta">{esc(post["date"])}</p>
{post["content"]}
    </article>"""
    return page(site, f"{post['title']} · {site['name']}", "Blog", body)


def main():
    site = load("site")
    pubs, posts, experience = load("publications"), load("posts"), load("experience")
    latest = collect_updates(load("updates"), pubs, posts, experience)

    out = {
        ROOT / "index.html": render_home(site, latest),
        ROOT / "publications" / "index.html": render_publications(site, pubs),
        ROOT / "experience" / "index.html": render_timeline(site, "Experience", experience),
        ROOT / "education" / "index.html": render_timeline(site, "Education", load("education")),
        ROOT / "blog" / "index.html": render_blog_index(site, posts),
    }
    for post in posts:
        out[ROOT / "blog" / f"{post['slug']}.html"] = render_post(site, post)

    for old in (ROOT / "blog").glob("*.html"):
        if old not in out:
            old.unlink()

    for path, content in out.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"built {len(out)} pages")


if __name__ == "__main__":
    main()
