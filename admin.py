#!/usr/bin/env python3
"""Local editing backend for the homepage.

Usage: python3 admin.py [port]     (default port 8000)
Then open http://localhost:8000/admin

Edits data/*.json through a web form and rebuilds the static pages on
every save (via build.py). Binds to 127.0.0.1 only — local use, do not
deploy. Only the Python standard library is required.
"""
import base64
import datetime
import http.server
import importlib
import io
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import zipfile
from pathlib import Path

import build

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
POSTS = ROOT / "posts"
ALLOWED = {"site", "updates", "publications", "experience", "education", "posts"}


def slugify(s):
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return s[:60]


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # no caching for previews, so a refresh after save shows the rebuild
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/admin":
            self._send(200, ADMIN_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/data":
            payload = {n: json.loads((DATA / f"{n}.json").read_text(encoding="utf-8")) for n in ALLOWED}
            self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
        elif self.path.startswith("/api/blog/md?"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            slug = (qs.get("slug") or [""])[0]
            md = POSTS / slug / "index.md"
            if slugify(slug) != slug or not slug or not md.exists():
                return self._send(404, b'{"error":"no such post"}', "application/json")
            self._send(200, json.dumps({"slug": slug, "content": md.read_text(encoding="utf-8")}).encode("utf-8"),
                       "application/json")
        else:
            super().do_GET()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            if self.path == "/api/save":
                name = req.get("name")
                if name not in ALLOWED:
                    return self._send(400, b'{"error":"bad module name"}', "application/json")
                (DATA / f"{name}.json").write_text(
                    json.dumps(req["data"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                importlib.reload(build).main()  # hot-reload: build.py edits apply without restart
                self._send(200, b'{"ok":true}', "application/json")
            elif self.path == "/api/avatar":
                self.save_avatar(req)
            elif self.path == "/api/blog/upload":
                self.blog_upload(req)
            elif self.path == "/api/blog/md":
                self.blog_save_md(req)
            elif self.path == "/api/blog/delete":
                self.blog_delete(req)
            elif self.path == "/api/publish":
                self.publish(req)
            else:
                self._send(404, b'{"error":"not found"}', "application/json")
        except Exception as e:  # report to the frontend instead of hanging the request
            self._send(500, json.dumps({"error": str(e)}).encode("utf-8"), "application/json")

    def save_avatar(self, req):
        ext = (req.get("filename", "").rsplit(".", 1)[-1] or "").lower()
        if ext == "jpeg":
            ext = "jpg"
        if ext not in {"jpg", "png", "webp", "gif"}:
            return self._send(400, b'{"error":"only jpg/png/webp/gif"}', "application/json")
        raw = base64.b64decode(req.get("data", ""))
        if len(raw) > 5 * 1024 * 1024:
            return self._send(400, b'{"error":"image larger than 5MB"}', "application/json")
        assets = ROOT / "assets"
        assets.mkdir(exist_ok=True)
        for old in assets.glob("avatar.*"):
            old.unlink()
        (assets / f"avatar.{ext}").write_bytes(raw)
        site = json.loads((DATA / "site.json").read_text(encoding="utf-8"))
        site["avatar"] = f"/assets/avatar.{ext}"
        (DATA / "site.json").write_text(
            json.dumps(site, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        importlib.reload(build).main()
        self._send(200, json.dumps({"ok": True, "avatar": site["avatar"]}).encode("utf-8"),
                   "application/json")

    def blog_upload(self, req):
        raw = base64.b64decode(req.get("data", ""))
        if len(raw) > 50 * 1024 * 1024:
            return self._send(400, b'{"error":"zip larger than 50MB"}', "application/json")
        slug = slugify(req.get("slug") or Path(req.get("filename", "")).stem)
        if not slug:
            return self._send(400, b'{"error":"cannot derive a slug"}', "application/json")

        zf = zipfile.ZipFile(io.BytesIO(raw))
        names = [n for n in zf.namelist()
                 if not n.endswith("/") and "__MACOSX" not in n
                 and not Path(n).name.startswith(".")]
        mds = [n for n in names if n.lower().endswith(".md")]
        if not mds:
            return self._send(400, json.dumps({"error": "No Markdown (.md) file found in the archive"}).encode("utf-8"),
                              "application/json")
        # prefer a root-level md named index/readme/post; extract the rest relative to its dir
        md_name = min(mds, key=lambda n: (
            n.count("/"), 0 if Path(n).stem.lower() in {"index", "readme", "post"} else 1, n))
        base = str(Path(md_name).parent)
        base = "" if base == "." else base + "/"

        dest = POSTS / slug
        if dest.exists():
            shutil.rmtree(dest)
        for n in names:
            if base and not n.startswith(base):
                continue
            rel = Path(n[len(base):]) if n != md_name else Path("index.md")
            if not rel.parts or ".." in rel.parts:  # zip-slip guard
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(n))

        b = importlib.reload(build)
        meta, body = b.parse_front_matter((dest / "index.md").read_text(encoding="utf-8"))
        m = re.match(r"\s*#\s+(.+)\n", body)
        title = meta.get("title") or (m.group(1).strip() if m else slug)
        date = meta.get("date") or datetime.date.today().strftime("%Y-%m-%d")

        posts = json.loads((DATA / "posts.json").read_text(encoding="utf-8"))
        replaced = next((p for p in posts if p.get("slug") == slug), None)
        if replaced:  # a body replacement keeps title/date edited in the admin
            title, date = replaced.get("title") or title, replaced.get("date") or date
            posts.remove(replaced)
        posts.insert(0, {"slug": slug, "title": title, "date": date})
        (DATA / "posts.json").write_text(
            json.dumps(posts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        b.main()
        self._send(200, json.dumps({"ok": True, "slug": slug, "title": title,
                                    "replaced": bool(replaced)}).encode("utf-8"),
                   "application/json")

    def blog_save_md(self, req):
        slug = req.get("slug", "")
        if not slug or slugify(slug) != slug:
            return self._send(400, b'{"error":"bad slug"}', "application/json")
        content = req.get("content", "")
        dest = POSTS / slug
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "index.md").write_text(content, encoding="utf-8")

        b = importlib.reload(build)
        posts = json.loads((DATA / "posts.json").read_text(encoding="utf-8"))
        # the md is the source of truth: sync title/date from front matter or the first heading
        meta, body = b.parse_front_matter(content)
        m = re.match(r"\s*#\s+(.+)\n", body)
        md_title = meta.get("title") or (m.group(1).strip() if m else None)
        md_date = meta.get("date")
        entry = next((p for p in posts if p.get("slug") == slug), None)
        if entry is None:
            posts.insert(0, {"slug": slug, "title": md_title or slug,
                             "date": md_date or datetime.date.today().strftime("%Y-%m-%d")})
        else:
            if md_title:
                entry["title"] = md_title
            if md_date:
                entry["date"] = md_date
        (DATA / "posts.json").write_text(
            json.dumps(posts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        b.main()
        self._send(200, json.dumps({"ok": True, "slug": slug}).encode("utf-8"), "application/json")

    def blog_delete(self, req):
        slug = req.get("slug", "")
        posts = json.loads((DATA / "posts.json").read_text(encoding="utf-8"))
        if not any(p.get("slug") == slug for p in posts):
            return self._send(404, b'{"error":"no such post"}', "application/json")
        shutil.rmtree(POSTS / slug, ignore_errors=True)
        shutil.rmtree(ROOT / "blog" / slug, ignore_errors=True)
        posts = [p for p in posts if p.get("slug") != slug]
        (DATA / "posts.json").write_text(
            json.dumps(posts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        importlib.reload(build).main()
        self._send(200, b'{"ok":true}', "application/json")

    def publish(self, req):
        def git(*args):
            return subprocess.run(["git", "-C", str(ROOT), *args],
                                  capture_output=True, text=True, timeout=60)

        dirty = git("status", "--porcelain").stdout.strip()
        parts = []
        if dirty:
            git("add", "-A")
            msg = (req.get("message") or "").strip() or \
                f"Update site content ({datetime.date.today().isoformat()})"
            r = git("commit", "-m", msg)
            if r.returncode != 0:
                return self._send(500, json.dumps(
                    {"error": (r.stderr or r.stdout).strip()}).encode("utf-8"), "application/json")
            parts.append("Created commit " + git("rev-parse", "--short", "HEAD").stdout.strip())
        ahead = git("rev-list", "--count", "@{u}..HEAD").stdout.strip() or "0"
        if ahead == "0":
            return self._send(200, json.dumps(
                {"ok": True, "detail": "Nothing to publish; local content is up to date with GitHub"},
                ensure_ascii=False).encode("utf-8"), "application/json")
        r = git("push", "origin", "main")
        if r.returncode != 0:
            return self._send(500, json.dumps(
                {"error": (r.stderr or r.stdout).strip()}).encode("utf-8"), "application/json")
        parts.append(f"Pushed {ahead} commit(s) to GitHub; Pages should update in a minute or two")
        self._send(200, json.dumps({"ok": True, "detail": "; ".join(parts)},
                                   ensure_ascii=False).encode("utf-8"), "application/json")

    def log_message(self, fmt, *args):
        pass  # keep the console quiet


ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Homepage Admin</title>
<style>
  :root { --accent:#2563eb; --border:#dfe3e8; --muted:#6b7280; --bg:#f6f7f9; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif;
         background:var(--bg); color:#1a1d21; font-size:15px; }
  header { background:#fff; border-bottom:1px solid var(--border); padding:0.7rem 1.2rem;
           display:flex; align-items:center; gap:1rem; position:sticky; top:0; z-index:5; }
  header h1 { font-size:1.05rem; margin:0 auto 0 0; }
  header a { color:var(--accent); text-decoration:none; font-size:0.9rem; }
  .tabs { display:flex; gap:0.4rem; padding:0.9rem 1.2rem 0; flex-wrap:wrap; }
  .tabs button { border:1px solid var(--border); background:#fff; border-radius:999px;
                 padding:0.35rem 0.95rem; cursor:pointer; font-size:0.9rem; }
  .tabs button.on { background:var(--accent); border-color:var(--accent); color:#fff; }
  main { max-width:56rem; margin:0 auto; padding:1rem 1.2rem 4rem; }
  .row { background:#fff; border:1px solid var(--border); border-radius:10px;
         padding:0.9rem 1rem; margin-bottom:0.8rem; }
  .row .ops { display:flex; gap:0.4rem; justify-content:flex-end; margin-top:0.5rem; }
  .ops button { border:1px solid var(--border); background:#fff; border-radius:6px;
                padding:0.15rem 0.6rem; cursor:pointer; font-size:0.85rem; }
  .ops button.del { color:#b91c1c; }
  label { display:block; font-size:0.8rem; color:var(--muted); margin:0.5rem 0 0.15rem; }
  input, textarea { width:100%; border:1px solid var(--border); border-radius:6px;
                    padding:0.4rem 0.55rem; font:inherit; }
  input.short { max-width:14rem; }
  textarea { resize:vertical; }
  .bar { display:flex; gap:0.7rem; align-items:center; margin:1rem 0; }
  .bar button { border:none; border-radius:8px; padding:0.5rem 1.2rem; cursor:pointer; font-size:0.95rem; }
  .bar .save { background:var(--accent); color:#fff; }
  .bar .add { background:#fff; border:1px solid var(--border); }
  #msg { font-size:0.9rem; color:var(--muted); }
  #msg.ok { color:#15803d; }  #msg.err { color:#b91c1c; }
  .hint { font-size:0.85rem; color:var(--muted); margin:0.2rem 0 0.8rem; }
</style>
</head>
<body>
<header>
  <h1>Homepage Admin</h1>
  <button id="pubBtn" style="border:none;border-radius:8px;background:#15803d;color:#fff;
    padding:0.4rem 1rem;cursor:pointer;font-size:0.9rem">Publish to GitHub</button>
  <a href="/" target="_blank">Preview site ↗</a>
</header>
<div class="tabs" id="tabs"></div>
<main>
  <p class="hint" id="hint"></p>
  <div id="list"></div>
  <div class="bar">
    <button class="add" id="addBtn">+ Add entry</button>
    <button class="save" id="saveBtn">Save and rebuild</button>
    <span id="msg"></span>
  </div>
</main>
<script>
const SCHEMAS = {
  updates: { title:'Updates', hint:'The homepage shows the latest 6 updates from publications, posts, and work experience, ordered by date. Add other news here, such as awards or talks. Dates accept 2026 or 2026.03.',
    fields:[
      {k:'date', label:'Date', short:true},
      {k:'text', label:'Content (plain text or HTML, including <a href="/publications/">links</a>)', type:'textarea', rows:2}],
    blank:{date:'', text:''} },
  publications: { title:'Publications', hint:'Publications are grouped by year in reverse chronological order. X. Chen is highlighted in author lists. New publications also appear in homepage updates.',
    fields:[
      {k:'year', label:'Date (e.g. 2026.03)', short:true},
      {k:'title', label:'Title'},
      {k:'authors', label:'Authors (initials, comma-separated)'},
      {k:'venue', label:'Venue (conference/journal and year, or arXiv ID)'},
      {k:'url', label:'Paper URL (publisher, DOI, arXiv, or OpenReview)'}],
    blank:{year:'', title:'', authors:'', venue:'', url:''} },
  experience: { title:'Experience', hint:'Entries appear in list order; put the most recent first. Descriptions accept HTML links.',
    fields:[
      {k:'date', label:'Date range (e.g. 2024.8 – Present)', short:true},
      {k:'title', label:'Role'},
      {k:'org', label:'Company / organization'},
      {k:'desc', label:'Brief description (accepts <a href="…">links</a>)', type:'textarea', rows:2}],
    blank:{date:'', title:'', org:'', desc:''} },
  education: { title:'Education', hint:'Entries appear in list order; put the most recent first. Descriptions accept HTML links, such as an advisor homepage.',
    fields:[
      {k:'date', label:'Date range', short:true},
      {k:'title', label:'Degree / field'},
      {k:'org', label:'Institution'},
      {k:'desc', label:'Brief description (accepts <a href="…">links</a>)', type:'textarea', rows:2}],
    blank:{date:'', title:'', org:'', desc:''} },
  posts: { title:'Blog', hint:'Manage each post separately: upload a zip (Markdown and images) or use Edit Markdown. Titles and dates are parsed from front matter (--- title: … / date: … ---) or the first # heading, and can be edited below before saving.', blog:true },
  site: { title:'Site settings', hint:'Name, tagline, biography, and contact links.', site:true },
};
const ORDER = ['updates','publications','experience','education','posts','site'];
let data = null, cur = 'updates';

const $ = id => document.getElementById(id);

function el(tag, attrs, ...children) {
  const e = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs||{})) {
    if (k === 'oninput' || k === 'onclick') e[k] = v; else e.setAttribute(k, v);
  }
  for (const c of children) e.append(c);
  return e;
}

function field(obj, f) {
  const wrap = el('div');
  if (f.type === 'checkbox') {
    const lab = el('label', {style:'display:flex;align-items:center;gap:0.4rem;cursor:pointer;margin-top:0.6rem'});
    const box = el('input', {type:'checkbox', style:'width:auto', oninput: e => obj[f.k] = e.target.checked});
    box.checked = !!obj[f.k];
    lab.append(box, f.label);
    wrap.append(lab);
    return wrap;
  }
  wrap.append(el('label', {}, f.label));
  let input;
  if (f.type === 'textarea') {
    input = el('textarea', {rows: f.rows || 3, oninput: e => obj[f.k] = e.target.value});
    input.value = obj[f.k] || '';
  } else {
    input = el('input', {class: f.short ? 'short' : '', oninput: e => obj[f.k] = e.target.value});
    input.value = obj[f.k] || '';
  }
  wrap.append(input);
  return wrap;
}

function renderList() {
  const schema = SCHEMAS[cur], list = $('list');
  list.replaceChildren();
  const arr = data[cur];
  arr.forEach((item, i) => {
    const row = el('div', {class:'row'});
    for (const f of schema.fields) row.append(field(item, f));
    const ops = el('div', {class:'ops'});
    const move = d => { const j = i + d;
      if (j < 0 || j >= arr.length) return;
      [arr[i], arr[j]] = [arr[j], arr[i]]; renderList(); };
    ops.append(
      el('button', {onclick: () => move(-1)}, '↑'),
      el('button', {onclick: () => move(1)}, '↓'),
      el('button', {class:'del', onclick: () => { arr.splice(i,1); renderList(); }}, 'Delete'));
    row.append(ops);
    list.append(row);
  });
  $('addBtn').hidden = false;
}

function renderSite() {
  const list = $('list'), s = data.site;
  list.replaceChildren();

  const avatarRow = el('div', {class:'row'});
  avatarRow.append(el('label', {}, 'Avatar (shown as a circle; square image recommended; jpg/png/webp, up to 5 MB)'));
  const line = el('div', {style:'display:flex;gap:1rem;align-items:center'});
  if (s.avatar) {
    const img = el('img', {src: s.avatar + '?t=' + Date.now(),
      style:'width:72px;height:72px;border-radius:50%;object-fit:cover;border:1px solid var(--border)'});
    line.append(img);
  } else {
    line.append(el('span', {style:'color:var(--muted);font-size:0.9rem'}, 'No image uploaded'));
  }
  const file = el('input', {type:'file', accept:'image/*', style:'border:none;padding:0'});
  file.onchange = async () => {
    const f = file.files[0];
    if (!f) return;
    const msg = $('msg');
    msg.className = ''; msg.textContent = 'Uploading…';
    const b64 = await new Promise((ok, err) => {
      const r = new FileReader();
      r.onload = () => ok(r.result.split(',')[1]);
      r.onerror = err;
      r.readAsDataURL(f);
    });
    try {
      const res = await fetch('/api/avatar', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({filename: f.name, data: b64})});
      const out = await res.json();
      if (!res.ok) throw new Error(out.error || res.status);
      s.avatar = out.avatar;
      msg.className = 'ok'; msg.textContent = '✓ Avatar updated and site rebuilt';
      renderSite();
    } catch (e) {
      msg.className = 'err'; msg.textContent = 'Upload failed: ' + e.message;
    }
  };
  line.append(file);
  avatarRow.append(line);
  list.append(avatarRow);

  const row = el('div', {class:'row'});
  const scalars = [
    {k:'name', label:'Name'},
    {k:'tagline', label:'Tagline (below the homepage title)'},
    {k:'bio', label:'Biography (homepage; accepts HTML links)', type:'textarea', rows:4},
    {k:'highlight', label:'Author name to highlight in publications'},
    {k:'blog_intro', label:'Blog introduction'},
    {k:'source_url', label:'Footer source link'},
    {k:'goatcounter', label:'GoatCounter site code (registered at goatcounter.com; leave blank to disable analytics)', short:true},
  ];
  for (const f of scalars) row.append(field(s, f));
  list.append(row);

  const linksRow = el('div', {class:'row'});
  linksRow.append(el('label', {}, 'Contact / external links'));
  s.links.forEach((l, i) => {
    const line = el('div', {style:'display:flex;gap:0.5rem;margin-bottom:0.4rem;align-items:center'});
    const lab = el('input', {class:'short', oninput: e => l.label = e.target.value}); lab.value = l.label || '';
    const url = el('input', {oninput: e => l.url = e.target.value}); url.value = l.url || '';
    const del = el('button', {class:'del', onclick: () => { s.links.splice(i,1); renderSite(); },
                              style:'border:1px solid var(--border);background:#fff;border-radius:6px;cursor:pointer'}, 'Remove');
    line.append(lab, url, del);
    linksRow.append(line);
  });
  linksRow.append(el('button', {class:'add', style:'border:1px solid var(--border);background:#fff;border-radius:6px;padding:0.2rem 0.7rem;cursor:pointer',
    onclick: () => { s.links.push({label:'', url:''}); renderSite(); }}, '+ Add link'));
  list.append(linksRow);
  $('addBtn').hidden = true;
}

function readB64(f) {
  return new Promise((ok, err) => {
    const r = new FileReader();
    r.onload = () => ok(r.result.split(',')[1]);
    r.onerror = err;
    r.readAsDataURL(f);
  });
}

async function postJSON(url, payload) {
  const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)});
  const out = await res.json();
  if (!res.ok) throw new Error(out.error || res.status);
  return out;
}

async function uploadZip(fileInput, slug) {
  const f = fileInput.files[0];
  if (!f) return;
  const msg = $('msg');
  msg.className = ''; msg.textContent = 'Uploading and parsing…';
  try {
    const out = await postJSON('/api/blog/upload',
      {filename: f.name, slug: slug || '', data: await readB64(f)});
    data = await (await fetch('/api/data')).json();
    render();
    msg.className = 'ok';
    msg.textContent = (out.replaced ? '✓ Replaced and rebuilt: ' : '✓ Created and rebuilt: ') +
      out.slug + ' (' + out.title + ')';
  } catch (e) {
    msg.className = 'err'; msg.textContent = 'Upload failed: ' + e.message;
  }
}

let mdOpen = null, mdDraft = '', mdNew = false;

function mdEditor(slug) {
  const wrap = el('div', {style:'margin-top:0.7rem'});
  const ta = el('textarea', {rows:24,
    style:'font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:13px;line-height:1.55',
    oninput: e => mdDraft = e.target.value});
  ta.value = mdDraft;
  const bar = el('div', {class:'ops', style:'justify-content:flex-start'});
  bar.append(
    el('button', {onclick: () => saveMd(slug)}, 'Save Markdown and rebuild'),
    el('button', {onclick: () => { mdOpen = null; mdNew = false; render(); }}, 'Close'));
  wrap.append(ta, bar);
  return wrap;
}

async function openMd(slug) {
  if (mdOpen === slug && !mdNew) { mdOpen = null; render(); return; }
  const msg = $('msg');
  try {
    const res = await fetch('/api/blog/md?slug=' + encodeURIComponent(slug));
    const out = await res.json();
    if (!res.ok) throw new Error(out.error || res.status);
    mdNew = false; mdOpen = slug; mdDraft = out.content;
    render();
  } catch (e) {
    msg.className = 'err'; msg.textContent = 'Could not load: ' + e.message;
  }
}

async function saveMd(slug) {
  const msg = $('msg');
  msg.className = ''; msg.textContent = 'Saving…';
  try {
    await postJSON('/api/blog/md', {slug, content: mdDraft});
    data = await (await fetch('/api/data')).json();
    mdNew = false;
    render();
    $('msg').className = 'ok'; $('msg').textContent = '✓ Saved and rebuilt: ' + slug + '; refresh the preview to see the changes';
  } catch (e) {
    msg.className = 'err'; msg.textContent = 'Save failed: ' + e.message;
  }
}

function renderBlog() {
  const list = $('list');
  list.replaceChildren();

  const up = el('div', {class:'row'});
  up.append(el('label', {}, 'Upload a new post (zip)'));
  const line = el('div', {style:'display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap'});
  const slugIn = el('input', {class:'short', placeholder:'Slug (optional; defaults to zip filename)'});
  const file = el('input', {type:'file', accept:'.zip,application/zip', style:'border:none;padding:0'});
  file.onchange = () => uploadZip(file, slugIn.value);
  const newBtn = el('button', {style:'border:1px solid var(--border);background:#fff;border-radius:6px;padding:0.3rem 0.8rem;cursor:pointer',
    onclick: () => {
      const slug = slugIn.value.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-+|-+$/g,'');
      if (!slug) { $('msg').className='err'; $('msg').textContent='Enter a slug on the left before creating a post'; return; }
      mdNew = true; mdOpen = slug;
      mdDraft = '---\\ntitle: Post title\\ndate: ' + new Date().toISOString().slice(0,10) + '\\n---\\n\\nPost content…\\n';
      render();
    }}, '+ New Markdown post');
  line.append(slugIn, file, newBtn);
  up.append(line);
  up.append(el('div', {class:'hint', style:'margin:0.5rem 0 0'},
    'Include one .md file (with optional front matter) and its images in the zip. Use relative image paths, e.g. ![](fig1.png). ' +
    'Write math as $inline$ or $$display$$ (rendered with KaTeX). Set image widths with ![](fig1.png =60%) or =420 (pixels); smaller images are centered. ' +
    'An existing slug replaces the old post. Use New Markdown post to write posts without images in the editor.'));
  if (mdNew && mdOpen) {
    up.append(el('div', {style:'margin-top:0.6rem;font-weight:600'}, 'New post: ' + mdOpen));
    up.append(mdEditor(mdOpen));
  }
  list.append(up);

  data.posts.forEach(p => {
    const row = el('div', {class:'row'});
    const head = el('div', {style:'display:flex;gap:0.7rem;align-items:baseline'});
    head.append(el('strong', {}, p.slug),
      el('a', {href:'/blog/' + p.slug + '/', target:'_blank', style:'font-size:0.85rem'}, 'Preview ↗'));
    row.append(head);
    row.append(field(p, {k:'title', label:'Title'}));
    row.append(field(p, {k:'date', label:'Date (e.g. 2026-09-04)', short:true}));
    const ops = el('div', {class:'ops'});
    const rep = el('input', {type:'file', accept:'.zip,application/zip', style:'display:none'});
    rep.onchange = () => uploadZip(rep, p.slug);
    const editBtn = el('button', {onclick: () => openMd(p.slug)},
      mdOpen === p.slug && !mdNew ? 'Close editor' : 'Edit Markdown');
    const repBtn = el('button', {onclick: () => rep.click()}, 'Replace zip');
    const delBtn = el('button', {class:'del', onclick: async () => {
      if (!confirm('Delete post "' + p.slug + '"? This removes both source files and generated pages.')) return;
      const msg = $('msg');
      try {
        await postJSON('/api/blog/delete', {slug: p.slug});
        if (mdOpen === p.slug) { mdOpen = null; mdNew = false; }
        data = await (await fetch('/api/data')).json();
        render();
        $('msg').className = 'ok'; $('msg').textContent = '✓ Deleted and rebuilt';
      } catch (e) {
        msg.className = 'err'; msg.textContent = 'Delete failed: ' + e.message;
      }
    }}, 'Delete');
    ops.append(rep, editBtn, repBtn, delBtn);
    row.append(ops);
    if (mdOpen === p.slug && !mdNew) row.append(mdEditor(p.slug));
    list.append(row);
  });
  if (!data.posts.length)
    list.append(el('div', {class:'row', style:'color:var(--muted)'}, 'No posts yet. Upload a zip to get started.'));
  $('addBtn').hidden = true;
}

function render() {
  const schema = SCHEMAS[cur];
  $('hint').textContent = schema.hint;
  document.querySelectorAll('.tabs button').forEach(b =>
    b.classList.toggle('on', b.dataset.name === cur));
  schema.site ? renderSite() : schema.blog ? renderBlog() : renderList();
  $('msg').textContent = '';
}

async function save() {
  const msg = $('msg');
  msg.className = ''; msg.textContent = 'Saving…';
  try {
    const res = await fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name: cur, data: data[cur]})});
    const out = await res.json();
    if (!res.ok) throw new Error(out.error || res.status);
    msg.className = 'ok'; msg.textContent = '✓ Saved and rebuilt; refresh the preview to see the changes';
  } catch (e) {
    msg.className = 'err'; msg.textContent = 'Save failed: ' + e.message;
  }
}

async function init() {
  data = await (await fetch('/api/data')).json();
  const tabs = $('tabs');
  for (const name of ORDER) {
    const b = el('button', {onclick: () => { cur = name; render(); }}, SCHEMAS[name].title);
    b.dataset.name = name;
    tabs.append(b);
  }
  $('addBtn').onclick = () => { data[cur].unshift({...SCHEMAS[cur].blank}); renderList(); };
  $('saveBtn').onclick = save;
  $('pubBtn').onclick = async () => {
    const m = prompt('Publish to GitHub (updates the live site). Commit message (leave blank for default): ', '');
    if (m === null) return;
    const msg = $('msg'), btn = $('pubBtn');
    btn.disabled = true; btn.textContent = 'Publishing…';
    msg.className = ''; msg.textContent = 'Committing and pushing…';
    try {
      const out = await postJSON('/api/publish', {message: m});
      msg.className = 'ok'; msg.textContent = '✓ ' + out.detail;
    } catch (e) {
      msg.className = 'err'; msg.textContent = 'Publish failed: ' + e.message;
    } finally {
      btn.disabled = false; btn.textContent = 'Publish to GitHub';
    }
  };
  render();
}
init();
</script>
</body>
</html>
"""


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler) as srv:
        print(f"admin:   http://localhost:{port}/admin")
        print(f"preview: http://localhost:{port}/")
        print("Ctrl+C to quit")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
