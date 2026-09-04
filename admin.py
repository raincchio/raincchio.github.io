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
        # 预览时禁止缓存，保证保存后刷新即见
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
                importlib.reload(build).main()  # 热加载，改 build.py 后无需重启
                self._send(200, b'{"ok":true}', "application/json")
            elif self.path == "/api/avatar":
                self.save_avatar(req)
            elif self.path == "/api/blog/upload":
                self.blog_upload(req)
            elif self.path == "/api/blog/md":
                self.blog_save_md(req)
            elif self.path == "/api/blog/delete":
                self.blog_delete(req)
            else:
                self._send(404, b'{"error":"not found"}', "application/json")
        except Exception as e:  # 把错误报给前端而不是让请求挂起
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
            return self._send(400, json.dumps({"error": "压缩包里没有 .md 文件"}).encode("utf-8"),
                              "application/json")
        # 优先根目录、名为 index/readme/post 的 md；其余文件相对它的目录解压
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
            if not rel.parts or ".." in rel.parts:  # zip-slip 防护
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
        if replaced:  # 替换正文时保留后台里改过的标题/日期
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
        if not any(p.get("slug") == slug for p in posts):  # 新建：从 md 解析标题/日期
            meta, body = b.parse_front_matter(content)
            m = re.match(r"\s*#\s+(.+)\n", body)
            title = meta.get("title") or (m.group(1).strip() if m else slug)
            date = meta.get("date") or datetime.date.today().strftime("%Y-%m-%d")
            posts.insert(0, {"slug": slug, "title": title, "date": date})
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

    def log_message(self, fmt, *args):
        pass  # 安静一点


ADMIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>主页后台</title>
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
  <h1>主页后台</h1>
  <a href="/" target="_blank">预览站点 ↗</a>
</header>
<div class="tabs" id="tabs"></div>
<main>
  <p class="hint" id="hint"></p>
  <div id="list"></div>
  <div class="bar">
    <button class="add" id="addBtn">＋ 添加一条</button>
    <button class="save" id="saveBtn">保存并重建</button>
    <span id="msg"></span>
  </div>
</main>
<script>
const SCHEMAS = {
  updates: { title:'动态', hint:'主页动态会自动从论文、博客、工作经历按时间聚合出最新 3 条，这里只用于添加额外的手动动态（如获奖、报告）。日期写 2026 或 2026.03 均可。',
    fields:[
      {k:'date', label:'日期', short:true},
      {k:'text', label:'内容（纯文本或 HTML，可写 <a href="/publications/">链接</a>）', type:'textarea', rows:2}],
    blank:{date:'', text:''} },
  publications: { title:'论文', hint:'按年份自动分组、倒序排列；作者里出现「X. Chen」会自动加粗。新论文也会自动进入主页最新动态。',
    fields:[
      {k:'year', label:'年份（如 2026.03）', short:true},
      {k:'title', label:'标题'},
      {k:'authors', label:'作者（缩写、逗号分隔）'},
      {k:'venue', label:'发表于（会议/期刊 + 年份，或 arXiv 编号）'}],
    blank:{year:'', title:'', authors:'', venue:''} },
  experience: { title:'工作经历', hint:'按列表顺序展示，请把最新的放最上面。描述可含 HTML 链接。',
    fields:[
      {k:'date', label:'时间段（如 2024.8 – Present）', short:true},
      {k:'title', label:'职位'},
      {k:'org', label:'公司 / 机构'},
      {k:'desc', label:'一句话描述（可含 <a href="…">链接</a>）', type:'textarea', rows:2}],
    blank:{date:'', title:'', org:'', desc:''} },
  education: { title:'学习经历', hint:'按列表顺序展示，请把最新的放最上面。描述可含 HTML 链接（如导师主页）。',
    fields:[
      {k:'date', label:'时间段', short:true},
      {k:'title', label:'学位 / 专业'},
      {k:'org', label:'学校'},
      {k:'desc', label:'一句话描述（可含 <a href="…">链接</a>）', type:'textarea', rows:2}],
    blank:{date:'', title:'', org:'', desc:''} },
  posts: { title:'博客', hint:'每篇博客独立管理：上传 zip（Markdown + 图片）或直接「编辑 Markdown」在线改正文。标题/日期从 front matter（--- title: … / date: … ---）或第一个 # 标题解析，也可在下方修改后点「保存并重建」。', blog:true },
  site: { title:'站点信息', hint:'姓名、签名、格言与联系方式链接。', site:true },
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
      el('button', {class:'del', onclick: () => { arr.splice(i,1); renderList(); }}, '删除'));
    row.append(ops);
    list.append(row);
  });
  $('addBtn').hidden = false;
}

function renderSite() {
  const list = $('list'), s = data.site;
  list.replaceChildren();

  const avatarRow = el('div', {class:'row'});
  avatarRow.append(el('label', {}, '头像（主页显示为圆形，建议正方形图，jpg/png/webp，≤5MB）'));
  const line = el('div', {style:'display:flex;gap:1rem;align-items:center'});
  if (s.avatar) {
    const img = el('img', {src: s.avatar + '?t=' + Date.now(),
      style:'width:72px;height:72px;border-radius:50%;object-fit:cover;border:1px solid var(--border)'});
    line.append(img);
  } else {
    line.append(el('span', {style:'color:var(--muted);font-size:0.9rem'}, '尚未上传'));
  }
  const file = el('input', {type:'file', accept:'image/*', style:'border:none;padding:0'});
  file.onchange = async () => {
    const f = file.files[0];
    if (!f) return;
    const msg = $('msg');
    msg.className = ''; msg.textContent = '上传中…';
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
      msg.className = 'ok'; msg.textContent = '✓ 头像已更新并重建';
      renderSite();
    } catch (e) {
      msg.className = 'err'; msg.textContent = '上传失败：' + e.message;
    }
  };
  line.append(file);
  avatarRow.append(line);
  list.append(avatarRow);

  const row = el('div', {class:'row'});
  const scalars = [
    {k:'name', label:'姓名'},
    {k:'tagline', label:'一句话签名（首页标题下方）'},
    {k:'motto', label:'格言（首页引用）', type:'textarea', rows:2},
    {k:'bio', label:'个人简介（首页 About，可含 HTML 链接）', type:'textarea', rows:4},
    {k:'highlight', label:'论文作者中要加粗的名字'},
    {k:'blog_intro', label:'博客页简介'},
    {k:'source_url', label:'页脚 Source 链接'},
  ];
  for (const f of scalars) row.append(field(s, f));
  list.append(row);

  const linksRow = el('div', {class:'row'});
  linksRow.append(el('label', {}, '联系方式 / 外部链接'));
  s.links.forEach((l, i) => {
    const line = el('div', {style:'display:flex;gap:0.5rem;margin-bottom:0.4rem;align-items:center'});
    const lab = el('input', {class:'short', oninput: e => l.label = e.target.value}); lab.value = l.label || '';
    const url = el('input', {oninput: e => l.url = e.target.value}); url.value = l.url || '';
    const del = el('button', {class:'del', onclick: () => { s.links.splice(i,1); renderSite(); },
                              style:'border:1px solid var(--border);background:#fff;border-radius:6px;cursor:pointer'}, '删');
    line.append(lab, url, del);
    linksRow.append(line);
  });
  linksRow.append(el('button', {class:'add', style:'border:1px solid var(--border);background:#fff;border-radius:6px;padding:0.2rem 0.7rem;cursor:pointer',
    onclick: () => { s.links.push({label:'', url:''}); renderSite(); }}, '＋ 加一条链接'));
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
  msg.className = ''; msg.textContent = '上传解析中…';
  try {
    const out = await postJSON('/api/blog/upload',
      {filename: f.name, slug: slug || '', data: await readB64(f)});
    data = await (await fetch('/api/data')).json();
    render();
    msg.className = 'ok';
    msg.textContent = (out.replaced ? '✓ 已替换并重建：' : '✓ 已发布并重建：') +
      out.slug + '（' + out.title + '）';
  } catch (e) {
    msg.className = 'err'; msg.textContent = '上传失败：' + e.message;
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
    el('button', {onclick: () => saveMd(slug)}, '保存 Markdown 并重建'),
    el('button', {onclick: () => { mdOpen = null; mdNew = false; render(); }}, '收起'));
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
    msg.className = 'err'; msg.textContent = '读取失败：' + e.message;
  }
}

async function saveMd(slug) {
  const msg = $('msg');
  msg.className = ''; msg.textContent = '保存中…';
  try {
    await postJSON('/api/blog/md', {slug, content: mdDraft});
    data = await (await fetch('/api/data')).json();
    mdNew = false;
    render();
    $('msg').className = 'ok'; $('msg').textContent = '✓ 已保存并重建：' + slug + '，刷新预览即可看到';
  } catch (e) {
    msg.className = 'err'; msg.textContent = '保存失败：' + e.message;
  }
}

function renderBlog() {
  const list = $('list');
  list.replaceChildren();

  const up = el('div', {class:'row'});
  up.append(el('label', {}, '上传新博客（zip）'));
  const line = el('div', {style:'display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap'});
  const slugIn = el('input', {class:'short', placeholder:'slug（可留空，取 zip 文件名）'});
  const file = el('input', {type:'file', accept:'.zip,application/zip', style:'border:none;padding:0'});
  file.onchange = () => uploadZip(file, slugIn.value);
  const newBtn = el('button', {style:'border:1px solid var(--border);background:#fff;border-radius:6px;padding:0.3rem 0.8rem;cursor:pointer',
    onclick: () => {
      const slug = slugIn.value.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-+|-+$/g,'');
      if (!slug) { $('msg').className='err'; $('msg').textContent='新建前先在左边填一个 slug'; return; }
      mdNew = true; mdOpen = slug;
      mdDraft = '---\ntitle: 文章标题\ndate: ' + new Date().toISOString().slice(0,10) + '\n---\n\n正文……\n';
      render();
    }}, '＋ 新建（纯 Markdown）');
  line.append(slugIn, file, newBtn);
  up.append(line);
  up.append(el('div', {class:'hint', style:'margin:0.5rem 0 0'},
    'zip 里放一个 .md（可带 front matter）和它引用的图片，图片用相对路径（如 ![](fig1.png)）。' +
    '公式写 $行内$ 或 $$独立公式$$（KaTeX 渲染）。slug 相同即覆盖旧文。不带图的文章可直接「新建」在线写。'));
  if (mdNew && mdOpen) {
    up.append(el('div', {style:'margin-top:0.6rem;font-weight:600'}, '新博客：' + mdOpen));
    up.append(mdEditor(mdOpen));
  }
  list.append(up);

  data.posts.forEach(p => {
    const row = el('div', {class:'row'});
    const head = el('div', {style:'display:flex;gap:0.7rem;align-items:baseline'});
    head.append(el('strong', {}, p.slug),
      el('a', {href:'/blog/' + p.slug + '/', target:'_blank', style:'font-size:0.85rem'}, '预览 ↗'));
    row.append(head);
    row.append(field(p, {k:'title', label:'标题'}));
    row.append(field(p, {k:'date', label:'日期（如 2026-09-04）', short:true}));
    const ops = el('div', {class:'ops'});
    const rep = el('input', {type:'file', accept:'.zip,application/zip', style:'display:none'});
    rep.onchange = () => uploadZip(rep, p.slug);
    const editBtn = el('button', {onclick: () => openMd(p.slug)},
      mdOpen === p.slug && !mdNew ? '收起编辑' : '编辑 Markdown');
    const repBtn = el('button', {onclick: () => rep.click()}, '替换 zip');
    const delBtn = el('button', {class:'del', onclick: async () => {
      if (!confirm('删除博客「' + p.slug + '」？源文件和页面都会被移除。')) return;
      const msg = $('msg');
      try {
        await postJSON('/api/blog/delete', {slug: p.slug});
        if (mdOpen === p.slug) { mdOpen = null; mdNew = false; }
        data = await (await fetch('/api/data')).json();
        render();
        $('msg').className = 'ok'; $('msg').textContent = '✓ 已删除并重建';
      } catch (e) {
        msg.className = 'err'; msg.textContent = '删除失败：' + e.message;
      }
    }}, '删除');
    ops.append(rep, editBtn, repBtn, delBtn);
    row.append(ops);
    if (mdOpen === p.slug && !mdNew) row.append(mdEditor(p.slug));
    list.append(row);
  });
  if (!data.posts.length)
    list.append(el('div', {class:'row', style:'color:var(--muted)'}, '还没有博客，上传一个 zip 开始吧。'));
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
  msg.className = ''; msg.textContent = '保存中…';
  try {
    const res = await fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name: cur, data: data[cur]})});
    const out = await res.json();
    if (!res.ok) throw new Error(out.error || res.status);
    msg.className = 'ok'; msg.textContent = '✓ 已保存并重建，刷新预览即可看到';
  } catch (e) {
    msg.className = 'err'; msg.textContent = '保存失败：' + e.message;
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
        print(f"后台: http://localhost:{port}/admin")
        print(f"预览: http://localhost:{port}/")
        print("Ctrl+C 退出")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
