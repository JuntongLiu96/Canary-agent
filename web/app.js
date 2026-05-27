const SID = Math.random().toString(36).slice(2, 10);
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const modelSel = document.getElementById('model');
const status = document.getElementById('status');
const slash = document.getElementById('slash');

let currentAssistant = null;
let toolBlocks = {}; // id -> details element

function addMsg(role, text='') {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function addTool(id, name) {
  const det = document.createElement('details');
  det.className = 'tool';
  det.innerHTML = `<summary>🔧 <b>${name}</b> <code>${id.slice(-6)}</code></summary><pre class="args"></pre><pre class="result" style="border-top:1px solid #5555; margin-top:6px; padding-top:6px;"></pre>`;
  log.appendChild(det);
  toolBlocks[id] = det;
  log.scrollTop = log.scrollHeight;
  return det;
}

async function loadModels() {
  try {
    const r = await fetch('/api/models');
    const j = await r.json();
    const models = (j.data || []).filter(m => !/(embedding|tts|whisper)/i.test(m.id));
    for (const m of models) {
      const o = document.createElement('option');
      o.value = m.id; o.textContent = m.id;
      modelSel.appendChild(o);
    }
    const def = models.find(m => m.id.startsWith('claude-sonnet-4')) || models[0];
    if (def) modelSel.value = def.id;
  } catch (e) { status.textContent = 'no copilot-api'; }
}

let prompts = [];
async function loadPrompts() {
  try {
    const r = await fetch('/api/prompts');
    prompts = (await r.json()).prompts || [];
  } catch (e) { prompts = []; }
}

function openStream() {
  const es = new EventSource('/api/stream/' + SID);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    handleEvent(ev);
  };
  es.onerror = () => { status.textContent = 'reconnecting…'; };
}

function handleEvent(ev) {
  switch (ev.type) {
    case 'text':
      if (!currentAssistant) currentAssistant = addMsg('assistant', '');
      currentAssistant.textContent += ev.text;
      log.scrollTop = log.scrollHeight;
      break;
    case 'tool_use_start':
      currentAssistant = null;
      addTool(ev.id, ev.name);
      break;
    case 'tool_use_delta':
      if (toolBlocks[ev.id]) toolBlocks[ev.id].querySelector('.args').textContent += ev.partial_json;
      break;
    case 'tool_use_stop': break;
    case 'tool_result':
      if (toolBlocks[ev.id]) {
        const r = toolBlocks[ev.id].querySelector('.result');
        r.textContent = ev.content;
        if (ev.is_error) r.style.color = '#f88';
      }
      break;
    case 'message_stop': currentAssistant = null; break;
    case 'error':
      addMsg('assistant', '⚠️ ' + ev.message);
      currentAssistant = null;
      break;
    case 'done':
      currentAssistant = null;
      status.textContent = '';
      break;
  }
}

form.onsubmit = async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  slash.style.display = 'none';

  // Slash command: /server:prompt or /prompt
  if (text.startsWith('/')) {
    const [head, ...rest] = text.slice(1).split(/\s+/);
    const [server, name] = head.includes(':') ? head.split(':') : [null, head];
    const p = prompts.find(x => x.name === name && (!server || x.server === server));
    if (p) {
      addMsg('user', text);
      const args = {};
      for (const a of (p.arguments || [])) {
        const v = prompt(`Argument "${a.name}"${a.required ? ' (required)' : ''}:`, '');
        if (v != null) args[a.name] = v;
      }
      const r = await fetch('/api/prompts/get', {
        method: 'POST', headers: {'content-type':'application/json'},
        body: JSON.stringify({server: p.server, name: p.name, arguments: args}),
      });
      const j = await r.json();
      await fetch('/api/chat', {
        method: 'POST', headers: {'content-type':'application/json'},
        body: JSON.stringify({session_id: SID, blocks: j.blocks, model: modelSel.value}),
      });
      status.textContent = '…';
      return;
    }
  }

  addMsg('user', text);
  status.textContent = '…';
  await fetch('/api/chat', {
    method: 'POST', headers: {'content-type':'application/json'},
    body: JSON.stringify({session_id: SID, message: text, model: modelSel.value}),
  });
};

input.addEventListener('input', () => {
  const v = input.value;
  if (!v.startsWith('/') || v.includes(' ')) { slash.style.display = 'none'; return; }
  const q = v.slice(1).toLowerCase();
  const matches = prompts.filter(p => p.name.toLowerCase().includes(q) || (p.server||'').toLowerCase().includes(q));
  slash.innerHTML = '';
  for (const p of matches.slice(0, 8)) {
    const d = document.createElement('div');
    d.textContent = `/${p.server}:${p.name}  — ${p.description || ''}`;
    d.onclick = () => { input.value = `/${p.server}:${p.name} `; slash.style.display = 'none'; input.focus(); };
    slash.appendChild(d);
  }
  slash.style.display = matches.length ? 'block' : 'none';
});

document.getElementById('reset').onclick = async () => {
  await fetch('/api/reset/' + SID, {method:'POST'});
  log.innerHTML = '';
  toolBlocks = {}; currentAssistant = null;
};
document.getElementById('refresh-mcp').onclick = async () => {
  status.textContent = 'refreshing MCP…';
  await fetch('/api/mcp/refresh', {method:'POST'});
  await loadPrompts();
  status.textContent = '';
};

// ─────── Manage drawer ───────
const drawer = document.getElementById('drawer');
const mKind = document.getElementById('m-kind');
const mList = document.getElementById('m-list');
const mWrap = document.getElementById('m-editor-wrap');
const mName = document.getElementById('m-name');
const mText = document.getElementById('m-text');
const mStatus = document.getElementById('m-status');
let mCurrent = null;  // name being edited, null = new

document.getElementById('manage').onclick = () => { drawer.style.display = 'block'; refreshManage(); };
document.getElementById('m-close').onclick = () => { drawer.style.display = 'none'; };
mKind.onchange = () => refreshManage();

async function refreshManage() {
  mStatus.textContent = '';
  mWrap.style.display = 'none';
  mCurrent = null;
  const kind = mKind.value;
  if (kind === 'mcp') {
    const r = await fetch('/api/manage/mcp');
    const j = await r.json();
    mList.innerHTML = '<div style="opacity:.6; margin-bottom:4px;">Single JSON file — edit and save. Restart server to apply.</div>';
    mWrap.style.display = 'block';
    mName.style.display = 'none';
    mText.value = j.text;
    mCurrent = '__mcp__';
  } else {
    mName.style.display = 'inline-block';
    const r = await fetch(`/api/manage/${kind}`);
    const j = await r.json();
    mList.innerHTML = '';
    for (const it of j.items) {
      const b = document.createElement('button');
      b.textContent = it.name;
      b.style.margin = '2px';
      b.onclick = () => openItem(kind, it.name);
      mList.appendChild(b);
    }
    const nb = document.createElement('button');
    nb.textContent = '+ new';
    nb.style.margin = '2px'; nb.style.background = '#264';
    nb.onclick = () => { mCurrent = null; mName.value = ''; mText.value = '---\nname: \ndescription: \n---\n'; mWrap.style.display = 'block'; mName.focus(); };
    mList.appendChild(nb);
  }
}

async function openItem(kind, name) {
  const r = await fetch(`/api/manage/${kind}/${encodeURIComponent(name)}`);
  const j = await r.json();
  mCurrent = name;
  mName.value = name;
  mText.value = j.text;
  mWrap.style.display = 'block';
}

document.getElementById('m-save').onclick = async () => {
  const kind = mKind.value;
  const name = mCurrent === '__mcp__' ? null : (mName.value.trim() || mCurrent);
  if (kind !== 'mcp' && !name) { mStatus.textContent = 'name required'; return; }
  const url = kind === 'mcp' ? '/api/manage/mcp' : `/api/manage/${kind}/${encodeURIComponent(name)}`;
  const r = await fetch(url, {
    method: 'PUT', headers: {'content-type':'application/json'},
    body: JSON.stringify({text: mText.value}),
  });
  const j = await r.json();
  if (!r.ok) { mStatus.textContent = j.detail || 'error'; return; }
  mStatus.textContent = j.note || 'saved';
  if (kind !== 'mcp') { mCurrent = name; await refreshManage(); }
};

document.getElementById('m-delete').onclick = async () => {
  const kind = mKind.value;
  if (kind === 'mcp' || !mCurrent) return;
  if (!confirm(`Delete ${mCurrent}?`)) return;
  await fetch(`/api/manage/${kind}/${encodeURIComponent(mCurrent)}`, {method: 'DELETE'});
  await refreshManage();
};

(async () => {
  await loadModels();
  await loadPrompts();
  openStream();
})();
