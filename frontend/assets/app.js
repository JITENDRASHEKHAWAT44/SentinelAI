/**
 * Sentinel AI — Application Controller
 * PDF upload → vector storage → conversational document QA → RAGAS evaluation
 */

const state = {
  view: 'chat',
  conversations: [],
  activeConversationId: null,
  activeDocument: { name: null, chunks: 0 },
  goldens: [],
  evalReport: null,
  isSending: false,
  isEvaluating: false,
};

const $ = (id) => document.getElementById(id);

/* ── Utilities ── */

function cleanDocName(name) {
  if (!name) return 'No document loaded';
  return String(name).replace(/^[a-f0-9]{32}_/i, '');
}

function esc(text) {
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function md(text) {
  if (!text) return '';
  let h = esc(text);
  h = h.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.*?)\*/g, '<em>$1</em>');
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  h = h.replace(/\[(\d+)\]/g, '<span class="citation-badge" data-cite="$1" title="View source #$1">[$1]</span>');
  h = h.replace(/\n/g, '<br>');
  return h;
}

function fmtBytes(b) {
  if (!b) return '0 B';
  const k = 1024, s = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(b) / Math.log(k));
  return parseFloat((b / Math.pow(k, i)).toFixed(1)) + ' ' + s[i];
}

function fmtScore(v) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return (v * 100).toFixed(1) + '%';
}

function scoreClass(v) {
  if (v === null || v === undefined || isNaN(v)) return '';
  if (v >= 0.8) return 'score-high';
  if (v >= 0.5) return 'score-mid';
  return 'score-low';
}

const API_BASE = (window.SENTINEL_API_URL || localStorage.getItem('sentinel_api_url') || '').replace(/\/+$/, '');

async function api(path, opts = {}) {
  const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

/* ── Init ── */

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initUpload();
  initComposer();
  initSourcesPanel();
  initEvaluation();
  initDropOverlay();

  loadHealth();
  loadInitialQuestions();
  loadConversations();
  loadEvaluationReport();
  loadGoldens();
});

/* ── Navigation ── */

function initNavigation() {
  document.querySelectorAll('.tab-btn').forEach(tab => {
    tab.addEventListener('click', () => {
      const view = tab.dataset.view;
      state.view = view;
      document.querySelectorAll('.tab-btn').forEach(t => t.classList.toggle('is-active', t.dataset.view === view));
      document.querySelectorAll('.workspace-view').forEach(v => { v.hidden = v.id !== `view-${view}`; });
      if (view === 'evaluation') { loadEvaluationReport(); loadGoldens(); }
    });
  });
}

/* ── Health ── */

async function loadHealth() {
  try {
    const d = await api('/api/database/stats');
    const n = d.total_chunks || 0;
    const badge = $('db-status-badge');
    if (badge) badge.textContent = `Connected · ${n} chunks`;
    if (d.sources && d.sources.length > 0) {
      state.activeDocument = { name: d.sources[0].filename, chunks: d.sources[0].chunks };
      updateDocBar();
    }
  } catch (e) { console.error('Health check failed:', e); }
}

function updateDocBar() {
  const el = $('active-doc-name');
  const cleanName = cleanDocName(state.activeDocument.name);
  if (el) {
    if (state.activeDocument.name) {
      el.textContent = cleanName;
      el.title = `${cleanName} · ${state.activeDocument.chunks || 0} chunks`;
    } else {
      el.textContent = 'No document loaded';
      el.title = 'Upload a PDF to get started';
    }
  }
  const subtext = $('welcome-subtext');
  if (subtext) {
    if (state.activeDocument.name) {
      subtext.innerHTML = `
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <span>Ask any question about <strong>${esc(cleanName)}</strong> below or pick a question from the top bar.</span>
      `;
    } else {
      subtext.innerHTML = `
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <span>Upload a PDF to get started, or ask a question below.</span>
      `;
    }
  }
}

/* ── Upload ── */

function initUpload() {
  const btn = $('sidebar-upload-trigger');
  const input = $('sidebar-file-picker');
  const feedback = $('upload-feedback-bar');

  btn?.addEventListener('click', () => input?.click());
  input?.addEventListener('change', async () => {
    if (!input.files?.length) return;
    await uploadFile(input.files[0], feedback);
    input.value = '';
  });
}

async function uploadFile(file, statusEl) {
  if (!statusEl) return;
  statusEl.hidden = false;
  statusEl.className = 'upload-feedback loading';
  statusEl.textContent = `Uploading and indexing ${file.name}…`;

  const fd = new FormData();
  fd.append('files', file);

  try {
    const res = await api('/api/files', { method: 'POST', body: fd });
    if (res.errors?.length && (!res.ingested || res.ingested.length === 0)) {
      throw new Error(res.errors[0].error || 'Indexing failed');
    }
    const doc = res.ingested?.[0];
    if (doc) {
      state.activeDocument = { name: doc.filename, chunks: doc.chunks };
      updateDocBar();
    }
    statusEl.className = 'upload-feedback success';
    statusEl.textContent = `${file.name} indexed successfully (${doc?.chunks || 0} chunks) — loading questions…`;
    await loadHealth();
    await loadInitialQuestions();
    setTimeout(() => { statusEl.hidden = true; }, 4000);
  } catch (e) {
    statusEl.className = 'upload-feedback error';
    statusEl.textContent = `Upload failed: ${e.message}`;
  }
}

/* ── Drop Overlay ── */

function initDropOverlay() {
  const overlay = $('drop-overlay');
  let dragCounter = 0;

  document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    if (overlay) overlay.hidden = false;
  });

  document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) { dragCounter = 0; if (overlay) overlay.hidden = true; }
  });

  document.addEventListener('dragover', (e) => e.preventDefault());

  document.addEventListener('drop', (e) => {
    e.preventDefault();
    dragCounter = 0;
    if (overlay) overlay.hidden = true;
    if (e.dataTransfer.files?.length) {
      // Switch to chat view
      document.querySelector('[data-view="chat"]')?.click();
      uploadFile(e.dataTransfer.files[0], $('upload-feedback-bar'));
    }
  });
}

/* ── Conversations ── */

async function loadConversations() {
  try {
    const d = await api('/api/conversations');
    state.conversations = d.conversations || [];
    if (!state.activeConversationId && state.conversations.length > 0) {
      selectConversation(state.conversations[0].id);
    } else {
      renderMessages();
    }
  } catch (e) { console.error('Conversations load failed:', e); }
}

function selectConversation(id) {
  state.activeConversationId = id;
  renderMessages();
}

function currentConversation() {
  return state.conversations.find(c => c.id === state.activeConversationId);
}

/* ── Messages ── */

function renderMessages() {
  const conv = currentConversation();
  const welcome = $('chat-welcome-state');
  const feed = $('chat-message-feed');
  const endBtn = $('btn-end-chat');

  if (!conv || !conv.messages?.length) {
    if (welcome) {
      welcome.hidden = false;
      welcome.style.display = 'flex';
    }
    if (feed) {
      feed.hidden = true;
      feed.style.display = 'none';
      feed.innerHTML = '';
    }
    if (endBtn) endBtn.style.display = 'none';
    renderSources(null);
    return;
  }

  const lastMsg = conv.messages[conv.messages.length - 1];
  const lastAsst = [...conv.messages].reverse().find(m => m.role === 'assistant');
  const hasExit = Boolean(lastMsg?.is_exit || (lastAsst?.is_exit && lastMsg?.role === 'assistant'));

  if (welcome) {
    welcome.hidden = true;
    welcome.style.display = 'none';
  }
  if (feed) {
    feed.hidden = false;
    feed.style.display = 'flex';
  }
  if (endBtn) endBtn.style.display = hasExit ? 'none' : 'inline-flex';

  let feedHtml = conv.messages.map(msg => {
    if (msg.role === 'user') {
      return `
        <div class="message-row user" data-msg-id="${msg.id}">
          <div class="avatar user-avatar" title="You">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
          </div>
          <div class="msg-body">
            <div class="msg-bubble">${esc(msg.content)}</div>
            <div class="msg-meta">${fmtTime(msg.created_at)}</div>
          </div>
        </div>`;
    }
    const sources = msg.sources || [];
    return `
      <div class="message-row assistant" data-msg-id="${msg.id}">
        <div class="avatar ai-avatar" title="Sentinel AI">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        </div>
        <div class="msg-body">
          <div class="msg-bubble">
            ${msg.thinking ? `
              <details class="thinking-accordion">
                <summary class="thinking-trigger">
                  <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                  Reasoning
                </summary>
                <div class="thinking-body">${esc(msg.thinking)}</div>
              </details>` : ''}
            <div>${md(msg.content)}</div>
            ${sources.length ? `
              <div class="message-sources-row">
                ${sources.map(s => `
                  <button class="source-pill-btn" data-source-idx="${s.index}">
                    <strong>[${s.index}]</strong> ${esc(s.filename)} ${s.page ? '· p.' + s.page : ''}
                  </button>`).join('')}
              </div>` : ''}
          </div>
          <div class="msg-actions">
            <button class="btn-ghost-sm copy-btn" data-text="${esc(msg.content)}">Copy</button>
          </div>
          <div class="msg-meta">${fmtTime(msg.created_at)}</div>
        </div>
      </div>`;
  }).join('');

  if (hasExit) {
    feedHtml += `
      <div class="chat-exit-banner">
        <span class="chat-exit-line-text">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
          Conversation ended
        </span>
        <span class="chat-exit-hint">You can start a fresh conversation or purge this session's uploaded PDF data.</span>
        <div style="display:flex;gap:8px;align-items:center;margin-top:6px;">
          <button type="button" class="btn-ghost-sm" id="btn-restart-chat">+ Start New Chat</button>
          <button type="button" class="btn-ghost-sm" id="btn-exit-clear" style="color:var(--danger);border-color:rgba(197,48,48,0.25);">Wipe PDF & Data</button>
        </div>
      </div>`;
  }

  feed.innerHTML = feedHtml;
  feed.scrollTop = feed.scrollHeight;

  // Bind citation + source pill clicks
  feed.querySelectorAll('.citation-badge, .source-pill-btn').forEach(el => {
    const row = el.closest('.message-row');
    const msgId = row?.dataset?.msgId;
    el.addEventListener('click', () => highlightSource(el.dataset.cite || el.dataset.sourceIdx, msgId));
  });

  // Copy buttons
  feed.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(btn.dataset.text);
      btn.textContent = 'Copied';
      setTimeout(() => btn.textContent = 'Copy', 1500);
    });
  });

  // Restart chat button
  $('btn-restart-chat')?.addEventListener('click', startNewConversation);
  $('btn-exit-clear')?.addEventListener('click', clearSession);

  // Update sources panel with last assistant message
  renderSources(lastAsst);
}

function fmtTime(iso) {
  if (!iso) return '';
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/* ── Composer & Conversation Control ── */

function initComposer() {
  const form = $('chat-composer-form');
  const input = $('chat-query-input');

  input?.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  });

  input?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form?.requestSubmit(); }
  });

  // End Chat & Clear Session actions in doc bar
  $('btn-end-chat')?.addEventListener('click', endCurrentConversation);
  $('btn-clear-session')?.addEventListener('click', clearSession);

  form?.addEventListener('submit', (e) => {
    e.preventDefault();
    submitQuery(input?.value);
  });
}

async function submitQuery(rawQuery) {
  const q = (rawQuery || '').trim();
  if (!q || state.isSending) return;

  const input = $('chat-query-input');
  const btn = $('btn-submit-query');

  state.isSending = true;
  if (btn) btn.disabled = true;
  if (input) {
    input.value = '';
    input.style.height = 'auto';
  }

  let conv = currentConversation();
  if (!conv) {
    const nc = await api('/api/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: q.slice(0, 50) }),
    });
    nc.messages = [];
    state.conversations = state.conversations || [];
    state.conversations.unshift(nc);
    state.activeConversationId = nc.id;
    conv = nc;
  }

  conv.messages = conv.messages || [];
  conv.messages.push({
    id: 'tmp-' + Date.now(),
    role: 'user',
    content: q,
    created_at: new Date().toISOString()
  });

  // 1. Immediately render the question in the chat feed (ChatGPT style)
  renderMessages();

  const isExit = ['exit', 'quit', 'stop', 'bye', '/exit', 'exit.', 'end', 'end chat'].includes(q.toLowerCase());
  const hint = $('sources-query-hint');
  if (hint) hint.textContent = isExit ? 'Ending conversation…' : 'Retrieving context…';

  // 2. Immediately render the assistant response animation with pulsing AI logo
  const feed = $('chat-message-feed');
  if (feed && !isExit) {
    const typingEl = document.createElement('div');
    typingEl.id = 'chat-typing-indicator';
    typingEl.className = 'message-row assistant typing-row';
    typingEl.innerHTML = `
      <div class="avatar ai-avatar ai-avatar-pulse" title="Sentinel AI analyzing document">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
      </div>
      <div class="msg-body">
        <div class="msg-bubble typing-bubble">
          <span class="typing-status-text">Thinking & searching document</span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      </div>`;
    feed.appendChild(typingEl);
    feed.scrollTop = feed.scrollHeight;
  }

  try {
    const k = parseInt($('input-k-chunks')?.value) || 5;
    await api('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, conversation_id: state.activeConversationId, k, sanitize: true }),
    });
    await loadConversations();
  } catch (err) {
    document.getElementById('chat-typing-indicator')?.remove();
    conv.messages.pop();
    renderMessages();
    alert('Query failed: ' + err.message);
  } finally {
    document.getElementById('chat-typing-indicator')?.remove();
    state.isSending = false;
    if (btn) btn.disabled = false;
    input?.focus();
  }
}

async function endCurrentConversation() {
  const conv = currentConversation();
  if (!conv || !conv.messages?.length) return;
  const lastMsg = conv.messages[conv.messages.length - 1];
  const lastAsst = [...conv.messages].reverse().find(m => m.role === 'assistant');
  const isAlreadyEnded = Boolean(lastMsg?.is_exit || (lastAsst?.is_exit && lastMsg?.role === 'assistant'));
  if (isAlreadyEnded) return;

  submitQuery('exit');
}

async function startNewConversation() {
  try {
    const nc = await api('/api/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'New Conversation' }),
    });
    state.activeConversationId = nc.id;
    await loadConversations();
    const input = $('chat-query-input');
    if (input) {
      input.value = '';
      input.focus();
    }
  } catch (e) {
    console.error('Failed to start new conversation:', e);
  }
}

/* ── Initial Questions (Upper side bar & Welcome grid) ── */

async function loadInitialQuestions() {
  try {
    let questions = [];
    try {
      const res = await api('/api/generate-questions', { method: 'POST' });
      questions = res.questions || [];
    } catch {
      // Fallback
    }
    if (!questions.length) {
      const d = await api('/api/suggestions');
      questions = d.suggestions || [];
    }
    renderInitialQuestions(questions);
  } catch (e) {
    console.error('Initial questions load failed:', e);
  }
}

function renderInitialQuestions(questions) {
  const strip = $('initial-questions-chips');
  if (!strip) return;

  strip.innerHTML = '';
  if (!questions || !questions.length) {
    strip.innerHTML = '<span style="font-size:12px;color:var(--ink-tertiary);padding:0 8px;">Upload a PDF to generate tailored questions.</span>';
    return;
  }

  questions.forEach(item => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'strip-chip';
    btn.innerHTML = `<span style="color:var(--accent);font-weight:700;">?</span> <span>${esc(item.question)}</span>`;
    btn.title = 'Ask this question';
    btn.addEventListener('click', () => {
      submitQuery(item.question);
    });
    strip.appendChild(btn);
  });
}

async function clearSession() {
  const confirmed = confirm('Clear session and wipe all document data? All uploaded PDFs, vector embeddings, and conversation history will be permanently deleted so no data is retained for anyone else.');
  if (!confirmed) return;

  try {
    await api('/api/session/clear', { method: 'POST' });
    state.activeDocument = { name: null, chunks: 0 };
    state.conversations = [];
    state.activeConversationId = null;

    updateDocBar();
    renderMessages();
    renderSources(null);
    renderInitialQuestions([]);

    const badge = $('db-status-badge');
    if (badge) badge.textContent = 'Connected · 0 chunks';

    const feedback = $('upload-feedback-bar');
    if (feedback) {
      feedback.hidden = false;
      feedback.className = 'upload-feedback success';
      feedback.textContent = 'Session data and document embeddings permanently wiped.';
      setTimeout(() => { feedback.hidden = true; }, 3500);
    }
  } catch (err) {
    alert('Clear session failed: ' + err.message);
  }
}

/* ── Sources Panel ── */

function initSourcesPanel() {
  $('btn-toggle-sources')?.addEventListener('click', () => {
    $('sources-panel')?.classList.toggle('is-collapsed');
  });
  $('btn-close-sources')?.addEventListener('click', () => {
    $('sources-panel')?.classList.add('is-collapsed');
  });
}

function renderSources(msg) {
  const hint = $('sources-query-hint');
  const list = $('sources-cards-list');
  const badge = $('sources-count-badge');

  if (!msg || msg.role !== 'assistant') {
    if (hint) hint.textContent = 'Awaiting query';
    if (badge) badge.textContent = '0';
    if (list) list.innerHTML = `
      <div class="empty-sources">
        <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <p>Retrieved document excerpts will appear here when you ask a question.</p>
      </div>`;
    return;
  }

  const sources = msg.sources || [];
  if (badge) badge.textContent = sources.length;
  if (hint) hint.textContent = `${sources.length} sources retrieved`;

  if (!sources.length) {
    if (list) list.innerHTML = '<p style="padding:16px;font-size:12px;color:var(--ink-tertiary);text-align:center;">No sources for this response.</p>';
    return;
  }

  list.innerHTML = sources.map(s => {
    const pct = Math.round((s.relevance || 0.7) * 100);
    return `
      <div class="source-item-card" id="source-card-${s.index}">
        <div class="card-meta-top">
          <span class="source-num-tag">#${s.index}</span>
          <div class="relevance-meter-wrap">
            <div class="meter-track"><div class="meter-fill" style="width:${pct}%"></div></div>
            <span class="meter-text">${pct}%</span>
          </div>
        </div>
        <div class="source-filename-info">${esc(s.filename)} ${s.page ? '· Page ' + s.page : ''}</div>
        <div class="source-text-snippet">${esc(s.content)}</div>
        <div style="display:flex;justify-content:flex-end;margin-top:2px;">
          <button class="btn-ghost-sm copy-snippet" data-snippet="${esc(s.content)}">Copy</button>
        </div>
      </div>`;
  }).join('');

  list.querySelectorAll('.copy-snippet').forEach(btn => {
    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(btn.dataset.snippet);
      btn.textContent = 'Copied';
      setTimeout(() => btn.textContent = 'Copy', 1500);
    });
  });
}

function highlightSource(idx, msgId) {
  if (msgId) {
    const conv = currentConversation();
    const targetMsg = conv?.messages?.find(m => m.id === msgId);
    if (targetMsg && targetMsg.sources?.length) {
      renderSources(targetMsg);
    }
  }
  $('sources-panel')?.classList.remove('is-collapsed');
  const card = document.getElementById(`source-card-${idx}`);
  if (card) {
    document.querySelectorAll('.source-item-card').forEach(c => c.classList.remove('highlight'));
    card.classList.add('highlight');
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    setTimeout(() => card.classList.remove('highlight'), 2500);
  }
}

/* ── Evaluation ── */

function initEvaluation() {
  const runBtn = $('btn-run-evaluation');
  const banner = $('eval-progress-banner');

  runBtn?.addEventListener('click', async () => {
    if (state.isEvaluating) return;
    state.isEvaluating = true;
    runBtn.disabled = true;

    const count = parseInt($('eval-samples-count')?.value) || 1;
    if (banner) {
      banner.hidden = false;
      banner.className = 'eval-progress-banner running';
      banner.innerHTML = `
        <svg class="loading-spinner" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 10 10"/></svg>
        Evaluating ${count} sample(s)… this may take a few minutes.`;
    }

    try {
      const res = await api('/api/evaluation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ samples: count }),
      });
      if (banner) {
        banner.className = 'eval-progress-banner success';
        banner.textContent = 'Evaluation complete — scores updated.';
      }
      if (res?.report) {
        state.evalReport = res.report;
        renderMetricCards(res.report);
        renderSampleTable(res.report);
      } else {
        await loadEvaluationReport();
      }
    } catch (err) {
      if (banner) {
        banner.className = 'eval-progress-banner error';
        banner.textContent = 'Evaluation failed: ' + err.message;
      }
    } finally {
      state.isEvaluating = false;
      runBtn.disabled = false;
    }
  });

  $('btn-modal-close')?.addEventListener('click', closeModal);
  $('btn-modal-cancel')?.addEventListener('click', closeModal);
  $('btn-modal-save')?.addEventListener('click', saveGolden);

  // Close modal when clicking on backdrop or pressing Escape
  $('edit-golden-modal')?.addEventListener('click', (e) => {
    if (e.target === $('edit-golden-modal')) closeModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !$('edit-golden-modal')?.hidden) closeModal();
  });
}

async function loadEvaluationReport() {
  try {
    const d = await api('/api/evaluation');
    if (!d.available || !d.report) {
      $('ragas-metrics-grid').innerHTML = '<p class="empty-msg">No evaluation data yet. Click "Run Evaluation" to benchmark the RAG pipeline.</p>';
      return;
    }
    state.evalReport = d.report;
    renderMetricCards(d.report);
    renderSampleTable(d.report);
  } catch (e) { console.error('Eval load failed:', e); }
}

function renderMetricCards(report) {
  const s = report.summary || {};
  const m = s.metrics || {};
  const composite = s.composite_ragas_score;

  const metrics = [
    { key: 'composite', label: 'Composite Score', val: composite, desc: 'Overall geometric mean across all retrieval and generation metrics', hero: true },
    { key: 'faithfulness', label: 'Faithfulness', val: m.faithfulness, desc: 'Factual fidelity — answers grounded in retrieved context' },
    { key: 'answer_relevancy', label: 'Relevance', val: m.answer_relevancy, desc: 'How directly the answer addresses the question' },
    { key: 'context_precision', label: 'Precision', val: m.context_precision, desc: 'Most relevant chunks ranked first by vector search' },
    { key: 'context_recall', label: 'Recall', val: m.context_recall, desc: 'All reference facts found in retrieved context' },
    { key: 'answer_correctness', label: 'Correctness', val: m.answer_correctness, desc: 'Semantic alignment with ground-truth reference' },
  ];

  $('ragas-metrics-grid').innerHTML = metrics.map(mt => {
    const pct = mt.val != null && !isNaN(mt.val) ? Math.round(mt.val * 100) : null;
    return `
      <div class="metric-card ${mt.hero ? 'is-hero' : ''}">
        <span class="metric-label">${mt.label}</span>
        <span class="metric-value ${mt.val != null ? scoreClass(mt.val) : ''}">${pct !== null ? pct + '%' : '—'}</span>
        <div class="metric-bar"><div class="metric-bar-fill" style="width:${pct || 0}%"></div></div>
        <span class="metric-desc">${mt.desc}</span>
      </div>`;
  }).join('');
}

function renderSampleTable(report) {
  const rows = report.scorecard_per_sample || [];
  const tbody = $('eval-samples-table-rows');
  if (!tbody) return;

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="table-empty">No per-sample results available.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><strong>${esc(r.user_input)}</strong></td>
      <td><span class="badge ${scoreClass(r.faithfulness)}">${fmtScore(r.faithfulness)}</span></td>
      <td><span class="badge ${scoreClass(r.answer_relevancy)}">${fmtScore(r.answer_relevancy)}</span></td>
      <td><span class="badge ${scoreClass(r.context_precision)}">${fmtScore(r.context_precision)}</span></td>
      <td><span class="badge ${scoreClass(r.context_recall)}">${fmtScore(r.context_recall)}</span></td>
      <td><span class="badge ${scoreClass(r.answer_correctness)}">${fmtScore(r.answer_correctness)}</span></td>
    </tr>`).join('');
}

/* ── Goldens ── */

async function loadGoldens() {
  try {
    const d = await api('/api/evaluation/goldens');
    state.goldens = d.goldens || [];
    renderGoldensTable();
  } catch (e) { console.error('Goldens load failed:', e); }
}

function renderGoldensTable() {
  const tbody = $('goldens-table-rows');
  if (!tbody) return;

  if (!state.goldens.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="table-empty">No golden standards loaded.</td></tr>';
    return;
  }

  tbody.innerHTML = state.goldens.map((g, i) => `
    <tr>
      <td><span class="badge">${i + 1}</span></td>
      <td><strong>${esc(g.question)}</strong></td>
      <td style="font-size:12.5px;color:var(--ink-secondary);max-width:300px;line-height:1.45">${esc(g.ground_truth)}</td>
      <td><button class="btn-ghost-sm" data-edit-golden="${g.id}">Edit</button></td>
    </tr>`).join('');

  tbody.querySelectorAll('[data-edit-golden]').forEach(btn => {
    btn.addEventListener('click', () => openModal(btn.dataset.editGolden));
  });
}

function openModal(id) {
  const g = state.goldens.find(x => x.id === id);
  if (!g) return;
  $('modal-field-id').value = g.id;
  $('modal-field-question').textContent = g.question;
  $('modal-field-answer').value = g.ground_truth;
  $('edit-golden-modal').hidden = false;
  $('modal-field-answer')?.focus();
}

function closeModal() {
  $('edit-golden-modal').hidden = true;
}

async function saveGolden() {
  const id = $('modal-field-id')?.value;
  const answer = $('modal-field-answer')?.value.trim();
  if (!answer) return;

  try {
    await api('/api/evaluation/goldens', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, ground_truth: answer }),
    });
    closeModal();
    await loadGoldens();
  } catch (e) {
    alert('Could not save: ' + e.message);
  }
}
