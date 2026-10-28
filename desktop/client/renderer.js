const messagesEl = document.getElementById('messages');
const composer = document.getElementById('composer');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const stopBtn = document.getElementById('stopBtn');
const newChatBtn = document.getElementById('newChatBtn');
const newProjectBtn = document.getElementById('newProjectBtn');
const projectCreator = document.getElementById('projectCreator');
const projectNameInput = document.getElementById('projectNameInput');
const cancelProjectBtn = document.getElementById('cancelProjectBtn');
const tempToggle = document.getElementById('tempToggle');
const openEndedToggle = document.getElementById('openEndedToggle');
const modeHint = document.getElementById('modeHint');
const pinnedSection = document.getElementById('pinnedSection');
const pinnedList = document.getElementById('pinnedList');
const chatList = document.getElementById('chatList');
const projectsSection = document.getElementById('projectsSection');
const projectList = document.getElementById('projectList');
const chatsSection = document.getElementById('chatsSection');
const feedbackModal = document.getElementById('feedbackModal');
const feedbackConsent = document.getElementById('feedbackConsent');
const feedbackComment = document.getElementById('feedbackComment');
const feedbackStatus = document.getElementById('feedbackStatus');
const feedbackCancel = document.getElementById('feedbackCancel');
const feedbackConfirm = document.getElementById('feedbackConfirm');
const attachBtn = document.getElementById('attachBtn');
const documentFileInput = document.getElementById('documentFileInput');
const attachmentRow = document.getElementById('attachmentRow');
const attachmentStatus = document.getElementById('attachmentStatus');
const chatArea = document.getElementById('chatArea');
const coursesList = document.getElementById('coursesList');
const courseView = document.getElementById('courseView');
const courseViewTitle = document.getElementById('courseViewTitle');
const courseViewBody = document.getElementById('courseViewBody');
const courseBackBtn = document.getElementById('courseBackBtn');

// Task 56: how many documents one chat may have attached at once. Bounds
// both UI clutter and retrieval load -- AssistantService.quiz()'s
// document branch pools up to 6 chunks PER document before picking its
// best 6 overall, and document_chat() does its own per-document FAISS
// search per attached file, so this is a real resource knob, not just a
// UI nicety. 5 is a deliberate, modest cap, not derived from a measured
// limit -- revisit with real multi-document latency numbers if this
// turns out to be too low (or too generous) in practice.
const MAX_ATTACHED_DOCUMENTS = 5;

let pendingFeedback = null;

let serverUrl = 'http://localhost:8000';
let currentTopic = 'auto';

// Saved chats only -- temporary chats are never added to this array, so they
// never get persisted and never show up in the sidebar list, matching
// "temporary chat (won't be saved)".
let chats = [];
let projects = [];
let activeChat = null;

const EMPTY_STATE_HTML = `
  <div class="empty-state">
    <p class="empty-title">What can I help with?</p>
    <div class="topic-cards">
      <button class="topic-card" data-topic="cos">
        <span class="topic-card-title">Research</span>
        <span class="topic-card-sub">AUM's research projects and symposium</span>
      </button>
      <button class="topic-card" data-topic="housing">
        <span class="topic-card-title">Housing &amp; Community</span>
        <span class="topic-card-sub">Residence hall policies and standards</span>
      </button>
    </div>
    <p class="empty-sub">Or just type your question below.</p>
  </div>
`;

// ---------- Server config (loaded silently, no UI) ----------

async function loadServerConfig() {
  const cfg = await window.aumApp.getServerConfig();
  serverUrl = cfg.serverUrl || serverUrl;
}

// Open-ended mode is intentionally session-only. It starts OFF whenever the
// desktop app launches, so users always opt in visibly before Mistral is used
// outside the grounded AUM routes.
function updateModeHint() {
  modeHint.textContent = openEndedToggle.checked
    ? "Open-ended questions are answered by Mistral using its pretrained knowledge; they are not grounded in AUM's research records or housing policy."
    : "Answers are grounded in AUM's research records and housing policy — not general knowledge.";
}

openEndedToggle.addEventListener('change', updateModeHint);
updateModeHint();

// ---------- Chat and project storage ----------

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function truncateTitle(text) {
  const t = text.trim();
  return t.length > 40 ? `${t.slice(0, 40)}…` : t;
}

function projectIdFor(chat) {
  return chat.project_id || null;
}

async function loadChats() {
  chats = await window.aumApp.getChats();
  if (!Array.isArray(chats)) chats = [];
}

async function loadProjects() {
  projects = await window.aumApp.getProjects();
  if (!Array.isArray(projects)) projects = [];
}

async function persistChats() {
  // Old chat records without project_id remain valid; it is added only when
  // the user next saves or changes a chat, rather than forcing a migration.
  chats = await window.aumApp.setChats(chats.map((chat) => ({
    ...chat,
    project_id: projectIdFor(chat),
  })));
}

async function persistProjects() {
  projects = await window.aumApp.setProjects(projects);
}

function startNewChat() {
  showChatArea();
  activeChat = {
    id: makeId(),
    title: null,
    messages: [],
    pinned: false,
    project_id: null,
    temporary: tempToggle.checked,
    createdAt: Date.now(),
    documentIds: [],
    attachedDocuments: [],
  };
  currentTopic = 'auto';
  renderActiveChatMessages();
  renderChatList();
  renderAttachmentRow();
}

function switchToChat(id) {
  const found = chats.find((c) => c.id === id);
  if (!found) return;
  showChatArea();
  activeChat = found;
  if (!activeChat.documentIds) activeChat.documentIds = [];
  if (!activeChat.attachedDocuments) activeChat.attachedDocuments = [];
  currentTopic = 'auto';
  renderActiveChatMessages();
  renderChatList();
  renderAttachmentRow();
}

// ---------- Document attachment (Task 35: grounded document Q&A; Task 56:
// up to MAX_ATTACHED_DOCUMENTS PDFs per chat instead of one). The chat's
// own id doubles as the server-side document session_id, so an attached
// document is private to this one chat -- matching the per-session
// privacy scope the backend already enforces (core/document_service.py /
// stores/document_store.py). AssistantService.quiz()/document_chat()
// already pool and rank chunks across every document_id given to them
// (document_chat() always did; quiz()'s document branch had a real bug
// where a document listed later could be silently dropped entirely --
// fixed in core/assistant_service.py, Task 56) -- this section only adds
// the client-side ability to actually attach more than one.

function renderAttachmentRow() {
  const docs = (activeChat && activeChat.attachedDocuments) || [];
  attachmentRow.innerHTML = '';
  if (!docs.length) {
    attachmentRow.classList.add('hidden');
    return;
  }
  attachmentRow.classList.remove('hidden');
  docs.forEach((doc) => {
    const group = document.createElement('span');
    group.className = 'attachment-chip-group';

    const chip = document.createElement('span');
    chip.className = 'attachment-chip';
    chip.textContent = `📄 ${doc.filename}`;
    group.appendChild(chip);

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'attachment-remove';
    removeBtn.setAttribute('aria-label', `Remove ${doc.filename}`);
    removeBtn.textContent = '✕';
    removeBtn.addEventListener('click', () => removeAttachedDocument(doc.document_id));
    group.appendChild(removeBtn);

    attachmentRow.appendChild(group);
  });
}

function setAttachmentStatus(text) {
  if (!text) {
    attachmentStatus.classList.add('hidden');
    attachmentStatus.textContent = '';
    return;
  }
  attachmentStatus.classList.remove('hidden');
  attachmentStatus.textContent = text;
}

attachBtn.addEventListener('click', () => {
  ensureActiveChat();
  documentFileInput.click();
});

async function uploadOneDocument(file) {
  const form = new FormData();
  form.append('session_id', activeChat.id);
  form.append('file', file);
  const res = await fetch(`${serverUrl}/api/documents`, { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `server returned ${res.status}`);
  }
  return res.json();
}

documentFileInput.addEventListener('change', async () => {
  const files = Array.from(documentFileInput.files || []);
  documentFileInput.value = '';
  if (!files.length) return;
  ensureActiveChat();

  const alreadyAttached = activeChat.attachedDocuments ? activeChat.attachedDocuments.length : 0;
  const room = MAX_ATTACHED_DOCUMENTS - alreadyAttached;
  const toUpload = files.slice(0, Math.max(room, 0));
  const skipped = files.length - toUpload.length;

  if (toUpload.length === 0) {
    setAttachmentStatus(`Already at the ${MAX_ATTACHED_DOCUMENTS}-document limit for this chat. Remove one first.`);
    return;
  }

  const failures = [];
  for (const file of toUpload) {
    setAttachmentStatus(`Uploading ${file.name}…`);
    try {
      const metadata = await uploadOneDocument(file);
      activeChat.documentIds = [...(activeChat.documentIds || []), metadata.document_id];
      activeChat.attachedDocuments = [
        ...(activeChat.attachedDocuments || []),
        { document_id: metadata.document_id, filename: metadata.filename },
      ];
      renderAttachmentRow();
    } catch (err) {
      failures.push(`${file.name}: ${err.message}`);
    }
  }

  await saveActiveChatIfNeeded();
  const notes = [];
  if (failures.length) notes.push(`Couldn't attach ${failures.join('; ')}`);
  if (skipped > 0) notes.push(`${skipped} file(s) skipped — ${MAX_ATTACHED_DOCUMENTS}-document limit reached.`);
  if (!notes.length) {
    const count = activeChat.attachedDocuments.length;
    setAttachmentStatus(`${count} document${count === 1 ? '' : 's'} attached, ready for questions.`);
  } else {
    setAttachmentStatus(notes.join(' '));
  }
});

async function removeAttachedDocument(documentId) {
  if (!activeChat || !activeChat.documentIds) return;
  activeChat.documentIds = activeChat.documentIds.filter((id) => id !== documentId);
  activeChat.attachedDocuments = (activeChat.attachedDocuments || []).filter(
    (doc) => doc.document_id !== documentId,
  );
  renderAttachmentRow();
  setAttachmentStatus('');
  await saveActiveChatIfNeeded();
  // Best-effort server-side cleanup; the document also auto-expires after
  // 24h regardless, so a failure here is not user-visible or data-unsafe.
  fetch(`${serverUrl}/api/documents/${documentId}?session_id=${encodeURIComponent(activeChat.id)}`, {
    method: 'DELETE',
  }).catch(() => {});
}

// ---------- Quiz (Task 36; Task 53/55 -- button removed, natural-language
// intent detection in chat is now the only way to request a quiz. See
// pipeline/quiz_intent.py and the composer's SSE completion handler
// below, which calls renderQuizCard() for any response with kind==='quiz'
// regardless of how the server decided to generate it.) ----------

const QUIZ_SOURCE_LABELS = {
  pretrained: 'Quiz source: Mistral pretrained knowledge',
  housing: 'Quiz source: AUM Housing policy',
  document: 'Quiz source: your attached document',
};

// Task 57-A: `record` is the stored chat message ({kind: 'quiz', quiz_id,
// quiz, source_mode, progress}). Selections and the submitted score are
// written back into record.progress and persisted, so reopening the chat
// restores the interactive quiz exactly as the user left it.
function renderQuizCard(quiz, sourceMode, record) {
  ensureActiveChat();
  const quizId = record.quiz_id;
  const progress = record.progress || (record.progress = { answers: [], submitted: false });
  const card = document.createElement('div');
  card.className = 'quiz-card';

  const sourceLabel = document.createElement('div');
  sourceLabel.className = 'quiz-source-label';
  sourceLabel.textContent = QUIZ_SOURCE_LABELS[sourceMode] || 'Quiz';
  card.appendChild(sourceLabel);

  const title = document.createElement('div');
  title.className = 'card-meta';
  title.textContent = quiz.title;
  card.appendChild(title);

  quiz.questions.forEach((q, qi) => {
    const qWrap = document.createElement('div');
    qWrap.className = 'quiz-question';

    const qText = document.createElement('div');
    qText.className = 'quiz-question-text';
    qText.textContent = `${qi + 1}. ${q.question}`;
    qWrap.appendChild(qText);

    const optsWrap = document.createElement('div');
    optsWrap.className = 'quiz-options';
    q.options.forEach((opt, oi) => {
      const label = document.createElement('label');
      label.className = 'quiz-option';
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = `quiz-${quizId}-${qi}`;
      radio.value = String(oi);
      radio.checked = progress.answers[qi] === oi;
      radio.addEventListener('change', () => {
        progress.answers[qi] = oi;
        saveActiveChatIfNeeded();
      });
      label.appendChild(radio);
      label.append(opt);
      optsWrap.appendChild(label);
    });
    qWrap.appendChild(optsWrap);

    const explanation = document.createElement('div');
    explanation.className = 'quiz-explanation hidden';
    explanation.textContent = q.explanation || '';
    qWrap.appendChild(explanation);

    qWrap.dataset.correctIndex = String(q.correct_index);
    card.appendChild(qWrap);
  });

  const submitBtn = document.createElement('button');
  submitBtn.type = 'button';
  submitBtn.className = 'quiz-submit-btn';
  submitBtn.textContent = 'Check answers';
  card.appendChild(submitBtn);

  const scoreEl = document.createElement('div');
  scoreEl.className = 'quiz-score';
  card.appendChild(scoreEl);

  const grade = () => {
    let correct = 0;
    card.querySelectorAll('.quiz-question').forEach((qWrap) => {
      const correctIndex = parseInt(qWrap.dataset.correctIndex, 10);
      const radios = qWrap.querySelectorAll('.quiz-option input');
      const selected = qWrap.querySelector('.quiz-option input:checked');
      const selectedIndex = selected ? parseInt(selected.value, 10) : -1;
      if (selectedIndex === correctIndex) correct += 1;
      radios.forEach((radio, idx) => {
        radio.disabled = true;
        const optionLabel = radio.closest('.quiz-option');
        if (idx === correctIndex) optionLabel.classList.add('correct');
        else if (idx === selectedIndex) optionLabel.classList.add('incorrect');
      });
      qWrap.querySelector('.quiz-explanation').classList.remove('hidden');
    });
    scoreEl.textContent = `Score: ${correct} / ${quiz.questions.length}`;
    submitBtn.disabled = true;
    return correct;
  };
  submitBtn.addEventListener('click', () => {
    progress.score = grade();
    progress.submitted = true;
    saveActiveChatIfNeeded();
  });
  if (progress.submitted) grade();

  messagesEl.appendChild(card);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function ensureActiveChat() {
  if (!activeChat) startNewChat();
}

async function saveActiveChatIfNeeded() {
  if (!activeChat || activeChat.temporary) return;
  // persistChats() replaces `chats` with IPC copies, so after the first save
  // the entry here is no longer `activeChat` -- the old `if (!existing)` then
  // re-saved that stale copy and every exchange after the first was lost.
  // Messages come from activeChat; pin/project/title edits made through the
  // chat menu live on the stored entry and are kept.
  const i = chats.findIndex((c) => c.id === activeChat.id);
  if (i === -1) chats.unshift(activeChat);
  else chats[i] = { ...chats[i], messages: activeChat.messages, title: chats[i].title || activeChat.title };
  await persistChats();
  renderChatList();
}

newChatBtn.addEventListener('click', startNewChat);
newProjectBtn.addEventListener('click', () => {
  projectCreator.classList.remove('hidden');
  projectNameInput.focus();
});
cancelProjectBtn.addEventListener('click', () => {
  projectCreator.classList.add('hidden');
  projectNameInput.value = '';
});
projectCreator.addEventListener('submit', async (event) => {
  event.preventDefault();
  const trimmedName = projectNameInput.value.trim();
  if (!trimmedName) {
    projectNameInput.focus();
    return;
  }
  projects.push({ id: makeId(), name: trimmedName, createdAt: Date.now() });
  await persistProjects();
  projectCreator.classList.add('hidden');
  projectNameInput.value = '';
  renderChatList();
});

// ---------- Sidebar chat and project rendering ----------

function renderChatList() {
  const pinned = chats.filter((chat) => chat.pinned && !projectIdFor(chat));
  const unassigned = chats.filter((chat) => !chat.pinned && !projectIdFor(chat));

  pinnedSection.classList.toggle('hidden', pinned.length === 0);
  chatsSection.classList.toggle('hidden', unassigned.length === 0);
  projectsSection.classList.toggle('hidden', projects.length === 0);
  pinnedList.innerHTML = '';
  chatList.innerHTML = '';
  projectList.innerHTML = '';

  pinned.forEach((chat) => appendChatRow(pinnedList, chat));
  unassigned.forEach((chat) => appendChatRow(chatList, chat));

  projects.forEach((project) => {
    const group = document.createElement('div');
    group.className = 'project-group';
    const heading = document.createElement('p');
    heading.className = 'project-name';
    heading.textContent = project.name;
    const list = document.createElement('div');
    list.className = 'chat-list';
    chats.filter((chat) => projectIdFor(chat) === project.id)
      .forEach((chat) => appendChatRow(list, chat));
    group.appendChild(heading);
    group.appendChild(list);
    projectList.appendChild(group);
  });
}

function closeChatMenus(except = null) {
  document.querySelectorAll('.chat-menu:not(.hidden)').forEach((menu) => {
    if (menu !== except) menu.classList.add('hidden');
  });
}

function menuButton(label, action, className = '') {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `chat-menu-item ${className}`.trim();
  button.textContent = label;
  button.addEventListener('click', action);
  return button;
}

function appendChatRow(container, chatObj) {
  const row = document.createElement('div');
  row.className = 'chat-row';
  row.dataset.id = chatObj.id;
  if (activeChat && activeChat.id === chatObj.id) row.classList.add('active');

  const title = document.createElement('span');
  title.className = 'chat-row-title';
  title.textContent = chatObj.title || 'New chat';

  const menuToggle = document.createElement('button');
  menuToggle.type = 'button';
  menuToggle.className = 'chat-menu-toggle';
  menuToggle.textContent = '⋯';
  menuToggle.title = 'Chat actions';
  menuToggle.setAttribute('aria-label', `Chat actions for ${chatObj.title || 'New chat'}`);

  const menu = document.createElement('div');
  menu.className = 'chat-menu hidden';
  menu.appendChild(menuButton(chatObj.pinned ? 'Unpin chat' : 'Pin chat', async (event) => {
    event.stopPropagation();
    chatObj.pinned = !chatObj.pinned;
    await persistChats();
    renderChatList();
  }));

  projects.forEach((project) => {
    menu.appendChild(menuButton(`Move to ${project.name}`, async (event) => {
      event.stopPropagation();
      chatObj.project_id = project.id;
      await persistChats();
      renderChatList();
    }));
  });

  if (projectIdFor(chatObj)) {
    menu.appendChild(menuButton('Remove from project', async (event) => {
      event.stopPropagation();
      chatObj.project_id = null;
      await persistChats();
      renderChatList();
    }));
  }

  menu.appendChild(menuButton('Delete chat', async (event) => {
    event.stopPropagation();
    if (!window.confirm(`Delete “${chatObj.title || 'New chat'}”?`)) return;
    const wasActive = activeChat && activeChat.id === chatObj.id;
    chats = chats.filter((chat) => chat.id !== chatObj.id);
    await persistChats();
    if (wasActive) startNewChat();
    else renderChatList();
  }, 'danger'));

  menuToggle.addEventListener('click', (event) => {
    event.stopPropagation();
    const opening = menu.classList.contains('hidden');
    closeChatMenus(menu);
    menu.classList.toggle('hidden', !opening);
  });
  row.addEventListener('click', () => switchToChat(chatObj.id));

  row.appendChild(title);
  row.appendChild(menuToggle);
  row.appendChild(menu);
  container.appendChild(row);
}

document.addEventListener('click', () => closeChatMenus());

// ---------- Topic cards (empty state) ----------

function wireTopicCards() {
  document.querySelectorAll('.topic-card').forEach((btn) => {
    btn.addEventListener('click', () => {
      // Choosing an AUM source card is an explicit grounded-mode choice.
      openEndedToggle.checked = false;
      updateModeHint();
      currentTopic = btn.dataset.topic;
      questionInput.focus();
    });
  });
}

// ---------- Message rendering ----------

function renderActiveChatMessages() {
  messagesEl.innerHTML = '';
  if (!activeChat || activeChat.messages.length === 0) {
    messagesEl.innerHTML = EMPTY_STATE_HTML;
    wireTopicCards();
    return;
  }
  activeChat.messages.forEach((m) => {
    if (m.kind === 'quiz' && m.quiz) renderQuizCard(m.quiz, m.source_mode, m);
    else addCard({ text: m.text, who: m.who, topic: m.topic, query_id: m.query_id, skipStore: true });
  });
}

function clearEmptyState() {
  const empty = messagesEl.querySelector('.empty-state');
  if (empty) empty.remove();
}

function addCard({ text, who, topic, query_id, pending, skipStore }) {
  clearEmptyState();
  const card = document.createElement('div');
  card.className = `card ${who}`;
  if (topic) card.classList.add(`topic-${topic}`);
  if (pending) card.classList.add('pending');

  const meta = document.createElement('div');
  meta.className = 'card-meta';
  meta.textContent = who === 'user' ? 'You' : labelForTopic(topic, pending);

  const body = document.createElement('div');
  body.className = 'card-body';
  body.textContent = text;

  card.appendChild(meta);
  card.appendChild(body);
  if (who === 'bot' && !pending) addReactionControls(card, query_id);
  messagesEl.appendChild(card);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  if (!skipStore && activeChat) {
    activeChat.messages.push({ who, text, topic: topic || null, query_id: query_id || null });
  }
  return card;
}

function addReactionControls(card, query_id) {
  const controls = document.createElement('div');
  controls.className = 'reaction-controls';
  const canSubmit = Boolean(query_id);

  [['up', '👍', 'Helpful'], ['down', '👎', 'Not helpful']].forEach(([reaction, icon, label]) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'reaction-btn';
    button.textContent = icon;
    button.title = canSubmit ? label : 'Feedback is available for new responses.';
    button.setAttribute('aria-label', label);
    button.disabled = !canSubmit;
    button.addEventListener('click', () => openFeedbackModal(reaction, query_id, card, controls));
    controls.appendChild(button);
  });
  card.appendChild(controls);
}

async function consumeSSE(response, onDelta, onMeta) {
  if (!response.body) throw new Error('streaming is not supported by this response');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let eventName = 'message';
  let dataLines = [];
  let donePayload = null;

  const dispatch = () => {
    if (!dataLines.length) return;
    const data = dataLines.join('\n');
    if (eventName === 'done') donePayload = JSON.parse(data);
    // "meta" announces the real query_id as soon as the server knows it
    // -- well before the answer is finished -- specifically so a Stop
    // button has something real to send (server/api_server.py's
    // /api/ask/stop). Without this, query_id is only ever known from
    // "done", by which point there's nothing left to stop.
    else if (eventName === 'meta') { if (onMeta) onMeta(JSON.parse(data)); }
    else onDelta(data);
    eventName = 'message';
    dataLines = [];
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split(/\r?\n/);
    buffer = done ? '' : lines.pop();
    for (const line of lines) {
      if (!line) { dispatch(); continue; }
      if (line.startsWith('event:')) eventName = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
    }
    if (done) break;
  }
  dispatch();
  if (!donePayload) throw new Error('stream ended without a completion event');
  return donePayload;
}

function selectedFeedbackScope() {
  return document.querySelector('input[name="feedbackScope"]:checked').value;
}

function updateFeedbackConsent() {
  feedbackConsent.textContent = selectedFeedbackScope() === 'conversation'
    ? 'With your permission, your full conversation, including your messages and AUM Chatbot responses, will be logged to help improve answer accuracy. No account or personal identity is attached beyond what you choose to type.'
    : 'With your permission, this message and the AUM Chatbot response will be logged to help improve answer accuracy. No account or personal identity is attached beyond what you choose to type.';
}

function openFeedbackModal(reaction, query_id, card, controls) {
  pendingFeedback = { reaction, query_id, card, controls };
  feedbackComment.value = '';
  feedbackStatus.textContent = '';
  document.querySelector('input[name="feedbackScope"][value="single"]').checked = true;
  updateFeedbackConsent();
  feedbackModal.classList.remove('hidden');
  feedbackComment.focus();
}

function closeFeedbackModal() {
  feedbackModal.classList.add('hidden');
  pendingFeedback = null;
}

document.querySelectorAll('input[name="feedbackScope"]').forEach((input) => {
  input.addEventListener('change', updateFeedbackConsent);
});
feedbackCancel.addEventListener('click', closeFeedbackModal);
feedbackModal.addEventListener('click', (event) => {
  if (event.target === feedbackModal) closeFeedbackModal();
});

feedbackConfirm.addEventListener('click', async () => {
  if (!pendingFeedback || !activeChat) return;
  const scope = selectedFeedbackScope();
  feedbackConfirm.disabled = true;
  feedbackStatus.textContent = 'Sending feedback…';
  try {
    const payload = {
      reaction: pendingFeedback.reaction,
      scope,
      query_id: pendingFeedback.query_id,
      session_id: activeChat.id,
      comment: feedbackComment.value.trim() || null,
    };
    if (scope === 'conversation') payload.conversation = activeChat.messages;

    const response = await fetch(`${serverUrl}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`server returned ${response.status}`);

    pendingFeedback.controls.querySelectorAll('.reaction-btn').forEach((button) => {
      button.disabled = true;
    });
    const activeButton = pendingFeedback.controls.querySelector(
      `.reaction-btn:nth-child(${pendingFeedback.reaction === 'up' ? 1 : 2})`,
    );
    activeButton.classList.add('selected');
    closeFeedbackModal();
  } catch (error) {
    feedbackStatus.textContent = `Couldn't send feedback (${error.message}). Please try again.`;
  } finally {
    feedbackConfirm.disabled = false;
  }
});


function labelForTopic(topic, pending) {
  if (pending) return 'Thinking…';
  if (topic === 'cos') return 'Research Symposium';
  if (topic === 'housing') return 'Housing & Community Standards';
  if (topic === 'general') return 'Open-ended · Mistral';
  if (topic === 'open_ended_disabled') return 'Grounded AUM mode';
  if (topic === 'document') return 'Your document';
  if (topic === 'quiz_unavailable') return 'Quiz needs a source';
  return 'AUM Chatbot';
}

// ---------- Composer ----------

// Set once "meta" announces the query_id of whatever is currently
// streaming (see consumeSSE's onMeta below); cleared once that request
// finishes. stopBtn's click handler sends whatever this holds, so it's
// null (nothing to stop) outside of an active generation.
let currentStreamQueryId = null;

stopBtn.addEventListener('click', async () => {
  if (!currentStreamQueryId) return;
  stopBtn.disabled = true;
  try {
    await fetch(`${serverUrl}/api/ask/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_id: currentStreamQueryId }),
    });
    // No client-side UI change here beyond this -- the in-flight
    // fetch/consumeSSE loop above is still reading the same stream, and
    // finishes normally (a real, if short, "done" event) once the
    // server-side stop takes effect, same code path as any other
    // completed answer.
  } catch (err) {
    // Stop is best-effort: if this request itself fails, the answer
    // just keeps streaming -- nothing else to show the user here.
  } finally {
    stopBtn.disabled = false;
  }
});

composer.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  ensureActiveChat();

  addCard({ text: question, who: 'user', query_id: null });
  if (!activeChat.title) activeChat.title = truncateTitle(question);
  questionInput.value = '';
  sendBtn.disabled = true;

  const pendingCard = addCard({ text: '…', who: 'bot', pending: true, skipStore: true });

  try {
    const res = await fetch(`${serverUrl}/api/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // GENERAL is explicit and only selected through the visible opt-in toggle.
      body: JSON.stringify({
        question,
        topic: openEndedToggle.checked ? 'general' : currentTopic,
        session_id: activeChat.id,
        // An attached document always wins server-side (see api_server.py
        // _resolve_topic), so this is sent regardless of the mode toggle.
        document_ids: activeChat.documentIds && activeChat.documentIds.length
          ? activeChat.documentIds : undefined,
      }),
      // The streaming endpoint keeps the same deliberate safety margin.
      // TASK 18 (2026-08-24): raised from 30s -- correct against the old
      // Phi-3 backend this was tested with, but not against Mistral. Real
      // Mistral answers currently measure ~10s (see workspace.md Entry
      // 017), but the ORIGINAL 124-134s figure that motivated this whole
      // investigation was never fully explained, so 180s is a deliberate
      // safety margin, not a claim that answers actually take that long.
      signal: AbortSignal.timeout(180000),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    let streamedAnswer = '';
    const data = await consumeSSE(res, (delta) => {
      streamedAnswer += delta;
      pendingCard.lastChild.textContent = streamedAnswer;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }, (meta) => {
      if (meta.query_id) {
        currentStreamQueryId = meta.query_id;
        stopBtn.classList.remove('hidden');
      }
    });

    if (data.kind === 'quiz') {
      // Task 53/55: a natural-language quiz request ("generate 5 MCQs on
      // X") typed into ordinary chat is now the only way to get a quiz --
      // render it the same structured way the old button used to, not as
      // streamed prose. The placeholder "Generating your quiz…" delta
      // (server/api_server.py:ask_stream) was only ever meant to show
      // progress; drop it rather than leaving it as a stray bot bubble.
      pendingCard.remove();
      // Task 57-A: stored as the full quiz (plus a text summary so older
      // code paths and the chat list still have something to show).
      const record = {
        who: 'bot', kind: 'quiz', topic: 'quiz', query_id: null,
        quiz_id: crypto.randomUUID(), quiz: data.quiz, source_mode: data.source_mode,
        progress: { answers: [], submitted: false },
        text: `Generated a ${data.quiz.questions.length}-question quiz: "${data.quiz.title}"`,
      };
      activeChat.messages.push(record);
      renderQuizCard(data.quiz, data.source_mode, record);
    } else {
      pendingCard.classList.remove('pending');
      pendingCard.classList.add(`topic-${data.topic_used}`);
      pendingCard.querySelector('.card-meta').textContent = labelForTopic(data.topic_used, false);
      pendingCard.lastChild.textContent = streamedAnswer;
      addReactionControls(pendingCard, data.query_id);

      activeChat.messages.push({
        who: 'bot', text: streamedAnswer, topic: data.topic_used, query_id: data.query_id || null,
      });
    }
  } catch (err) {
    const errText = `Couldn't reach the server (${err.message}). Please try again in a moment.`;
    pendingCard.classList.remove('pending');
    pendingCard.querySelector('.card-meta').textContent = 'Error';
    pendingCard.lastChild.textContent = errText;
    addReactionControls(pendingCard, null);
    activeChat.messages.push({ who: 'bot', text: errText, topic: null, query_id: null });
  } finally {
    sendBtn.disabled = false;
    stopBtn.classList.add('hidden');
    currentStreamQueryId = null;
    questionInput.focus();
    renderChatList();
    await saveActiveChatIfNeeded();
  }
});

// ---------- Courses (pre-loaded sample data; clicking an assignment
// dumps its raw content -- no LLM call, no summarization, no chunking.
// See core/course_service.py for why the data is fictional sample
// content, not a live AUM course feed. ----------

let courses = [];

function showChatArea() {
  courseView.classList.add('hidden');
  chatArea.classList.remove('hidden');
  document.querySelectorAll('.course-row.active').forEach((row) => row.classList.remove('active'));
}

function showCourseView() {
  chatArea.classList.add('hidden');
  courseView.classList.remove('hidden');
}

function metaSpan(text) {
  const span = document.createElement('span');
  span.textContent = text;
  return span;
}

async function loadCourses() {
  try {
    const res = await fetch(`${serverUrl}/api/courses`);
    courses = res.ok ? await res.json() : [];
  } catch (err) {
    courses = [];
  }
  renderCoursesList();
}

function renderCoursesList() {
  coursesList.innerHTML = '';
  courses.forEach((course) => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'course-row';
    row.dataset.courseId = course.course_id;
    row.textContent = course.course_name;
    row.addEventListener('click', () => openCourse(course.course_id));
    coursesList.appendChild(row);
  });
}

async function openCourse(courseId) {
  showCourseView();
  document.querySelectorAll('.course-row').forEach((row) => {
    row.classList.toggle('active', row.dataset.courseId === courseId);
  });

  const course = courses.find((c) => c.course_id === courseId);
  courseViewTitle.textContent = course ? course.course_name : 'Course';
  courseBackBtn.classList.remove('hidden');
  courseBackBtn.textContent = '← Back to chat';
  courseBackBtn.dataset.mode = 'list';
  courseBackBtn.dataset.courseId = courseId;
  courseViewBody.innerHTML = '<p class="course-meta">Loading assignments…</p>';

  let assignments = [];
  try {
    const res = await fetch(`${serverUrl}/api/courses/${encodeURIComponent(courseId)}/assignments`);
    assignments = res.ok ? await res.json() : [];
  } catch (err) {
    assignments = [];
  }

  courseViewBody.innerHTML = '';
  if (course) {
    const meta = document.createElement('p');
    meta.className = 'course-meta';
    meta.textContent = [course.instructor, course.term].filter(Boolean).join(' · ');
    courseViewBody.appendChild(meta);
  }
  const list = document.createElement('div');
  list.className = 'assignment-list';
  assignments.forEach((a) => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'assignment-row';
    const title = document.createElement('span');
    title.className = 'assignment-row-title';
    title.textContent = a.title;
    const due = document.createElement('span');
    due.className = 'assignment-row-due';
    due.textContent = a.due_date ? `Due ${a.due_date}` : '';
    row.appendChild(title);
    row.appendChild(due);
    row.addEventListener('click', () => openAssignment(courseId, a.assignment_id));
    list.appendChild(row);
  });
  courseViewBody.appendChild(list);
}

async function openAssignment(courseId, assignmentId) {
  courseViewBody.innerHTML = '<p class="course-meta">Loading…</p>';
  courseBackBtn.textContent = '← Back to assignments';
  courseBackBtn.dataset.mode = 'dump';
  courseBackBtn.dataset.courseId = courseId;

  let assignment = null;
  try {
    const res = await fetch(
      `${serverUrl}/api/courses/${encodeURIComponent(courseId)}/assignments/${encodeURIComponent(assignmentId)}`,
    );
    if (res.ok) assignment = await res.json();
  } catch (err) {
    assignment = null;
  }

  courseViewBody.innerHTML = '';
  if (!assignment) {
    const errP = document.createElement('p');
    errP.className = 'course-meta';
    errP.textContent = "Couldn't load this assignment.";
    courseViewBody.appendChild(errP);
    return;
  }
  courseViewTitle.textContent = assignment.title;

  const meta = document.createElement('div');
  meta.className = 'assignment-dump-meta';
  if (assignment.due_date) meta.appendChild(metaSpan(`Due ${assignment.due_date}`));
  if (assignment.points != null) meta.appendChild(metaSpan(`${assignment.points} points`));
  meta.appendChild(metaSpan(assignment.course_name));
  courseViewBody.appendChild(meta);

  // The actual "dump": the content field is rendered exactly as stored,
  // no processing of any kind -- see core/course_service.py's own
  // docstring on get_assignment().
  const content = document.createElement('pre');
  content.className = 'assignment-dump-content';
  content.textContent = assignment.content;
  courseViewBody.appendChild(content);
}

courseBackBtn.addEventListener('click', () => {
  if (courseBackBtn.dataset.mode === 'dump') openCourse(courseBackBtn.dataset.courseId);
  else showChatArea();
});

// ---------- Init ----------

(async function init() {
  await loadServerConfig();
  await Promise.all([loadChats(), loadProjects(), loadCourses()]);
  startNewChat();
})();
