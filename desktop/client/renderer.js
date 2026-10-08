const messagesEl = document.getElementById('messages');
const composer = document.getElementById('composer');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
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
const attachmentChip = document.getElementById('attachmentChip');
const removeAttachmentBtn = document.getElementById('removeAttachmentBtn');
const attachmentStatus = document.getElementById('attachmentStatus');
const quizBtn = document.getElementById('quizBtn');
const quizModal = document.getElementById('quizModal');
const quizTopicInput = document.getElementById('quizTopicInput');
const quizCountInput = document.getElementById('quizCountInput');
const quizSourceDocument = document.getElementById('quizSourceDocument');
const quizStatus = document.getElementById('quizStatus');
const quizCancel = document.getElementById('quizCancel');
const quizGenerate = document.getElementById('quizGenerate');

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
  activeChat = found;
  if (!activeChat.documentIds) activeChat.documentIds = [];
  if (!activeChat.attachedDocuments) activeChat.attachedDocuments = [];
  currentTopic = 'auto';
  renderActiveChatMessages();
  renderChatList();
  renderAttachmentRow();
}

// ---------- Document attachment (Task 35: grounded document Q&A) ----------
// One PDF per chat for this first pass. The chat's own id doubles as the
// server-side document session_id, so an attached document is private to
// this one chat -- matching the per-session privacy scope the backend
// already enforces (core/document_service.py / stores/document_store.py).

function renderAttachmentRow() {
  const docs = (activeChat && activeChat.attachedDocuments) || [];
  if (!docs.length) {
    attachmentRow.classList.add('hidden');
    attachmentChip.textContent = '';
    return;
  }
  attachmentRow.classList.remove('hidden');
  attachmentChip.textContent = `📄 ${docs[0].filename} — ask questions about this document`;
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

documentFileInput.addEventListener('change', async () => {
  const file = documentFileInput.files[0];
  documentFileInput.value = '';
  if (!file) return;
  ensureActiveChat();
  setAttachmentStatus(`Uploading ${file.name}…`);
  try {
    const form = new FormData();
    form.append('session_id', activeChat.id);
    form.append('file', file);
    const res = await fetch(`${serverUrl}/api/documents`, { method: 'POST', body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `server returned ${res.status}`);
    }
    const metadata = await res.json();
    activeChat.documentIds = [metadata.document_id];
    activeChat.attachedDocuments = [{ document_id: metadata.document_id, filename: metadata.filename }];
    renderAttachmentRow();
    setAttachmentStatus(`Attached — ${metadata.pages} page(s), ready for questions.`);
    await saveActiveChatIfNeeded();
  } catch (err) {
    setAttachmentStatus(`Couldn't attach ${file.name}: ${err.message}`);
  }
});

removeAttachmentBtn.addEventListener('click', async () => {
  if (!activeChat || !activeChat.documentIds || !activeChat.documentIds.length) return;
  const [documentId] = activeChat.documentIds;
  activeChat.documentIds = [];
  activeChat.attachedDocuments = [];
  renderAttachmentRow();
  setAttachmentStatus('');
  await saveActiveChatIfNeeded();
  // Best-effort server-side cleanup; the document also auto-expires after
  // 24h regardless, so a failure here is not user-visible or data-unsafe.
  fetch(`${serverUrl}/api/documents/${documentId}?session_id=${encodeURIComponent(activeChat.id)}`, {
    method: 'DELETE',
  }).catch(() => {});
});

// ---------- Quiz (Task 36) ----------

const QUIZ_SOURCE_LABELS = {
  pretrained: 'Quiz source: Mistral pretrained knowledge',
  aum: 'Quiz source: AUM Housing policy',
  document: 'Quiz source: your attached document',
};

function openQuizModal() {
  ensureActiveChat();
  const hasDocument = activeChat.documentIds && activeChat.documentIds.length > 0;
  quizSourceDocument.disabled = !hasDocument;
  if (!hasDocument && quizSourceDocument.checked) {
    document.querySelector('input[name="quizSource"][value="pretrained"]').checked = true;
  }
  quizStatus.textContent = '';
  quizModal.classList.remove('hidden');
  quizTopicInput.focus();
}

function closeQuizModal() {
  quizModal.classList.add('hidden');
}

quizBtn.addEventListener('click', openQuizModal);
quizCancel.addEventListener('click', closeQuizModal);

quizGenerate.addEventListener('click', async () => {
  const topic = quizTopicInput.value.trim();
  const count = Math.max(1, Math.min(20, parseInt(quizCountInput.value, 10) || 5));
  const sourceMode = document.querySelector('input[name="quizSource"]:checked').value;
  if (!topic) {
    quizStatus.textContent = 'Please enter a topic.';
    return;
  }
  quizGenerate.disabled = true;
  quizStatus.textContent = 'Generating quiz… this can take a while for more questions.';
  try {
    const res = await fetch(`${serverUrl}/api/quiz`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic, count, source_mode: sourceMode,
        document_ids: sourceMode === 'document' ? activeChat.documentIds : undefined,
        session_id: activeChat.id,
      }),
      signal: AbortSignal.timeout(300000),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || `server returned ${res.status}`);
    closeQuizModal();
    renderQuizCard(body.quiz, sourceMode);
  } catch (err) {
    quizStatus.textContent = `Couldn't generate a quiz: ${err.message}`;
  } finally {
    quizGenerate.disabled = false;
  }
});

function renderQuizCard(quiz, sourceMode) {
  ensureActiveChat();
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
      radio.name = `quiz-${quiz.title}-${qi}`;
      radio.value = String(oi);
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

  submitBtn.addEventListener('click', () => {
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
  });

  messagesEl.appendChild(card);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function ensureActiveChat() {
  if (!activeChat) startNewChat();
}

async function saveActiveChatIfNeeded() {
  if (!activeChat || activeChat.temporary) return;
  const existing = chats.find((c) => c.id === activeChat.id);
  if (!existing) chats.unshift(activeChat);
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
    addCard({ text: m.text, who: m.who, topic: m.topic, query_id: m.query_id, skipStore: true });
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

async function consumeSSE(response, onDelta) {
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
  return 'AUM Chatbot';
}

// ---------- Composer ----------

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
    });

    pendingCard.classList.remove('pending');
    pendingCard.classList.add(`topic-${data.topic_used}`);
    pendingCard.querySelector('.card-meta').textContent = labelForTopic(data.topic_used, false);
    pendingCard.lastChild.textContent = streamedAnswer;
    addReactionControls(pendingCard, data.query_id);

    activeChat.messages.push({
      who: 'bot', text: streamedAnswer, topic: data.topic_used, query_id: data.query_id || null,
    });
  } catch (err) {
    const errText = `Couldn't reach the server (${err.message}). Please try again in a moment.`;
    pendingCard.classList.remove('pending');
    pendingCard.querySelector('.card-meta').textContent = 'Error';
    pendingCard.lastChild.textContent = errText;
    addReactionControls(pendingCard, null);
    activeChat.messages.push({ who: 'bot', text: errText, topic: null, query_id: null });
  } finally {
    sendBtn.disabled = false;
    questionInput.focus();
    renderChatList();
    await saveActiveChatIfNeeded();
  }
});

// ---------- Init ----------

(async function init() {
  await loadServerConfig();
  await Promise.all([loadChats(), loadProjects()]);
  startNewChat();
})();
