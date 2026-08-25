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
  };
  currentTopic = 'auto';
  renderActiveChatMessages();
  renderChatList();
}

function switchToChat(id) {
  const found = chats.find((c) => c.id === id);
  if (!found) return;
  activeChat = found;
  currentTopic = 'auto';
  renderActiveChatMessages();
  renderChatList();
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
    const res = await fetch(`${serverUrl}/api/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // GENERAL is explicit and only selected through the visible opt-in toggle.
      body: JSON.stringify({
        question,
        topic: openEndedToggle.checked ? 'general' : currentTopic,
        session_id: activeChat.id,
      }),
      // TASK 18 (2026-08-24): raised from 30s -- correct against the old
      // Phi-3 backend this was tested with, but not against Mistral. Real
      // Mistral answers currently measure ~10s (see workspace.md Entry
      // 017), but the ORIGINAL 124-134s figure that motivated this whole
      // investigation was never fully explained, so 180s is a deliberate
      // safety margin, not a claim that answers actually take that long.
      // The real fix is still switching to /api/ask/stream (not done here
      // -- this is the minimal change to unblock manual testing today).
      signal: AbortSignal.timeout(180000),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    const data = await res.json();

    pendingCard.classList.remove('pending');
    pendingCard.classList.add(`topic-${data.topic_used}`);
    pendingCard.querySelector('.card-meta').textContent = labelForTopic(data.topic_used, false);
    pendingCard.lastChild.textContent = data.answer;
    addReactionControls(pendingCard, data.query_id);

    activeChat.messages.push({
      who: 'bot', text: data.answer, topic: data.topic_used, query_id: data.query_id || null,
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
