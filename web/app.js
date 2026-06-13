const STORAGE_KEY = "legal-rag-chat-v1";

const el = {
  newChatBtn: document.getElementById("newChatBtn"),
  conversationList: document.getElementById("conversationList"),
  chatTitle: document.getElementById("chatTitle"),
  statusBadge: document.getElementById("statusBadge"),

  // Ưu tiên id mới chatMessages, fallback về id cũ messages
  messages:
    document.getElementById("chatMessages") ||
    document.getElementById("messages"),

  chatForm: document.getElementById("chatForm"),
  questionInput: document.getElementById("questionInput"),
  sendBtn: document.getElementById("sendBtn"),
  strategySelect: document.getElementById("strategySelect"),
  topKInput: document.getElementById("topKInput"),
  generateToggle: document.getElementById("generateToggle"),
  keywordInput: document.getElementById("keywordInput"),
  lookupBtn: document.getElementById("lookupBtn"),
  selectionHint: document.getElementById("selectionHint"),
  latestSources: document.getElementById("latestSources"),
  lookupResults: document.getElementById("lookupResults"),
};

let conversations = loadState();
let activeId = conversations[0]?.id || null;
let lastSelection = "";
let selectionTimer = null;

function getMessagesElement() {
  return (
    el.messages ||
    document.getElementById("chatMessages") ||
    document.getElementById("messages")
  );
}

function scrollChatToBottom() {
  const messagesEl = getMessagesElement();

  if (!messagesEl) {
    console.warn(
      'Không tìm thấy vùng tin nhắn. Hãy gắn id="chatMessages" cho div chứa bong bóng chat.'
    );
    return;
  }

  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations.slice(0, 30)));
}

function createConversation(options = {}) {
  const shouldRender = options.shouldRender !== false;

  const convo = {
    id:
      window.crypto && crypto.randomUUID
        ? crypto.randomUUID()
        : String(Date.now()),
    title: "Cuộc trò chuyện mới",
    messages: [],
    createdAt: Date.now(),
  };

  conversations.unshift(convo);
  activeId = convo.id;
  saveState();

  if (shouldRender) {
    render();
    scrollChatToBottom();
  }

  return convo;
}

function activeConversation() {
  return conversations.find((c) => c.id === activeId) || conversations[0];
}

function setStatus(text, type = "ready") {
  if (!el.statusBadge) return;

  el.statusBadge.textContent = text;
  el.statusBadge.className = `status-badge ${
    type === "loading" ? "loading" : type === "error" ? "error" : ""
  }`;
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatAnswerHtml(text) {
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

function formatMs(ms) {
  if (ms == null) return "";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function sourceTitle(doc) {
  const meta = doc.metadata || {};
  const title = doc.title || meta.title || "Văn bản pháp luật";
  const article = doc.article || meta.article;
  const so = doc.so_ky_hieu || meta.so_ky_hieu;
  const parts = [title];

  if (so) parts.push(so);
  if (article) parts.push(`Điều ${article}`);

  return parts.join(" | ");
}

function renderSourceCard(doc) {
  const meta = doc.metadata || {};
  const title = sourceTitle(doc);
  const article = doc.article || meta.article;
  const clause = doc.clause || meta.clause;
  const status = doc.status || meta.tinh_trang_hieu_luc;
  const route = doc.route || meta.route;
  const snippet = doc.snippet || doc.page_content || "";
  const url = doc.source_url || meta.source_url || meta.url;

  const metaParts = [];

  if (article) metaParts.push(`Điều ${escapeHtml(article)}`);
  if (clause) metaParts.push(`Khoản ${escapeHtml(clause)}`);
  if (status) metaParts.push(escapeHtml(status));
  if (route) metaParts.push(`route=${escapeHtml(route)}`);

  return `
    <article class="source-card">
      <h3>${escapeHtml(title)}</h3>
      <div class="meta">${metaParts.join(" · ") || "Nguồn pháp luật"}</div>
      <p>${escapeHtml(snippet)}</p>
      ${
        url
          ? `<a href="${escapeHtml(
              url
            )}" target="_blank" rel="noopener noreferrer">Mở văn bản gốc</a>`
          : ""
      }
    </article>
  `;
}

function renderSources(target, docs) {
  if (!target) return;

  if (!docs || docs.length === 0) {
    target.innerHTML = `<div class="selection-hint">Chưa có nguồn.</div>`;
    return;
  }

  target.innerHTML = docs.map(renderSourceCard).join("");
}

function renderConversations() {
  if (!el.conversationList) return;

  el.conversationList.innerHTML = conversations
    .map(
      (c) => `
    <div class="conversation-item ${c.id === activeId ? "active" : ""}" data-id="${c.id}">
      <div class="conversation-title">${escapeHtml(c.title)}</div>
      <div class="conversation-meta">${c.messages.length} tin nhắn</div>
    </div>
  `
    )
    .join("");

  document.querySelectorAll(".conversation-item").forEach((item) => {
    item.addEventListener("click", () => {
      activeId = item.dataset.id;
      saveState();
      render();
      scrollChatToBottom();
    });
  });
}

function renderMessages() {
  const messagesEl = getMessagesElement();
  const convo = activeConversation();

  if (!messagesEl) {
    console.warn(
      'Không tìm thấy vùng tin nhắn. Hãy kiểm tra index.html có div id="chatMessages" hoặc id="messages" chưa.'
    );
    return;
  }

  if (el.chatTitle) {
    el.chatTitle.textContent = convo?.title || "Cuộc trò chuyện mới";
  }

  if (!convo || convo.messages.length === 0) {
    messagesEl.innerHTML = `
      <div class="empty-state">
        <h3>Hỏi đáp pháp luật Hôn nhân & Gia đình</h3>
        <p>Ví dụ: “Điều kiện công nhận thuận tình ly hôn là gì?” hoặc “Con dưới 36 tháng tuổi khi ly hôn giao cho ai nuôi?”</p>
      </div>
    `;

    renderSources(el.latestSources, []);
    scrollChatToBottom();
    return;
  }

  messagesEl.innerHTML = convo.messages
    .map((m, idx) => {
      const sources = m.sources || [];

      const chips = sources.length
        ? `
      <div class="source-chips">
        ${sources
          .slice(0, 5)
          .map(
            (s, i) => `
              <button class="source-chip" data-message="${idx}" data-source="${i}" type="button">
                Nguồn ${i + 1}: ${escapeHtml(
              s.article ? `Điều ${s.article}` : s.so_ky_hieu || "VB"
            )}
              </button>
            `
          )
          .join("")}
      </div>
    `
        : "";

      const timing = m.timings?.total_ms
        ? ` · ${formatMs(m.timings.total_ms)}`
        : "";

      const bubbleContent =
        m.role === "assistant"
          ? formatAnswerHtml(m.content)
          : escapeHtml(m.content);

      return `
      <div class="message ${escapeHtml(m.role)}">
        <div class="bubble">${bubbleContent}</div>
        <div class="message-meta">${
          m.role === "user" ? "Bạn" : "Trợ lý"
        }${timing}</div>
        ${chips}
      </div>
    `;
    })
    .join("");

  const lastAssistant = [...convo.messages]
    .reverse()
    .find((m) => m.role === "assistant" && m.sources?.length);

  renderSources(el.latestSources, lastAssistant?.sources || []);

  document.querySelectorAll(".source-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const msg = convo.messages[Number(btn.dataset.message)];
      const source = msg?.sources?.[Number(btn.dataset.source)];

      if (source) {
        renderSources(el.lookupResults, [source]);
      }
    });
  });

  scrollChatToBottom();
}

function render() {
  renderConversations();
  renderMessages();
}

function parseStreamLine(line) {
  const raw = line.trim();
  if (!raw) return null;

  // Hỗ trợ cả JSONL thuần và SSE dạng "data: {...}"
  const jsonText = raw.startsWith("data:")
    ? raw.slice(5).trim()
    : raw;

  if (!jsonText || jsonText === "[DONE]") {
    return { type: "done" };
  }

  return JSON.parse(jsonText);
}

function getStreamEventText(event) {
  return (
    event.text ??
    event.delta ??
    event.content ??
    event.answer_delta ??
    event.message ??
    ""
  );
}

async function sendQuestion(question) {
  const convo = activeConversation();
  if (!convo) return;

  const clean = question.trim();
  if (!clean) return;

  if (convo.title === "Cuộc trò chuyện mới") {
    convo.title = clean.slice(0, 50) + (clean.length > 50 ? "…" : "");
  }

  convo.messages.push({
    role: "user",
    content: clean,
    createdAt: new Date().toISOString(),
  });

  saveState();
  render();
  scrollChatToBottom();

  setStatus("Đang truy xuất...", "loading");

  if (el.sendBtn) {
    el.sendBtn.disabled = true;
  }

  let assistantMessage = null;

  try {
    const payload = {
      question: clean,
      k: Number(el.topKInput?.value || 5),
      strategy: el.strategySelect?.value || "auto",
      generate: el.generateToggle ? el.generateToggle.checked : true,

      // Giảm context để sinh nhanh hơn
      max_context_chars: 2200,

      // Backend QueryRequest đã bổ sung answer_style
      answer_style: "short",
    };

    const res = await fetch("/api/query/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    assistantMessage = {
      role: "assistant",
      content: "",
      sources: [],
      timings: {},
      createdAt: new Date().toISOString(),
    };

    convo.messages.push(assistantMessage);
    saveState();
    render();
    scrollChatToBottom();

    const messagesEl = getMessagesElement();
    const assistantBubbles = messagesEl
      ? messagesEl.querySelectorAll(".message.assistant .bubble")
      : [];

    const currentBubble = assistantBubbles[assistantBubbles.length - 1];

    if (!currentBubble) {
      throw new Error("Không tìm thấy bubble assistant để stream.");
    }

    currentBubble.textContent = "";

    // Nếu backend trả JSON thường thay vì stream, vẫn xử lý được
    const contentType = res.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
      const data = await res.json();

      assistantMessage.content =
        data.answer ||
        data.response ||
        data.message ||
        "";

      assistantMessage.sources =
        data.documents ||
        data.sources ||
        data.docs ||
        [];

      assistantMessage.timings = data.timings || {
        total_ms: data.latency_ms,
      };

      currentBubble.textContent = assistantMessage.content;

      renderSources(el.latestSources, assistantMessage.sources);

      saveState();
      render();
      scrollChatToBottom();

      const totalMs = assistantMessage.timings.total_ms;
      setStatus(totalMs ? `Xong · ${formatMs(totalMs)}` : "Sẵn sàng");
      return;
    }

    if (!res.body) {
      throw new Error("Trình duyệt không hỗ trợ ReadableStream.");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let buffer = "";
    let assistantText = "";
    let pendingText = "";
    let renderScheduled = false;
    let chunkCount = 0;
    let streamCompleted = false;

    function scheduleBubbleUpdate(text) {
      pendingText = text;

      if (renderScheduled) return;

      renderScheduled = true;

      requestAnimationFrame(() => {
        currentBubble.textContent = pendingText;
        renderScheduled = false;
      });
    }

    async function handleStreamEvent(event) {
      if (!event) return;

      // Khi debug cần thì mở dòng này:
      // console.log("STREAM EVENT:", event);

      const type = event.type || event.event || "";

      if (type === "sources" || type === "retrieval") {
        assistantMessage.sources =
          event.documents ||
          event.sources ||
          event.docs ||
          [];

        renderSources(el.latestSources, assistantMessage.sources);

        setStatus(
          event.latency_ms != null
            ? `Retrieval xong · ${formatMs(event.latency_ms)}`
            : "Đang sinh câu trả lời...",
          "loading"
        );

        // Tuyệt đối không gọi render() ở đây.
        return;
      }

      if (
        type === "delta" ||
        type === "token" ||
        type === "answer_delta" ||
        type === "message_delta"
      ) {
        const text = getStreamEventText(event);
        if (!text) return;

        assistantText += text;
        assistantMessage.content = assistantText;
        chunkCount += 1;

        // Chỉ update bubble hiện tại, không rebuild toàn bộ DOM.
        scheduleBubbleUpdate(assistantText);

        // Scroll thưa để tránh layout thrashing.
        if (chunkCount % 12 === 0) {
          scrollChatToBottom();
        }

        return;
      }

      if (
        type === "done" ||
        type === "final" ||
        type === "completed"
      ) {
        if (event.answer && !assistantText) {
          assistantText = event.answer;
        }

        assistantMessage.content =
          assistantText || assistantMessage.content || "Đã hoàn thành.";

        assistantMessage.timings = event.timings || {};
        streamCompleted = true;

        currentBubble.textContent = assistantMessage.content;

        saveState();

        // Chỉ render 1 lần ở cuối để hiển thị source chips, timing, markdown nhẹ.
        render();
        scrollChatToBottom();

        const totalMs = assistantMessage.timings.total_ms;
        setStatus(totalMs ? `Xong · ${formatMs(totalMs)}` : "Sẵn sàng");

        return;
      }

      if (type === "error") {
        assistantText += `\n\n[Lỗi] ${event.message || "Không rõ lỗi."}`;
        assistantMessage.content = assistantText;

        currentBubble.textContent = assistantText;

        saveState();
        render();
        scrollChatToBottom();

        setStatus("Lỗi", "error");
        streamCompleted = true;
        return;
      }

      // Fallback nếu backend trả event có answer nhưng không có type chuẩn.
      if (event.answer && !assistantText) {
        assistantText = event.answer;
        assistantMessage.content = assistantText;
        currentBubble.textContent = assistantText;
      }
    }

    while (true) {
      const { value, done } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;

        try {
          const event = parseStreamLine(line);
          await handleStreamEvent(event);
        } catch (err) {
          console.warn("Không parse được stream line:", line, err);
        }
      }
    }

    // Xử lý nốt phần buffer cuối nếu có
    if (buffer.trim()) {
      try {
        const event = parseStreamLine(buffer);
        await handleStreamEvent(event);
      } catch (err) {
        console.warn("Không parse được stream buffer cuối:", buffer, err);
      }
    }

    // Nếu backend không gửi done, vẫn lưu và render cuối.
    if (!streamCompleted) {
      assistantMessage.content = assistantText || assistantMessage.content;

      saveState();
      render();
      scrollChatToBottom();

      setStatus("Sẵn sàng");
    }
  } catch (err) {
    if (assistantMessage) {
      assistantMessage.content += `\n\n[Lỗi khi gọi API stream] ${err.message}`;
      assistantMessage.sources = assistantMessage.sources || [];
    } else {
      convo.messages.push({
        role: "assistant",
        content: `Lỗi khi gọi API stream: ${err.message}`,
        sources: [],
        createdAt: new Date().toISOString(),
      });
    }

    saveState();
    render();
    scrollChatToBottom();

    setStatus("Lỗi", "error");
  } finally {
    if (el.sendBtn) {
      el.sendBtn.disabled = false;
    }
  }
}

async function lookupKeyword(keyword) {
  const clean = keyword.trim();

  if (!clean || clean.length < 2) return;

  if (el.keywordInput) {
    el.keywordInput.value = clean;
  }

  if (el.selectionHint) {
    el.selectionHint.textContent = `Đang tra cứu: “${clean}”`;
  }

  try {
    const res = await fetch("/api/keyword/lookup", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        keyword: clean,
        k: 5,
        strategy: "hybrid_rerank",
      }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();

    if (el.selectionHint) {
      el.selectionHint.textContent = `Từ khóa: “${data.keyword}” · ${formatMs(
        data.latency_ms
      )}`;
    }

    renderSources(el.lookupResults, data.documents || []);
  } catch (err) {
    if (el.selectionHint) {
      el.selectionHint.textContent = `Lỗi tra cứu keyword: ${err.message}`;
    }
  }
}

function handleSelection() {
  const selected = window.getSelection().toString().trim();

  if (!selected || selected.length < 2 || selected.length > 120) return;
  if (selected === lastSelection) return;

  lastSelection = selected;

  clearTimeout(selectionTimer);

  selectionTimer = setTimeout(() => {
    lookupKeyword(selected);
  }, 450);
}

if (el.newChatBtn) {
  el.newChatBtn.addEventListener("click", () => createConversation());
}

if (el.chatForm) {
  el.chatForm.addEventListener("submit", (e) => {
    e.preventDefault();

    const question = el.questionInput?.value || "";

    if (el.questionInput) {
      el.questionInput.value = "";
    }

    sendQuestion(question);
  });
}

if (el.questionInput) {
  el.questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      el.chatForm?.requestSubmit();
    }
  });
}

if (el.lookupBtn) {
  el.lookupBtn.addEventListener("click", () => {
    lookupKeyword(el.keywordInput?.value || "");
  });
}

if (el.keywordInput) {
  el.keywordInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      lookupKeyword(el.keywordInput.value);
    }
  });
}

document.addEventListener("mouseup", handleSelection);
document.addEventListener("keyup", handleSelection);

if (!activeId || conversations.length === 0) {
  createConversation({ shouldRender: false });
}

render();
scrollChatToBottom();