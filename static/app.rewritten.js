/**
 * VinFast Warranty AI Agent - Frontend Application
 *
 * Handles:
 * - Vehicle selection and display
 * - Chat messaging with the AI agent
 * - Booking flow with countdown timer (PENDING -> CONFIRMED)
 * - Markdown rendering for agent responses
 * - Session persistence in localStorage
 */

const state = {
  selectedVehicleId: null,
  messages: [],
  isLoading: false,
  pendingBookings: {},
};

const STORAGE_KEYS = {
  messages: 'vinbot-chat-messages',
  selectedVehicleId: 'vinbot-selected-vehicle',
};

const vehicleList = document.getElementById('vehicleList');
const messagesContainer = document.getElementById('messagesContainer');
const welcomeContainer = document.getElementById('welcomeContainer');
const typingIndicator = document.getElementById('typingIndicator');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const clearChatBtn = document.getElementById('clearChatBtn');

document.addEventListener('DOMContentLoaded', async () => {
  await loadVehicles();
  setupEventListeners();
  restoreSession();
});

function setupEventListeners() {
  sendBtn.addEventListener('click', sendMessage);

  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = `${Math.min(messageInput.scrollHeight, 120)}px`;
  });

  clearChatBtn.addEventListener('click', clearChat);

  document.querySelectorAll('.quick-action-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const msg = btn.dataset.message;
      if (msg) {
        messageInput.value = msg;
        sendMessage();
      }
    });
  });
}

function saveSession() {
  localStorage.setItem(STORAGE_KEYS.messages, JSON.stringify(state.messages));

  if (state.selectedVehicleId) {
    localStorage.setItem(STORAGE_KEYS.selectedVehicleId, state.selectedVehicleId);
  } else {
    localStorage.removeItem(STORAGE_KEYS.selectedVehicleId);
  }
}

function restoreSession() {
  const storedMessages = localStorage.getItem(STORAGE_KEYS.messages);
  const storedVehicleId = localStorage.getItem(STORAGE_KEYS.selectedVehicleId);

  if (storedVehicleId) {
    selectVehicle(storedVehicleId, { persist: false });
  }

  if (!storedMessages) {
    return;
  }

  try {
    const parsedMessages = JSON.parse(storedMessages);
    if (!Array.isArray(parsedMessages)) {
      return;
    }

    state.messages = parsedMessages.filter(
      (msg) => msg && (msg.role === 'user' || msg.role === 'assistant') && typeof msg.content === 'string'
    );

    if (!state.messages.length) {
      return;
    }

    welcomeContainer.style.display = 'none';
    state.messages.forEach((msg) => appendMessage(msg.role, msg.content));
    scrollToBottom();
  } catch {
    localStorage.removeItem(STORAGE_KEYS.messages);
  }
}

async function loadVehicles() {
  try {
    const res = await fetch('/api/vehicles');
    const vehicles = await res.json();
    renderVehicleList(vehicles);
    return vehicles;
  } catch {
    vehicleList.innerHTML = `
      <div class="empty-state">
        <div class="icon">⚠️</div>
        <div>Không thể tải danh sách xe</div>
      </div>`;
    return [];
  }
}

function renderVehicleList(vehicles) {
  if (!vehicles.length) {
    vehicleList.innerHTML = `
      <div class="empty-state">
        <div class="icon">🏍️</div>
        <div>Chưa có xe nào</div>
      </div>`;
    return;
  }

  vehicleList.innerHTML = vehicles.map((v) => {
    let statusClass = 'good';
    if (v.error_count > 0 && v.battery_soh_percent < 75) statusClass = 'error';
    else if (v.error_count > 0 || v.battery_soh_percent < 85) statusClass = 'warning';

    return `
      <div class="vehicle-card" data-id="${v.id}" onclick="selectVehicle('${v.id}')">
        <div class="vehicle-card-header">
          <span class="vehicle-model">${v.model}</span>
          <span class="vehicle-status-dot ${statusClass}"></span>
        </div>
        <div class="vehicle-vin">${v.vin}</div>
        <div class="vehicle-stats">
          <span class="vehicle-stat">
            <span class="icon">📏</span>
            <span class="value">${v.odo_km.toLocaleString()} km</span>
          </span>
          <span class="vehicle-stat">
            <span class="icon">🔋</span>
            <span class="value">${v.battery_soh_percent}%</span>
          </span>
          ${v.error_count > 0 ? `
          <span class="vehicle-stat">
            <span class="icon">⚠️</span>
            <span class="value">${v.error_count} lỗi</span>
          </span>` : ''}
        </div>
      </div>`;
  }).join('');
}

function selectVehicle(vehicleId, options = {}) {
  const { persist = true } = options;
  state.selectedVehicleId = vehicleId;

  document.querySelectorAll('.vehicle-card').forEach((card) => {
    card.classList.remove('active', 'just-selected');
    if (card.dataset.id === vehicleId) {
      card.classList.add('active', 'just-selected');
      setTimeout(() => card.classList.remove('just-selected'), 600);
    }
  });

  if (persist) {
    saveSession();
  }
}

async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text || state.isLoading) return;

  welcomeContainer.style.display = 'none';
  appendMessage('user', text);

  state.messages.push({ role: 'user', content: text });
  saveSession();

  messageInput.value = '';
  messageInput.style.height = 'auto';

  state.isLoading = true;
  sendBtn.disabled = true;
  typingIndicator.classList.add('visible');
  scrollToBottom();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: state.messages,
        selected_vehicle_id: state.selectedVehicleId,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Server error');
    }

    const data = await res.json();
    typingIndicator.classList.remove('visible');

    appendMessage('assistant', data.reply, data.tool_calls_log);
    state.messages.push({ role: 'assistant', content: data.reply });
    saveSession();

    if (data.booking) {
      appendBookingCard(data.booking);
    }
  } catch (err) {
    typingIndicator.classList.remove('visible');
    const errorReply = `⚠️ Xin lỗi, đã có lỗi xảy ra: ${err.message}. Vui lòng thử lại hoặc liên hệ hotline **1900 23 23 89**.`;
    appendMessage('assistant', errorReply);
    state.messages.push({ role: 'assistant', content: errorReply });
    saveSession();
  } finally {
    state.isLoading = false;
    sendBtn.disabled = false;
    scrollToBottom();
  }
}

function appendMessage(role, content, toolCallsLog = []) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;

  const avatarIcon = role === 'user' ? 'NA' : '🤖';
  const renderedContent = role === 'assistant' ? renderMarkdown(content) : escapeHtml(content);

  let toolLogHtml = '';
  if (toolCallsLog && toolCallsLog.length > 0) {
    const toolItems = toolCallsLog.map((tc) => `
      <div class="tool-call-item">
        <span class="tool-name">${tc.tool}</span>(${JSON.stringify(tc.arguments)})
      </div>`).join('');

    toolLogHtml = `
      <div class="tool-log">
        <button class="tool-log-toggle" onclick="this.nextElementSibling.classList.toggle('visible')">
          🔧 ${toolCallsLog.length} tool calls ▾
        </button>
        <div class="tool-log-content">${toolItems}</div>
      </div>`;
  }

  msgDiv.innerHTML = `
    <div class="message-avatar">${avatarIcon}</div>
    <div class="message-bubble">
      ${renderedContent}
      ${toolLogHtml}
    </div>`;

  messagesContainer.insertBefore(msgDiv, typingIndicator);
  scrollToBottom();
}

function appendBookingCard(booking) {
  const cardDiv = document.createElement('div');
  cardDiv.className = 'booking-card';
  cardDiv.id = `booking-${booking.booking_id}`;

  cardDiv.innerHTML = `
    <div class="booking-header">
      <span class="booking-id">${booking.booking_id}</span>
      <span class="booking-status pending" id="status-${booking.booking_id}">PENDING</span>
    </div>
    <div class="booking-details">
      <div class="booking-detail-item">
        <div class="label">Xưởng dịch vụ</div>
        <div class="value">${booking.center_name}</div>
      </div>
      <div class="booking-detail-item">
        <div class="label">Ngày hẹn</div>
        <div class="value">${booking.booking_date}</div>
      </div>
      <div class="booking-detail-item">
        <div class="label">Giờ hẹn</div>
        <div class="value">${booking.time_slot}</div>
      </div>
      <div class="booking-detail-item">
        <div class="label">Xe</div>
        <div class="value">${booking.vin_number}</div>
      </div>
    </div>
    <div class="countdown-section" id="countdown-${booking.booking_id}">
      <div class="countdown-timer" id="timer-${booking.booking_id}">
        <span class="timer-icon">⏱️</span>
        <span>Giữ chỗ còn </span>
        <span class="timer-value" id="timer-value-${booking.booking_id}">5:00</span>
      </div>
      <button class="confirm-btn" id="confirm-btn-${booking.booking_id}" onclick="confirmBooking('${booking.booking_id}')">
        ✓ Xác nhận
      </button>
    </div>`;

  messagesContainer.insertBefore(cardDiv, typingIndicator);
  scrollToBottom();
  startCountdown(booking.booking_id, booking.ttl_seconds || 300);
}

function startCountdown(bookingId, totalSeconds) {
  let remaining = totalSeconds;

  const interval = setInterval(() => {
    remaining -= 1;

    const timerValue = document.getElementById(`timer-value-${bookingId}`);
    const timerDiv = document.getElementById(`timer-${bookingId}`);
    const card = document.getElementById(`booking-${bookingId}`);

    if (!timerValue || remaining <= 0) {
      clearInterval(interval);
      if (card) {
        card.classList.add('expired');
        const statusEl = document.getElementById(`status-${bookingId}`);
        if (statusEl) {
          statusEl.textContent = 'HẾT HẠN';
          statusEl.className = 'booking-status expired';
        }
        const countdownSection = document.getElementById(`countdown-${bookingId}`);
        if (countdownSection) {
          countdownSection.innerHTML = '<span style="color: var(--status-error); font-size: 13px;">⏰ Đã hết thời gian giữ chỗ. Vui lòng đặt lại.</span>';
        }
      }
      delete state.pendingBookings[bookingId];
      return;
    }

    const mins = Math.floor(remaining / 60);
    const secs = remaining % 60;
    timerValue.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;

    if (remaining < 60 && timerDiv) {
      timerDiv.classList.add('urgent');
    }
  }, 1000);

  state.pendingBookings[bookingId] = {
    interval,
    expiresAt: Date.now() + totalSeconds * 1000,
  };
}

async function fetchBookings() {
  const res = await fetch('/api/bookings');
  if (!res.ok) {
    throw new Error('Không thể tải danh sách lịch hẹn');
  }
  return res.json();
}

function upsertBookingsSummaryCard(bookings) {
  const existingCard = document.getElementById('bookings-summary-card');
  if (existingCard) {
    existingCard.remove();
  }

  const summaryCard = document.createElement('div');
  summaryCard.className = 'booking-card bookings-summary-card';
  summaryCard.id = 'bookings-summary-card';

  const rows = bookings.map((booking) => `
    <tr>
      <td>${booking.booking_id}</td>
      <td>${booking.vin_number}</td>
      <td>${booking.center_name}</td>
      <td>${booking.booking_date}</td>
      <td>${booking.time_slot}</td>
      <td><span class="booking-status ${String(booking.status || '').toLowerCase()}">${booking.status || 'N/A'}</span></td>
    </tr>`).join('');

  summaryCard.innerHTML = `
    <div class="booking-header">
      <span class="booking-id">Lịch hẹn đã đặt</span>
      <span class="booking-status confirmed">${bookings.length} lịch</span>
    </div>
    <div class="bookings-table-wrapper">
      <table class="bookings-table">
        <thead>
          <tr>
            <th>Mã</th>
            <th>Xe</th>
            <th>Xưởng</th>
            <th>Ngày</th>
            <th>Giờ</th>
            <th>Trạng thái</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;

  messagesContainer.insertBefore(summaryCard, typingIndicator);
  scrollToBottom();
}

async function confirmBooking(bookingId) {
  const btn = document.getElementById(`confirm-btn-${bookingId}`);
  if (btn) {
    btn.disabled = true;
    btn.textContent = '⏳ Đang xác nhận...';
  }

  try {
    const res = await fetch('/api/booking/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ booking_id: bookingId }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Confirmation failed');
    }

    await res.json();

    if (state.pendingBookings[bookingId]) {
      clearInterval(state.pendingBookings[bookingId].interval);
      delete state.pendingBookings[bookingId];
    }

    const card = document.getElementById(`booking-${bookingId}`);
    if (card) {
      card.classList.add('confirmed');
      const statusEl = document.getElementById(`status-${bookingId}`);
      if (statusEl) {
        statusEl.textContent = 'CONFIRMED';
        statusEl.className = 'booking-status confirmed';
      }
      const countdownSection = document.getElementById(`countdown-${bookingId}`);
      if (countdownSection) {
        countdownSection.innerHTML = '<span style="color: var(--status-confirmed); font-size: 13px;">✅ Đã xác nhận thành công! Hẹn gặp anh/chị tại xưởng dịch vụ.</span>';
      }
    }

    const confirmReply = `✅ **Lịch hẹn ${bookingId} đã được xác nhận thành công!**\n\nAnh/chị vui lòng đến xưởng dịch vụ đúng giờ hẹn. Nếu cần thay đổi, vui lòng liên hệ hotline **1900 23 23 89**.`;
    appendMessage('assistant', confirmReply);
    state.messages.push({ role: 'assistant', content: confirmReply });
    saveSession();

    const bookings = await fetchBookings();
    if (Array.isArray(bookings) && bookings.length) {
      upsertBookingsSummaryCard(bookings);
    }
  } catch (err) {
    if (btn) {
      btn.disabled = false;
      btn.textContent = '✓ Xác nhận';
    }

    const failedReply = `⚠️ Không thể xác nhận lịch hẹn: ${err.message}. Có thể slot đã hết hạn. Anh/chị có muốn đặt lại không?`;
    appendMessage('assistant', failedReply);
    state.messages.push({ role: 'assistant', content: failedReply });
    saveSession();
  }
}

function clearChat() {
  state.messages = [];
  localStorage.removeItem(STORAGE_KEYS.messages);

  Object.values(state.pendingBookings).forEach((pb) => clearInterval(pb.interval));
  state.pendingBookings = {};

  Array.from(messagesContainer.children).forEach((child) => {
    if (child.id !== 'welcomeContainer' && child.id !== 'typingIndicator') {
      child.remove();
    }
  });

  welcomeContainer.style.display = 'flex';
  messageInput.value = '';
  messageInput.style.height = 'auto';
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderMarkdown(text) {
  if (!text) return '';

  let html = text;

  html = html
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^# (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/^\|(.+)\|$/gm, (match) => {
    const cells = match.split('|').filter((c) => c.trim());
    if (cells.every((c) => /^[\s-:]+$/.test(c))) return '';
    const cellHtml = cells.map((c) => `<td>${c.trim()}</td>`).join('');
    return `<tr>${cellHtml}</tr>`;
  });
  html = html.replace(/((?:<tr>.*<\/tr>\n?)+)/g, '<table>$1</table>');
  html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  html = html.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank">$1</a>');
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');

  if (!html.startsWith('<')) {
    html = `<p>${html}</p>`;
  }

  return html;
}

window.selectVehicle = selectVehicle;
window.confirmBooking = confirmBooking;
