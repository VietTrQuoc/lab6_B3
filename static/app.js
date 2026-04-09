/**
 * VinFast Warranty AI Agent — Frontend Application
 *
 * Handles:
 * - Vehicle selection and display
 * - Chat messaging with the AI agent
 * - Booking flow with countdown timer (PENDING → CONFIRMED)
 * - Markdown rendering for agent responses
 */

// ─── State ────────────────────────────────────────────────────────
const state = {
  selectedVehicleId: null,
  messages: [],           // conversation history sent to API
  isLoading: false,
  pendingBookings: {},    // bookingId -> { interval, expiresAt }
};

// ─── DOM References ───────────────────────────────────────────────
const vehicleList = document.getElementById('vehicleList');
const messagesContainer = document.getElementById('messagesContainer');
const welcomeContainer = document.getElementById('welcomeContainer');
const typingIndicator = document.getElementById('typingIndicator');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const clearChatBtn = document.getElementById('clearChatBtn');

// ─── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadVehicles();
  setupEventListeners();
});

function setupEventListeners() {
  sendBtn.addEventListener('click', sendMessage);

  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-resize textarea
  messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
  });

  clearChatBtn.addEventListener('click', clearChat);

  // Quick action buttons
  document.querySelectorAll('.quick-action-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const msg = btn.dataset.message;
      if (msg) {
        messageInput.value = msg;
        sendMessage();
      }
    });
  });
}

// ─── Vehicles ─────────────────────────────────────────────────────
async function loadVehicles() {
  try {
    const res = await fetch('/api/vehicles');
    const vehicles = await res.json();
    renderVehicleList(vehicles);
  } catch (err) {
    vehicleList.innerHTML = `
      <div class="empty-state">
        <div class="icon">⚠️</div>
        <div>Không thể tải danh sách xe</div>
      </div>`;
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

  vehicleList.innerHTML = vehicles.map(v => {
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

function selectVehicle(vehicleId) {
  state.selectedVehicleId = vehicleId;

  // Update UI
  document.querySelectorAll('.vehicle-card').forEach(card => {
    card.classList.remove('active', 'just-selected');
    if (card.dataset.id === vehicleId) {
      card.classList.add('active', 'just-selected');
      setTimeout(() => card.classList.remove('just-selected'), 600);
    }
  });
}

// ─── Chat ─────────────────────────────────────────────────────────
async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text || state.isLoading) return;

  // Hide welcome
  if (welcomeContainer) {
    welcomeContainer.style.display = 'none';
  }

  // Add user message to UI
  appendMessage('user', text);

  // Add to message history
  state.messages.push({ role: 'user', content: text });

  // Clear input
  messageInput.value = '';
  messageInput.style.height = 'auto';

  // Show typing indicator
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

    // Hide typing
    typingIndicator.classList.remove('visible');

    // Add assistant reply
    appendMessage('assistant', data.reply, data.tool_calls_log);

    // Add to history
    state.messages.push({ role: 'assistant', content: data.reply });

    // Handle booking if present
    if (data.booking) {
      appendBookingCard(data.booking);
    }

  } catch (err) {
    typingIndicator.classList.remove('visible');
    appendMessage('assistant', `⚠️ Xin lỗi, đã có lỗi xảy ra: ${err.message}. Vui lòng thử lại hoặc liên hệ hotline **1900 23 23 89**.`);
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
    const toolItems = toolCallsLog.map(tc =>
      `<div class="tool-call-item">
        <span class="tool-name">${tc.tool}</span>(${JSON.stringify(tc.arguments)})
      </div>`
    ).join('');

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

  // Insert before typing indicator
  messagesContainer.insertBefore(msgDiv, typingIndicator);
  scrollToBottom();
}

// ─── Booking Card ─────────────────────────────────────────────────
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
      <button class="confirm-btn" id="confirm-btn-${booking.booking_id}"
              onclick="confirmBooking('${booking.booking_id}')">
        ✓ Xác nhận
      </button>
    </div>`;

  messagesContainer.insertBefore(cardDiv, typingIndicator);
  scrollToBottom();

  // Start countdown
  startCountdown(booking.booking_id, booking.ttl_seconds || 300);
}

function startCountdown(bookingId, totalSeconds) {
  let remaining = totalSeconds;

  const interval = setInterval(() => {
    remaining--;

    const timerValue = document.getElementById(`timer-value-${bookingId}`);
    const timerDiv = document.getElementById(`timer-${bookingId}`);
    const card = document.getElementById(`booking-${bookingId}`);

    if (!timerValue || remaining <= 0) {
      clearInterval(interval);
      // Expired
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

    // Urgent styling when < 60 seconds
    if (remaining < 60 && timerDiv) {
      timerDiv.classList.add('urgent');
    }
  }, 1000);

  state.pendingBookings[bookingId] = { interval, expiresAt: Date.now() + totalSeconds * 1000 };
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

    const booking = await res.json();

    // Stop countdown
    if (state.pendingBookings[bookingId]) {
      clearInterval(state.pendingBookings[bookingId].interval);
      delete state.pendingBookings[bookingId];
    }

    // Update card UI
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

    // Add confirmation message
    appendMessage('assistant', `✅ **Lịch hẹn ${bookingId} đã được xác nhận thành công!**\n\nAnh/chị vui lòng đến xưởng dịch vụ đúng giờ hẹn. Nếu cần thay đổi, vui lòng liên hệ hotline **1900 23 23 89**.`);

  } catch (err) {
    if (btn) {
      btn.disabled = false;
      btn.textContent = '✓ Xác nhận';
    }
    appendMessage('assistant', `⚠️ Không thể xác nhận lịch hẹn: ${err.message}. Có thể slot đã hết hạn. Anh/chị có muốn đặt lại không?`);
  }
}

// ─── Utilities ────────────────────────────────────────────────────
function clearChat() {
  state.messages = [];
  // Clean up countdowns
  Object.values(state.pendingBookings).forEach(pb => clearInterval(pb.interval));
  state.pendingBookings = {};

  // Clear messages but keep welcome and typing indicator
  const children = Array.from(messagesContainer.children);
  children.forEach(child => {
    if (child.id !== 'welcomeContainer' && child.id !== 'typingIndicator') {
      child.remove();
    }
  });

  // Show welcome
  if (welcomeContainer) {
    welcomeContainer.style.display = 'flex';
  }
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

/**
 * Simple Markdown renderer for agent responses.
 * Handles: bold, italic, headers, lists, code, tables, links, line breaks.
 */
function renderMarkdown(text) {
  if (!text) return '';

  let html = text;

  // Escape HTML first (except tags we'll create)
  html = html
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Code blocks (```)
  html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

  // Inline code (`)
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Headers (### -> h3)
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^# (.+)$/gm, '<h3>$1</h3>');

  // Bold (**text**)
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Italic (*text*)
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Tables
  html = html.replace(/^\|(.+)\|$/gm, (match) => {
    const cells = match.split('|').filter(c => c.trim());
    // Check if this is a separator row
    if (cells.every(c => /^[\s-:]+$/.test(c))) return '';
    const isHeader = false; // We'll handle this differently
    const cellHtml = cells.map(c => `<td>${c.trim()}</td>`).join('');
    return `<tr>${cellHtml}</tr>`;
  });

  // Wrap table rows
  html = html.replace(/((?:<tr>.*<\/tr>\n?)+)/g, '<table>$1</table>');

  // Unordered list items
  html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

  // Ordered list items
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  // Links [text](url)
  html = html.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank">$1</a>');

  // Line breaks -> paragraphs
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');

  // Wrap in paragraph if not already wrapped
  if (!html.startsWith('<')) {
    html = `<p>${html}</p>`;
  }

  return html;
}
