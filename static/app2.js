// /static/app2.js
const API_BASE = ""; // Базовый URL API (оставь "" если бек на том же origin)

// --- Auth helpers & UI ---
const TOKEN_KEY = "tanym_token";
const USER_EMAIL_KEY = "tanym_user_email";
const USER_ID_KEY = "tanym_user_id";

const systemMessageEl = document.getElementById("systemMessage");

function showSystemMessage(message, type = 'info') {
    if (!systemMessageEl) return;
    systemMessageEl.innerHTML = message;
    systemMessageEl.style.display = "block";

    // Простая стилизация
    systemMessageEl.style.backgroundColor = type === 'success' ? '#e6ffed' : type === 'error' ? '#fff0f6' : '#fffbe6';
    systemMessageEl.style.borderColor = type === 'success' ? '#b7eb8f' : type === 'error' ? '#ffadd2' : '#ffe58f';
}

function clearSystemMessage() {
    if (systemMessageEl) systemMessageEl.style.display = "none";
}

if (localStorage.getItem('verification_success') === 'true') {
    showSystemMessage("✅ Ваш аккаунт успешно подтвержден! Теперь вы можете войти.", 'success');
    localStorage.removeItem('verification_success');
}

function authFetch(url, opts = {}) {
  opts = opts || {};
  opts.headers = opts.headers || {};
  const token = localStorage.getItem(TOKEN_KEY);

  // Если токена нет — покажем модал и отклоним запрос с понятной ошибкой
  if (!token) {
    // UX: не спамим модал — только откроем (возможно пользователь уже видит его)
    try { openAuthModal("login"); } catch (e) {}
    return Promise.reject(new Error("not_authenticated"));
  }

  opts.headers["Authorization"] = "Bearer " + token;

  return fetch(url, opts).then(async resp => {
    if (resp.status === 401) {
      // Сервер вернул 401 — откроем модал и пробросим понятную ошибку
      try { openAuthModal("login"); } catch (e) {}
      // попытаемся прочитать тело для логирования, но не показываем пользователю JSON
      let body = null;
      try { body = await resp.text(); } catch (e) {}
      const err = new Error("unauthorized");
      err.details = body;
      throw err;
    }
    return resp;
  });
}


function saveAuth(info) {
  if (info.token) localStorage.setItem(TOKEN_KEY, info.token);
  if (info.user_id) localStorage.setItem(USER_ID_KEY, info.user_id);
  if (info.email) localStorage.setItem(USER_EMAIL_KEY, info.email);
}

function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_ID_KEY);
  localStorage.removeItem(USER_EMAIL_KEY);
}

// --- DOM elements (auth modal + header buttons) ---
const modalEl = document.getElementById("authModal");
const loginForm = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");

const switchToLoginBtn = document.getElementById("switchToLogin");
const switchToRegisterBtn = document.getElementById("switchToRegister");
const authCloseBtn = document.getElementById("authCloseBtn");

const authEmail = document.getElementById("authEmail");
const authPassword = document.getElementById("authPassword");

const authName = document.getElementById("authName");
const regEmail = document.getElementById("regEmail");
const regPassword = document.getElementById("regPassword");

const authError = document.getElementById("authError");
const regError = document.getElementById("regError");

const authSubmitBtn = document.getElementById("authSubmitBtn");
const authCancelBtn = document.getElementById("authCancelBtn");
const regSubmitBtn = document.getElementById("regSubmitBtn");
const regCancelBtn = document.getElementById("regCancelBtn");

const showLoginBtn = document.getElementById("showLoginBtn");
const showRegisterBtn = document.getElementById("showRegisterBtn");

const userInfo = document.getElementById("userInfo");
const anonArea = document.getElementById("anonArea");
const userEmailSpan = document.getElementById("userEmail");
const logoutBtn = document.getElementById("logoutBtn");

// --- Modal open/close and switch logic ---
function openAuthModal(mode = "login") {
  if (!modalEl) return;
  // Reset errors and fields
  authError && (authError.style.display = "none");
  regError && (regError.style.display = "none");
  clearSystemMessage();

  if (mode === "login") {
    loginForm && (loginForm.style.display = "");
    registerForm && (registerForm.style.display = "none");
    switchToLoginBtn && switchToLoginBtn.classList.add("active");
    switchToRegisterBtn && switchToRegisterBtn.classList.remove("active");
    setTimeout(()=> authEmail?.focus(), 30);
  } else {
    loginForm && (loginForm.style.display = "none");
    registerForm && (registerForm.style.display = "");
    switchToLoginBtn && switchToLoginBtn.classList.remove("active");
    switchToRegisterBtn && switchToRegisterBtn.classList.add("active");
    setTimeout(()=> authName?.focus(), 30);
  }
  modalEl.classList.add("open");
  modalEl.setAttribute("aria-hidden","false");
}

function closeAuthModal() {
  if (!modalEl) return;
  modalEl.classList.remove("open");
  modalEl.setAttribute("aria-hidden","true");
  // hide error panels
  authError && (authError.style.display = "none");
  regError && (regError.style.display = "none");
}

// Close modal on overlay click or Esc
if (modalEl) {
  modalEl.addEventListener("click", (e) => {
    if (e.target === modalEl) closeAuthModal();
  });
}
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && modalEl && modalEl.classList.contains("open")) closeAuthModal();
});

// wire header buttons
showLoginBtn?.addEventListener("click", () => openAuthModal("login"));
showRegisterBtn?.addEventListener("click", () => openAuthModal("register"));
authCloseBtn?.addEventListener("click", closeAuthModal);

// switch tabs inside modal
switchToLoginBtn?.addEventListener("click", () => openAuthModal("login"));
switchToRegisterBtn?.addEventListener("click", () => openAuthModal("register"));

// cancel buttons
authCancelBtn?.addEventListener("click", closeAuthModal);
regCancelBtn?.addEventListener("click", closeAuthModal);

// expose for other scripts (old code may call openAuthModal)
window.openAuthModal = openAuthModal;
window.closeAuthModal = closeAuthModal;

// --- Auth action handlers ---
// Note: We provide the same endpoints as before (/login, /register) and reuse saveAuth/updateAuthUI
async function performLogin(email, pass) {
  authError && (authError.style.display = "none");
  if (!email || !pass) {
    if (authError) {
      authError.textContent = "Заполните email и пароль";
      authError.style.display = "block";
    }
    return;
  }
  authSubmitBtn && (authSubmitBtn.disabled = true);
  const orig = authSubmitBtn ? authSubmitBtn.innerHTML : null;
  if (authSubmitBtn) authSubmitBtn.innerHTML = "Подождите...";

  try {
    const form = new FormData();
    form.append("email", email);
    form.append("password", pass);
    const resp = await fetch(`${API_BASE}/login`, { method: "POST", body: form });
    const data = await resp.json().catch(()=>null);
    if (!resp.ok) {
        if (resp.status === 403) {
            closeAuthModal();
            showSystemMessage(data?.detail || "Аккаунт не подтвержден. Проверьте ваш email.", 'error');
            return;
        }
        throw new Error(data?.detail || data?.error || JSON.stringify(data) || `HTTP ${resp.status}`);
    }
    saveAuth({ token: data.token, user_id: data.user_id, email });
    closeAuthModal();
    updateAuthUI();
    clearSystemMessage();
    await updateStats().catch(()=>{});
    await loadUploadedFiles().catch(()=>{});
  } catch (err) {
    console.error("Login error:", err);
    if (authError) {
      authError.textContent = err.message || String(err);
      authError.style.display = "block";
    }
  } finally {
    if (authSubmitBtn) {
      authSubmitBtn.disabled = false;
      authSubmitBtn.innerHTML = orig || "Войти";
    }
  }
}

async function performRegister(name, email, pass) {
  regError && (regError.style.display = "none");
  if (!name || !email || !pass) {
    if (regError) {
      regError.textContent = "Заполните все поля";
      regError.style.display = "block";
    }
    return;
  }

  regSubmitBtn && (regSubmitBtn.disabled = true);
  const orig = regSubmitBtn ? regSubmitBtn.innerHTML : null;
  if (regSubmitBtn) regSubmitBtn.innerHTML = "Подождите...";

  try {
    const form = new FormData();
    form.append("name", name);
    form.append("email", email);
    form.append("password", pass);
    const resp = await fetch(`${API_BASE}/register`, { method: "POST", body: form });
    const data = await resp.json().catch(()=>null);
    if (!resp.ok) throw new Error(data?.detail || data?.error || JSON.stringify(data) || `HTTP ${resp.status}`);
    closeAuthModal();
    showSystemMessage(data.message || "Регистрация успешна. Можете войти.", 'success');
  } catch (err) {
    console.error("Register error:", err);
    if (regError) {
      regError.textContent = err.message || String(err);
      regError.style.display = "block";
    }
  } finally {
    if (regSubmitBtn) {
      regSubmitBtn.disabled = false;
      regSubmitBtn.innerHTML = orig || "Зарегистрироваться";
    }
  }
}

// bind modal submit buttons
authSubmitBtn?.addEventListener("click", () => {
  const email = authEmail?.value?.trim() || "";
  const pass = authPassword?.value || "";
  performLogin(email, pass);
});
regSubmitBtn?.addEventListener("click", () => {
  const name = authName?.value?.trim() || "";
  const email = regEmail?.value?.trim() || "";
  const pass = regPassword?.value || "";
  performRegister(name, email, pass);
});

// logout
logoutBtn?.addEventListener("click", () => {
  clearAuth();
  updateAuthUI();
  loadUploadedFiles().catch(()=>{});
  updateStats().catch(()=>{});
});

// update auth UI in header
function updateAuthUI() {
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) {
    anonArea && (anonArea.style.display = "none");
    userInfo && (userInfo.style.display = "flex");
    userEmailSpan && (userEmailSpan.textContent = localStorage.getItem(USER_EMAIL_KEY) || "User");
  } else {
    anonArea && (anonArea.style.display = "flex");
    userInfo && (userInfo.style.display = "none");
  }
}

// --- main variables from original script ---
let lastQuestion = "";
let lastAnswer = "";
let lastFeedbackKey = null; // локально — предотвращение дублей в одной сессии

// --- 1. Tab switching ---
document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

// --- 2. Rating display update ---
document.getElementById('rating')?.addEventListener('input', e=>{
  document.getElementById('ratingVal').textContent = e.target.value;
});

// --- 3. Q&A Logic (Ask Button) ---
const askBtn = document.getElementById('askBtn');
const clearBtn = document.getElementById('clearBtn');

askBtn?.addEventListener('click', async ()=>{
  const q = document.getElementById('q').value.trim();
  const show = document.getElementById('showSimilar').checked;
  const ansEl = document.getElementById('ans');
  const srcEl = document.getElementById('sources');
  const metaEl = document.getElementById('ansMetadata');

  if(!q){
    ansEl.value = "❌ Введите вопрос.";
    return;
  }

  try{
    askBtn.disabled = true;
    askBtn.innerHTML = 'Думаю... <span class="spinner"></span>';

    const fd = new FormData();
    fd.append("question", q);
    fd.append("show_articles", String(show));

    // use authFetch to include token if present
    const resp = await authFetch(`${API_BASE}/ask`, { method:"POST", body: fd });
    if(!resp.ok) {
      const txt = await resp.text().catch(()=>null);
      throw new Error(`HTTP ${resp.status} ${txt || ''}`);
    }
    const data = await resp.json();

    ansEl.value = data.answer || "";

    // Metadata
    if(data.metadata){
      const m = data.metadata;
      metaEl.innerHTML = `
        <span class="pill">Найдено: ${m.found_results}</span>
        <span class="pill">Релевантных: ${m.relevant_results}</span>
        <span class="pill">Использовано: ${m.used_results}</span>
        <span class="pill">Ср. релевантность: ${(m.avg_similarity * 100).toFixed(0)}%</span>
      `;
    } else {
      metaEl.innerHTML = "";
    }

    // Sources
    srcEl.innerHTML = "";
    if(data.sources_md){
      const lines = data.sources_md.split('\n');
      let html = '';

      for(let line of lines){
        if(line.startsWith('### ')){
          const match = line.match(/релевантность:\s*(\d+)%/);
          if(match){
            const pct = parseInt(match[1]);
            let badge = 'medium';
            if(pct >= 80) badge = 'high';
            else if(pct < 60) badge = 'low';
            html += `<h3>${line.replace('### ', '')}<span class="similarity-badge ${badge}">${pct}%</span></h3>`;
          } else {
            html += `<h3>${line.replace('### ', '')}</h3>`;
          }
        } else if(line.startsWith('## ')){
          html += `<h2>${line.replace('## ', '')}</h2>`;
        } else if(line.startsWith('**') && line.endsWith('**')){
          html += `<strong>${line.replace(/\*\*/g, '')}</strong><br>`;
        } else if(line.startsWith('_') && line.endsWith('_')){
          html += `<em style="color:var(--muted)">${line.replace(/_/g, '')}</em><br>`;
        } else if(line.startsWith('🔗 [')){
          const linkMatch = line.match(/\[([^\]]+)\]\(([^)]+)\)/);
          if(linkMatch){
            html += `<a href="${linkMatch[2]}" target="_blank" style="color:var(--accent)">${line}</a><br>`;
          }
        } else if(line.trim()){
          html += `${line}<br>`;
        }
      }

      srcEl.innerHTML = html || data.sources_md;
    } else {
      srcEl.innerHTML = '<div class="hint">Источники не найдены</div>';
    }

    lastQuestion = q;
    lastAnswer = data.answer || "";

  }catch(e){
    console.error(e);
    // Разбиение по типам ошибок: наша authFetch бросает not_authenticated/unauthorized
    const msgEl = document.getElementById('ans');
    if (!msgEl) return;

    if (e && (e.message === "not_authenticated" || e.message === "unauthorized")) {
      msgEl.value = "🔐 Войдите в платформу, чтобы задать вопрос.";
      // Дополнительно подсказка в источниках
      const srcEl = document.getElementById('sources');
      if (srcEl) srcEl.innerHTML = '<div class="hint">✳️ Для работы с приватной базой знаний требуется вход. Нажмите «Войти».<\/div>';
    } else {
      msgEl.value = "⚠️ Ошибка при запросе к API: " + (e.message || e);
    }
  }finally{
    askBtn.disabled = false;
    askBtn.textContent = "Задать вопрос";
  }
});

// ====== ГОЛОСОВОЙ ВВОД (Speech-to-Text) ======

const micBtn = document.getElementById("micBtn");
const voiceStatus = document.getElementById("voiceStatus");
let mediaRecorder = null;
let audioChunks = [];

function showVoiceStatus(message, isError = false) {
  if (!voiceStatus) return;
  voiceStatus.textContent = message;
  voiceStatus.style.display = "block";
  voiceStatus.style.color = isError ? "var(--red)" : "var(--accent)";

  if (!isError) {
    setTimeout(() => {
      voiceStatus.style.display = "none";
    }, 5000);
  }
}

micBtn?.addEventListener("click", async () => {
  // Если уже идет запись - останавливаем
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    micBtn.textContent = "⏳";
    micBtn.disabled = true;
    micBtn.title = "Обработка записи...";
    showVoiceStatus("⏳ Обработка записи...");
    return;
  }

  // Начинаем новую запись
  try {
    // Запрашиваем доступ к микрофону
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    // Создаем MediaRecorder
    const options = { mimeType: 'audio/webm' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
      options.mimeType = 'audio/ogg'; // Fallback для Safari
    }

    mediaRecorder = new MediaRecorder(stream, options);
    audioChunks = [];

    // Собираем аудио-чанки
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    // Когда запись остановлена - отправляем на сервер
    mediaRecorder.onstop = async () => {
      // Останавливаем микрофон
      stream.getTracks().forEach(track => track.stop());

      // Создаем blob из чанков
      const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });

      showVoiceStatus("🔄 Распознавание речи...");

      // Отправляем на сервер
      try {
        const formData = new FormData();
        const fileExtension = mediaRecorder.mimeType.includes('webm') ? 'webm' : 'ogg';
        formData.append('audio', audioBlob, `recording.${fileExtension}`);

        const resp = await authFetch(`${API_BASE}/voice/stt`, {
          method: 'POST',
          body: formData
        });

        if (!resp.ok) {
          const errorText = await resp.text().catch(() => '');
          throw new Error(errorText || `HTTP ${resp.status}`);
        }

        const data = await resp.json();

        if (data.ok && data.text) {
          // Вставляем распознанный текст в поле вопроса
          const qField = document.getElementById("q");
          qField.value = data.text;
          qField.focus();

          // Показываем уведомление с точностью
          const confPct = Math.round((data.confidence || 0.8) * 100);
          const previewText = data.text.length > 50 ? data.text.substring(0, 50) + '...' : data.text;
          showVoiceStatus(`✅ Распознано (${confPct}% уверенность): "${previewText}"`);

        } else {
          showVoiceStatus(data.message || "❌ Не удалось распознать речь. Попробуйте говорить четче.", true);
        }

      } catch (e) {
        console.error("STT Error:", e);

        if (e && (e.message === "not_authenticated" || e.message === "unauthorized")) {
          showVoiceStatus("🔐 Войдите в систему для использования голосового ввода.", true);
        } else {
          showVoiceStatus("❌ Ошибка распознавания: " + (e.message || "Неизвестная ошибка"), true);
        }
      } finally {
        micBtn.disabled = false;
        micBtn.textContent = "🎙️";
        micBtn.title = "Голосовой ввод: нажмите для записи";
        micBtn.classList.remove("recording");
      }
    };

    // Начинаем запись
    mediaRecorder.start();
    micBtn.textContent = "⏹️"; // Иконка остановки
    micBtn.title = "Идёт запись... Нажмите чтобы остановить";
    micBtn.classList.add("recording");
    showVoiceStatus("🔴 Запись... Говорите ваш вопрос (макс. 30 сек)");

    // Автоостановка через 30 секунд (защита от зависания)
    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        showVoiceStatus("⏱️ Достигнут лимит времени записи (30 сек)");
        mediaRecorder.stop();
      }
    }, 30000);

  } catch (e) {
    console.error("Microphone access error:", e);

    if (e.name === "NotAllowedError") {
      showVoiceStatus("❌ Доступ к микрофону запрещён. Разрешите доступ в настройках браузера.", true);
    } else if (e.name === "NotFoundError") {
      showVoiceStatus("❌ Микрофон не найден. Подключите микрофон и обновите страницу.", true);
    } else {
      showVoiceStatus("❌ Ошибка доступа к микрофону: " + e.message, true);
    }

    micBtn.textContent = "🎙️";
    micBtn.classList.remove("recording");
  }
});

// --- 4. Q&A Logic (Clear Button) ---
clearBtn?.addEventListener('click', ()=>{
  document.getElementById('q').value = "";
  document.getElementById('ans').value = "";
  document.getElementById('sources').innerHTML = '<div class="hint">Здесь будут показаны источники</div>';
  document.getElementById('ansMetadata').innerHTML = "";
  lastQuestion = "";
  lastAnswer = "";
});

// --- 5. Feedback Logic ---
const sendFbBtn = document.getElementById('sendFb');

sendFbBtn?.addEventListener('click', async ()=> {
  const rating = document.getElementById('rating').value;
  const comment = document.getElementById('comment').value;
  const corr = document.getElementById('corr').value.trim();
  const fbOut = document.getElementById('fbOut');

  if(!lastQuestion || !lastAnswer){
    fbOut.textContent = "❌ Сначала задайте вопрос";
    setTimeout(()=>fbOut.textContent="", 4000);
    return;
  }

  // локальная защита от одинаковых отправок подряд
  const feedbackKey = `${lastQuestion}|||${lastAnswer}|||${corr}|||${rating}`;
  if (feedbackKey === lastFeedbackKey) {
    fbOut.textContent = "⚠️ Похоже, вы уже отправляли этот отзыв (сессия). Подождите результат.";
    fbOut.style.color = "var(--yellow)";
    setTimeout(()=>fbOut.textContent="", 4000);
    return;
  }

  // UI: блокировка + сообщение о процессе
  sendFbBtn.disabled = true;
  const origBtnText = sendFbBtn.textContent;
  sendFbBtn.innerHTML = 'Отправка... <span class="spinner"></span>';
  fbOut.style.color = "var(--muted)";
  fbOut.textContent = "Проверка правильного ответа...";

  try {
    const fd = new FormData();
    fd.append("rating", rating);
    fd.append("comment", comment);
    fd.append("correct_answer", corr);
    fd.append("question", lastQuestion);
    fd.append("answer", lastAnswer);

    const resp = await authFetch(`${API_BASE}/feedback`, { method:"POST", body: fd });
    if(!resp.ok){
      const txt = await resp.text().catch(()=>null);
      throw new Error(`HTTP ${resp.status} ${txt || ''}`);
    }
    const data = await resp.json();

    // Показываем содержимое ответа от сервера — с маскированием технических причин
    let userMsg = data.msg || "OK";
    const rawReason = data.validation_reason || "";
    let friendlyReason = "";
    if (rawReason) {
      if (/llm/i.test(rawReason) || /недоступн/i.test(rawReason) || /embedd/i.test(rawReason)) {
        friendlyReason = " (Проверка выполнена автоматически.)";
      } else {
        friendlyReason = ` (${rawReason})`;
      }
    }

    fbOut.innerHTML = userMsg + (friendlyReason ? ` <span style="color:var(--muted)">${friendlyReason}</span>` : "");
    fbOut.style.color = data.ok ? "var(--green)" : "var(--red)";

    if (data.correct_answer_saved !== undefined) {
      if (data.correct_answer_saved) {
        fbOut.innerHTML += ' <strong style="color:var(--green)">✅ Правильный ответ сохранён</strong>';
      } else if (data.validation_reason) {
        fbOut.innerHTML += ' <span style="color:var(--yellow)">⚠️ ' + (data.validation_reason && !/llm/i.test(data.validation_reason) ? data.validation_reason : 'Правильный ответ не сохранился') + '</span>';
      }
    }

    lastFeedbackKey = feedbackKey;

    if(data.ok){
      document.getElementById('rating').value = 5;
      document.getElementById('ratingVal').textContent = 5;
      document.getElementById('comment').value = "";
      document.getElementById('corr').value = "";
    }

  } catch(e) {
    console.error(e);
    fbOut.textContent = "⚠️ Ошибка отправки: " + (e.message || e);
    fbOut.style.color = "var(--red)";
  } finally {
    sendFbBtn.disabled = false;
    sendFbBtn.textContent = origBtnText || "Отправить отзыв";
    setTimeout(()=>fbOut.textContent="", 8000);
  }
});

// --- 6. Document Upload / Indexing Logic ---
const drop = document.getElementById('drop');
const fileInput = document.getElementById('fileInput');
const indexBtn = document.getElementById('indexBtn');
const fileListEl = document.getElementById('fileList');
const idxOut = document.getElementById('idxOut');
let uploadedFiles = [];

function showNames(){
  const names = uploadedFiles.map(f=>f.name).join(", ");
  fileListEl.textContent = uploadedFiles.length
    ? `📚 Готово к загрузке: ${names}`
    : "";
}

// Drag & Drop event handlers
['dragenter','dragover'].forEach(ev=>{
  drop?.addEventListener(ev, e=>{ e.preventDefault(); drop.classList.add('drag');});
});

['dragleave','drop'].forEach(ev=>{
  drop?.addEventListener(ev, e=>{ e.preventDefault(); drop.classList.remove('drag');});
});

drop?.addEventListener('drop', e=>{
  uploadedFiles = uploadedFiles.concat([...e.dataTransfer.files]);
  showNames();
});

drop?.addEventListener('click', ()=> fileInput?.click());

fileInput?.addEventListener('change', e=>{
  uploadedFiles = uploadedFiles.concat([...e.target.files]);
  showNames();
  fileInput.value = null;
});

indexBtn?.addEventListener('click', async ()=>{
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) {
    openAuthModal("login");
    return;
  }

  const sourceTag = document.getElementById('sourceTag').value;

  if(!uploadedFiles.length){
    idxOut.innerHTML = '<div class="msg warn">⚠️ Выберите файлы</div>';
    return;
  }

  indexBtn.disabled = true;
  indexBtn.innerHTML = 'Индексация... <span class="spinner"></span>';
  idxOut.innerHTML = "";

  try{
    const fd = new FormData();
    uploadedFiles.forEach(f=>fd.append("files", f));
    fd.append("source_tag", sourceTag);

    const resp = await authFetch(`${API_BASE}/ingest`, { method:"POST", body: fd });
    if(!resp.ok) {
      const txt = await resp.text().catch(()=>null);
      throw new Error(`HTTP ${resp.status} ${txt || ''}`);
    }
    const data = await resp.json();

    if(data.ok && data.results){
      data.results.forEach(res => {
        const el = document.createElement('div');
        if (res.status === 'ok') {
          el.className = 'msg ok';
          el.textContent = `✅ ${res.file}: ${res.chunks} чанков (вставлено: ${res.inserted}, пропущено: ${res.skipped})`;
        } else if (res.status === 'warning') {
          el.className = 'msg warn';
          el.textContent = `⚠️ ${res.file}: ${res.message}`;
        } else {
          el.className = 'msg err';
          el.textContent = `❌ ${res.file}: ${res.message}`;
        }
        idxOut.appendChild(el);
      });
      await updateStats();
      await loadUploadedFiles();

      uploadedFiles = [];
      showNames();
    } else {
      idxOut.innerHTML = `<div class="msg err">❌ Ошибка обработки: ${JSON.stringify(data)}</div>`;
    }

  } catch(e) {
    idxOut.innerHTML = `<div class="msg err">❌ ${e.message}</div>`;
  } finally {
    indexBtn.disabled = false;
    indexBtn.textContent = "Индексировать";
  }
});

// --- 7. Stats Functions ---
const statsContent = document.getElementById('statsContent');
const refreshStatsBtn = document.getElementById('refreshStatsBtn');

function renderStats(data) {
    const db_stats = data.database || {};

    if (!db_stats || !db_stats.total_documents) {
        statsContent.innerHTML = `<div class="hint">База данных пуста или недоступна.</div>`;
        return;
    }

    let html = `
        <p><strong>Количество проиндексированных разделов:</strong> ${db_stats.total_documents}</p>
        <p><strong>Количество уникальных файлов:</strong> ${db_stats.unique_files}</p>
        <h4>По источникам:</h4>
        <ul style="margin-top: 5px; list-style-type: none; padding-left: 10px;">
    `;

    const sources = Object.entries(db_stats.by_source || {});
    sources.sort(([, countA], [, countB]) => countB - countA);

    sources.forEach(([source, count]) => {
        html += `<li>• ${source}: <strong>${count}</strong></li>`;
    });

    html += `</ul>
        <div class="hint" style="margin-top: 10px;">
            Эмбеддер: ${data.embedder?.model || '—'} (${data.embedder?.dimension || '—'} dim)
        </div>
    `;
    statsContent.innerHTML = html;
}

async function updateStats() {
    if (!statsContent) return;
    statsContent.innerHTML = 'Загрузка... <span class="spinner"></span>';
    try {
        refreshStatsBtn.disabled = true;

        const resp = await authFetch(`${API_BASE}/stats`);
        if (!resp.ok) {
          const txt = await resp.text().catch(()=>null);
          throw new Error(`HTTP ${resp.status} ${txt || ''}`);
        }

        const data = await resp.json();
        renderStats(data);

    } catch(e) {
        if (e && (e.message === "not_authenticated" || e.message === "unauthorized")) {
            statsContent.innerHTML = `<div class="hint">🔐 Войдите в платформу, чтобы увидеть статистику базы знаний.</div>`;
        } else {
            statsContent.innerHTML = `<div class="msg err">❌ Ошибка загрузки статистики: ${e.message}</div>`;
        }
    } finally {
        refreshStatsBtn.disabled = false;
    }
}

// --- 8. Rebuild Cache Button ---
const rebuildCacheBtn = document.getElementById('rebuildCacheBtn');

rebuildCacheBtn?.addEventListener('click', async () => {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) { openAuthModal("login"); return; }

  if(!confirm("Пересобрать embedded.pkl из текущей БД? Это обновит кэш.")) return;
  rebuildCacheBtn.disabled = true;
  rebuildCacheBtn.innerHTML = 'Пересборка... <span class="spinner"></span>';
  try {
    const resp = await authFetch(`${API_BASE}/rebuild_cache`, { method: 'POST' });
    if(!resp.ok) {
      const txt = await resp.text().catch(()=>null);
      throw new Error(`HTTP ${resp.status} ${txt || ''}`);
    }
    const data = await resp.json();
    alert(`Кэш пересобран: ${data.count} записей`);
    await updateStats();
    await loadUploadedFiles();
  } catch(e) {
    alert('Ошибка: ' + e.message);
  } finally {
    rebuildCacheBtn.disabled = false;
    rebuildCacheBtn.textContent = 'Пересобрать кэш';
  }
});

refreshStatsBtn?.addEventListener('click', updateStats);

// --- 9. Load Uploaded Files List ---
const uploadedFilesList = document.getElementById('uploadedFilesList');

async function loadUploadedFiles() {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) {
    uploadedFilesList && (uploadedFilesList.innerHTML = '<div class="hint">Войдите чтобы увидеть свои файлы.</div>');
    return;
  }

  uploadedFilesList.innerHTML = 'Загрузка... <span class="spinner"></span>';
  try {
    const resp = await authFetch(`${API_BASE}/documents`);
    if (!resp.ok) {
      const txt = await resp.text().catch(()=>null);
      throw new Error(`HTTP ${resp.status} ${txt || ''}`);
    }
    const data = await resp.json();
    if (!data.ok) throw new Error("Ошибка сервера");

    const docs = data.documents || [];
    if (docs.length === 0) {
      uploadedFilesList.innerHTML = '<div class="hint">Файлы отсутствуют.</div>';
      return;
    }

    uploadedFilesList.innerHTML = '';
    docs.forEach(doc => {
      try {
        const row = document.createElement('div');
        row.className = 'doc-row';
        const left = document.createElement('div');

        const safeFilename = doc.filename && doc.filename.trim() ? doc.filename.trim() : '(нет имени файла)';
        const safeSource = doc.source || '';

        left.innerHTML = `<strong>${safeFilename}</strong><div class="doc-meta">${doc.chunks} чанков • ${safeSource}</div>`;

        const right = document.createElement('div');

        const viewBtn = document.createElement('button');
        viewBtn.className = 'btn';
        viewBtn.textContent = 'Просмотреть';
        viewBtn.addEventListener('click', ()=> {
          window.open(`${API_BASE}/documents/download?filename=${encodeURIComponent(safeFilename)}`, '_blank');
        });

        const delBtn = document.createElement('button');
        delBtn.className = 'btn';
        delBtn.style.marginLeft = '8px';
        delBtn.textContent = 'Удалить';
        delBtn.addEventListener('click', async ()=> {
          if (!confirm(`Удалить все чанки файла "${safeFilename}"?`)) return;
          try {
            const fd = new FormData();
            fd.append('filename', safeFilename);
            const r = await authFetch(`${API_BASE}/documents/delete`, { method: 'POST', body: fd });
            const res = await r.json();
            if (res.ok) {
              await updateStats();
              await loadUploadedFiles();
            } else {
              alert('Ошибка удаления');
            }
          } catch(e) {
            alert('Ошибка: ' + e.message);
          }
        });

        right.appendChild(viewBtn);
        right.appendChild(delBtn);

        row.appendChild(left);
        row.appendChild(right);
        uploadedFilesList.appendChild(row);
      } catch (e) {
          console.error(`Ошибка при обработке файла ${doc.filename}:`, e);
          const errorRow = document.createElement('div');
          errorRow.className = 'doc-row';
          errorRow.innerHTML = `❌ <strong style="color:var(--red)">Не удалось отобразить файл</strong>: ${doc.filename}`;
          uploadedFilesList.appendChild(errorRow);
      }
    });

  } catch(e) {
    uploadedFilesList.innerHTML = `<div class="msg err">❌ ${e.message}</div>`;
  }
}

// --- 10. Initial Load ---
document.addEventListener('DOMContentLoaded', () => {
  updateAuthUI();
  updateStats();
  loadUploadedFiles();
});
