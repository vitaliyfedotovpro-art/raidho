#!/bin/bash
# Raidho ᚱ — coder agent with VSA memory. Guided installer.
#
#   bash install.sh
#
# Interactive, explains every step, idempotent — re-run any time, it converges.
# Non-interactive (CI / scripted): set RAIDHO_PROVIDER and the provider key env
# (DEEPSEEK_API_KEY / ANTHROPIC_API_KEY) before running.
#
# The guided-installer concept follows MavKa by Oles Lytvyn (MozgAI) —
# https://github.com/MozgAI/MavKa — "AI installs itself", MIT. See README
# acknowledgments: MozgAI was also this project's critic throughout.
set -e

on_interrupt() {
  echo ""
  warn "$(t3 'Installation interrupted.' 'Установка прервана.' 'Встановлення перервано.')"
  info "$(t3 'Re-run the same command — the installer is idempotent.' \
          'Запусти ту же команду снова — установщик идемпотентен.' \
          'Запусти ту саму команду знову — інсталятор ідемпотентний.')"
  exit 130
}
trap on_interrupt INT TERM

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
RED='\033[0;31m'; WHITE='\033[1;37m'; DIM='\033[2m'; BOLD='\033[1m'; NC='\033[0m'
step() { echo -e "\n${GREEN}▸${NC} ${WHITE}$1${NC}"; }
info() { echo -e "  ${DIM}$1${NC}"; }
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "\n${RED}✗ $1${NC}"; exit 1; }

# ─── Language / Язык / Мова ───────────────────────────────────
# t3 <en> <ru> <uk> → echoes the line in the chosen language.
LANG_C="${RAIDHO_LANG:-}"
t3() { case "$LANG_C" in ru) echo "$2";; uk) echo "$3";; *) echo "$1";; esac; }
pick_lang() {
  [ -n "$LANG_C" ] && return
  echo ""
  echo -e "  ${WHITE}Language / Язык / Мова:${NC}  1) English   2) Русский   3) Українська"
  read -r -p "  [1/2/3] (default 1): " l
  case "$l" in 2) LANG_C=ru;; 3) LANG_C=uk;; *) LANG_C=en;; esac
}
show_url() {
  echo ""
  echo -e "    ${WHITE}${BOLD}$1${NC}"
  echo -e "      ${CYAN}${BOLD}$2${NC}"
  command -v qrencode >/dev/null 2>&1 && qrencode -t ANSI -m 1 "$2" 2>/dev/null | sed 's/^/      /' || true
  echo ""
}

cd "$(dirname "$0")"

echo -e "${CYAN}${BOLD}"
echo '   ᚱ  R a i d h o'
echo -e "${NC}${DIM}   Coder agent: dual-provider (plan smart / execute cheap),"
echo -e "   algebraic VSA memory, context-first mode, council debates.${NC}"
pick_lang
info "$(t3 'One installer, explains every step. ~5 minutes.' \
        'Один установщик, всё объяснит по ходу. ~5 минут.' \
        'Один інсталятор, усе пояснить по ходу. ~5 хвилин.')"

# ── 1/7 system check ─────────────────────────────────────────────────────────
step "[1/7] $(t3 'System check' 'Проверка системы' 'Перевірка системи')"
command -v python3 >/dev/null 2>&1 || fail "python3 not found — install Python 3.11+ first"
PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' \
  || fail "$(t3 "Python $PYV found, need 3.11+" "Python $PYV найден, нужен 3.11+" "Python $PYV знайдено, потрібен 3.11+")"
ok "Python $PYV"
command -v git >/dev/null 2>&1 && ok "git" || warn "git not found (optional)"

# ── 2/7 virtualenv + package ─────────────────────────────────────────────────
step "[2/7] $(t3 'Virtual environment & package' 'Окружение и пакет' 'Оточення та пакет')"
info "$(t3 'Everything goes into ./.venv — your system Python stays untouched.' \
        'Всё ставится в ./.venv — системный Python не трогаем.' \
        'Усе ставиться в ./.venv — системний Python не чіпаємо.')"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip -q install --upgrade pip
pip -q install -e .
ok "raidho installed into .venv"

# ── 3/7 provider choice ──────────────────────────────────────────────────────
step "[3/7] $(t3 'AI provider' 'Провайдер модели' 'Провайдер моделі')"
info "$(t3 "Raidho's flagship trick: think on a SMART model, execute on a CHEAP one." \
        'Фишка Raidho: думать умной моделью, исполнять — дешёвой.' \
        'Фішка Raidho: думати розумною моделлю, виконувати — дешевою.')"
echo ""
echo "    1) DeepSeek   — $(t3 'cheapest (~$2/month), great start' 'дешевле всего (~$2/мес), отличный старт' 'найдешевше (~$2/міс), чудовий старт')"
echo "    2) Anthropic  — $(t3 'Claude, smartest execution' 'Claude, самый умный execution' 'Claude, найрозумніше виконання')"
echo "    3) $(t3 'Both' 'Оба' 'Обидва')       — $(t3 'reason on Claude + execute on DeepSeek (split)' 'reasoning на Claude + execution на DeepSeek (split)' 'reasoning на Claude + execution на DeepSeek (split)')"
echo ""
PROVIDER="${RAIDHO_PROVIDER:-}"
if [ -z "$PROVIDER" ]; then
  read -r -p "  $(t3 'Choose' 'Выбери' 'Обери') [1/2/3] (default 1): " PROVIDER
  PROVIDER="${PROVIDER:-1}"
fi

need_key() {  # $1 env-name  $2 label  $3 signup-url
  local cur="${!1:-}"
  if [ -z "$cur" ] && [ -f .env ]; then
    cur=$(grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2-) || true
  fi
  if [ -z "$cur" ]; then
    show_url "$(t3 "Get a $2 API key here:" "$2 — ключ берётся тут:" "$2 — ключ береться тут:")" "$3"
    read -r -s -p "  $(t3 "Paste your $2 API key (hidden):" "Вставь ключ $2 (ввод скрыт):" "Встав ключ $2 (ввід прихований):") " cur
    echo ""
  else
    ok "$(t3 "$2 key found — reusing" "$2: ключ найден, переиспользую" "$2: ключ знайдено, перевикористовую")"
  fi
  [ -n "$cur" ] || fail "$2 key is empty"
  printf -v "$1" '%s' "$cur"
}

verify_deepseek() {
  info "$(t3 'Verifying the key with a live call…' 'Проверяю ключ живым вызовом…' 'Перевіряю ключ живим викликом…')"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 20 https://api.deepseek.com/chat/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
    -d '{"model":"deepseek-chat","max_tokens":1,"messages":[{"role":"user","content":"ping"}]}')
  [ "$code" = "200" ] && ok "DeepSeek key works" || fail "DeepSeek key check failed (HTTP $code)"
}
verify_anthropic() {
  info "$(t3 'Verifying the key (free call)…' 'Проверяю ключ (бесплатный вызов)…' 'Перевіряю ключ (безкоштовний виклик)…')"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 20 https://api.anthropic.com/v1/models \
    -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01")
  [ "$code" = "200" ] && ok "Anthropic key works" || fail "Anthropic key check failed (HTTP $code)"
}

ENV_LINES=()
case "$PROVIDER" in
  1) pip -q install -e '.[openai-compat]'
     need_key DEEPSEEK_API_KEY "DeepSeek" "https://platform.deepseek.com/api_keys"
     verify_deepseek
     ENV_LINES+=("CODER_PROVIDER=deepseek" "DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY") ;;
  2) pip -q install -e '.[anthropic]'
     need_key ANTHROPIC_API_KEY "Anthropic" "https://console.anthropic.com/settings/keys"
     verify_anthropic
     ENV_LINES+=("CODER_PROVIDER=anthropic" "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY") ;;
  3) pip -q install -e '.[anthropic,openai-compat]'
     need_key DEEPSEEK_API_KEY "DeepSeek" "https://platform.deepseek.com/api_keys"
     verify_deepseek
     need_key ANTHROPIC_API_KEY "Anthropic" "https://console.anthropic.com/settings/keys"
     verify_anthropic
     info "$(t3 'Split: Claude plans, DeepSeek executes (cheap tool-loop).' 'Split: Claude планирует, DeepSeek исполняет (дёшево).' 'Split: Claude планує, DeepSeek виконує (дешево).')"
     ENV_LINES+=("CODER_PROVIDER=deepseek" "DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY"
                 "CODER_REASON_PROVIDER=anthropic" "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY") ;;
  *) fail "unknown choice: $PROVIDER" ;;
esac

# ── 4/7 semantic memory (optional) ───────────────────────────────────────────
step "[4/7] $(t3 'Semantic memory (optional)' 'Семантическая память (опц.)' 'Семантична памʼять (опц.)')"
info "$(t3 'Without it memory recalls EXACT keywords only; with it — by meaning' \
        'Без неё память ищет только точные слова; с ней — по смыслу' \
        'Без неї памʼять шукає лише точні слова; з нею — за змістом')"
info "$(t3 '(a paraphrase finds the fact). ~400MB of models.' \
        '(парафраз находит факт). ~400МБ моделей.' \
        '(парафраз знаходить факт). ~400МБ моделей.')"
EMB="${RAIDHO_EMBED:-}"
if [ -z "$EMB" ]; then
  read -r -p "  $(t3 'Install semantic embedder?' 'Ставить семантический эмбеддер?' 'Ставити семантичний ембедер?') [y/N]: " EMB
fi
case "$EMB" in
  y|Y|yes|1) pip -q install -e '.[embed]' && ok "sentence-transformers installed" ;;
  *) info "$(t3 'Skipped — re-run the installer any time to add it.' 'Пропущено — можно добавить повторным запуском.' 'Пропущено — можна додати повторним запуском.')" ;;
esac

# ── 5/7 .env + smoke test ────────────────────────────────────────────────────
step "[5/7] $(t3 'Config & smoke test' 'Конфиг и проверка боем' 'Конфіг та перевірка боєм')"
{ echo "# Raidho config (created by install.sh — re-run to change)"
  for l in "${ENV_LINES[@]}"; do echo "$l"; done; } > .env
chmod 600 .env
ok ".env written (chmod 600, gitignored)"
info "$(t3 'Asking the agent one real question end-to-end…' 'Один живой вопрос агенту end-to-end…' 'Одне живе питання агенту end-to-end…')"
SMOKE=$(set -a && source .env && set +a && \
  .venv/bin/python -W ignore -c "
import asyncio
from agent.cli import _make_session
s = _make_session('.')
print(asyncio.run(s.chat('Reply with exactly: RAIDHO OK')))" 2>/dev/null | tail -1)
case "$SMOKE" in
  *"RAIDHO OK"*) ok "$(t3 'Live answer received — the agent works' 'Живой ответ получен — агент работает' 'Жива відповідь отримана — агент працює')" ;;
  *) warn "$(t3 "Smoke test answered unexpectedly: '$SMOKE' — check the key/balance." "Smoke-тест ответил странно: '$SMOKE' — проверь ключ/баланс." "Smoke-тест відповів дивно: '$SMOKE' — перевір ключ/баланс.")" ;;
esac

# ── 6/7 Open WebUI (optional) ────────────────────────────────────────────────
# Raidho ships a PLUGIN for the official Open WebUI (a Pipe Function), not its own
# UI. This step brings up the real Open WebUI and wires our plugin into it. Skip
# it freely — the CLI above is fully working, and power users can point any
# interface (or an existing Open WebUI) at Raidho instead.
step "[7/7] $(t3 'Open WebUI — web interface (optional)' 'Open WebUI — веб-интерфейс (опц.)' 'Open WebUI — веб-інтерфейс (опц.)')"
info "$(t3 'Nice browser chat UI (by the Open WebUI team). Raidho plugs in as' \
        'Удобный веб-чат (команда Open WebUI). Raidho подключается к нему' \
        'Зручний веб-чат (команда Open WebUI). Raidho підключається до нього')"
info "$(t3 'selectable models. Heavy-ish; skip if you are happy in the terminal.' \
        'моделями. Тяжеловат; можно пропустить, если хватает терминала.' \
        'моделями. Важкуватий; можна пропустити, якщо вистачає термінала.')"
WEBUI="${RAIDHO_WEBUI:-}"
if [ -z "$WEBUI" ]; then
  read -r -p "  $(t3 'Set up Open WebUI now?' 'Поднять Open WebUI?' 'Підняти Open WebUI?') [y/N]: " WEBUI
fi

# autowire args from the chosen provider (keys are still in scope from step 3)
owui_args() {
  case "$PROVIDER" in
    1) printf -- '--provider deepseek --key %s --model deepseek-chat' "$DEEPSEEK_API_KEY" ;;
    2) printf -- '--provider anthropic --key %s' "$ANTHROPIC_API_KEY" ;;
    3) printf -- '--provider deepseek --key %s --reason-provider anthropic --reason-key %s' \
         "$DEEPSEEK_API_KEY" "$ANTHROPIC_API_KEY" ;;
  esac
}
wait_health() {  # $1 = base url
  info "Waiting for Open WebUI to come up… / Жду старта Open WebUI…"
  for _ in $(seq 1 60); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$1/health" 2>/dev/null)" = "200" ] && return 0
    sleep 3
  done
  return 1
}

webui_up=false; PORT=""
case "$WEBUI" in
  y|Y|yes|1)
    # admin account for the automatic wiring (change it later in the UI)
    OWUI_EMAIL="${RAIDHO_WEBUI_EMAIL:-admin@raidho.local}"
    OWUI_PASS="${RAIDHO_WEBUI_PASS:-$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom 2>/dev/null | head -c 16 || echo raidho-$$)}"

    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
      info "Docker found — using the official image (recommended)."
      PORT=3000
      if ! docker ps -a --format '{{.Names}}' | grep -qx raidho-webui; then
        info "Pulling & starting ghcr.io/open-webui/open-webui (first time ~700MB)…"
        docker run -d --name raidho-webui -p 3000:8080 \
          -v raidho-webui-data:/app/backend/data \
          -v "$(pwd):/mnt/raidho:ro" \
          --add-host=host.docker.internal:host-gateway \
          ghcr.io/open-webui/open-webui:main >/dev/null 2>&1 \
          || warn "docker run failed — see 'docker logs raidho-webui'."
      else
        docker start raidho-webui >/dev/null 2>&1 || true
      fi
      if docker ps --format '{{.Names}}' | grep -qx raidho-webui; then
        # Raidho MUST be importable by the OWUI process, else the function errors
        # with "No module named 'agent'". Install it INTO the container.
        info "Installing Raidho into the container (so OWUI can import it)…"
        docker exec raidho-webui pip install -q -e '/mnt/raidho[anthropic,openai-compat]' >/dev/null 2>&1 \
          && docker restart raidho-webui >/dev/null 2>&1 \
          && ok "Raidho installed in container; restarted" \
          || warn "in-container install failed — autowire may report a module error"
        webui_up=true
      fi
    else
      warn "Docker not available — falling back to 'pip install open-webui'."
      info "(Docker is the cleaner path: https://docs.docker.com/get-docker/ )"
      PORT=8080
      # Into THE SAME .venv as Raidho — that is what lets OWUI import agent/vsa.
      if pip -q install open-webui 2>/dev/null; then
        ok "open-webui installed into .venv (same env as Raidho ✓)"
        if ! curl -s -o /dev/null -m 2 "http://localhost:$PORT/health" 2>/dev/null; then
          info "Starting Open WebUI in the background (logs: ./.owui.log)…"
          DATA_DIR="$(pwd)/.owui-data" WEBUI_SECRET_KEY="raidho-$(whoami)" \
            nohup .venv/bin/open-webui serve --port "$PORT" >./.owui.log 2>&1 &
        fi
        webui_up=true
      else
        warn "pip install open-webui failed. Open WebUI is optional — the CLI works."
      fi
    fi

    if [ "$webui_up" = true ] && wait_health "http://localhost:$PORT"; then
      info "Wiring the Raidho plugin via the Open WebUI API (no manual paste)…"
      if .venv/bin/python scripts/owui_autowire.py \
           --base "http://localhost:$PORT" --email "$OWUI_EMAIL" --password "$OWUI_PASS" \
           $(owui_args); then
        ok "$(t3 'Raidho is live in Open WebUI — models appear in the selector' 'Raidho живёт в Open WebUI — модели в селекторе' 'Raidho живе в Open WebUI — моделі в селекторі')"
        echo ""
        show_url "$(t3 'Open Open WebUI' 'Открой Open WebUI' 'Відкрий Open WebUI')" "http://localhost:$PORT"
        echo -e "    $(t3 'Admin login' 'Вход админа' 'Вхід адміна'):  ${CYAN}$OWUI_EMAIL${NC}  /  ${CYAN}$OWUI_PASS${NC}"
        echo -e "    ${DIM}$(t3 'Change the password in the UI. Models: Raidho · chat / · council.' 'Смени пароль в UI. Модели: Raidho · chat / · council.' 'Зміни пароль в UI. Моделі: Raidho · chat / · council.')"
        echo -e "    $(t3 'The code model stays OFF (unsandboxed shell).' 'Модель code выключена (unsandboxed shell).' 'Модель code вимкнена (unsandboxed shell).') docs/OPENWEBUI.md${NC}"
        echo ""
      else
        warn "Auto-wiring failed — Open WebUI is up at http://localhost:$PORT."
        info "Add the plugin manually: Workspace → Functions → paste"
        info "  $(pwd)/integrations/openwebui_raidho.py , then set Valves. (docs/OPENWEBUI.md)"
      fi
    elif [ "$webui_up" = true ]; then
      warn "Open WebUI did not become healthy in time — check ./.owui.log or 'docker logs raidho-webui'."
    fi ;;
  *) info "Skipped — see docs/OPENWEBUI.md to add it later, or use any UI you like." ;;
esac

# ── done: how to use ─────────────────────────────────────────────────────────
step "[done] $(t3 'How to use' 'Как пользоваться' 'Як користуватися')"
echo ""
echo "    $(t3 'Start' 'Запуск' 'Запуск'):"
echo "      source .venv/bin/activate && set -a && source .env && set +a"
echo "      coder                       # $(t3 'interactive REPL' 'интерактивный REPL' 'інтерактивний REPL')"
echo "      coder \"fix the bug in x.py\" # $(t3 'one-shot task' 'разовая задача' 'разове завдання')"
echo ""
echo "    $(t3 'REPL modes' 'Режимы REPL' 'Режими REPL'):"
echo "      /code     $(t3 'agentic coding (bash, read, write)  ← default' 'агентное кодирование (bash, read, write)  ← по умолч.' 'агентне кодування (bash, read, write)  ← типово')"
echo "      /text     $(t3 'discuss & plan, no tools' 'обсуждение и план, без инструментов' 'обговорення та план, без інструментів')"
echo "      /ctx      $(t3 'toggle context-first (workspace into the 1st call)' 'переключить context-first (контекст в 1-й вызов)' 'перемкнути context-first (контекст у 1-й виклик)')"
echo "      /council  $(t3 'two providers debate → consensus' 'два провайдера спорят → консенсус' 'два провайдери сперечаються → консенсус')"
echo ""
echo -e "    ${YELLOW}⚠${NC} $(t3 'Security: the bash tool is UNSANDBOXED in your workdir —' 'Безопасность: bash БЕЗ песочницы в рабочем каталоге —' 'Безпека: bash БЕЗ пісочниці в робочому каталозі —')"
echo "      $(t3 'run in a project folder you trust, not in $HOME. See SECURITY.md.' 'запускай в доверенном проекте, не в $HOME. См. SECURITY.md.' 'запускай у довіреному проєкті, не в $HOME. Див. SECURITY.md.')"
echo ""
echo "    $(t3 'Memory: stores durable facts (remember) and recalls them into the prompt.' 'Память: хранит факты (remember) и подмешивает их в промпт.' 'Памʼять: зберігає факти (remember) і підмішує їх у промпт.')"
echo "    $(t3 'Web UI: re-run the installer and say yes at the Open WebUI step.' 'Веб-интерфейс: перезапусти установщик и согласись на шаге Open WebUI.' 'Веб-інтерфейс: перезапусти інсталятор і погодься на кроці Open WebUI.')"
echo ""
echo -e "  ${DIM}Installer concept: MavKa by Oles Lytvyn (MozgAI) — github.com/MozgAI/MavKa${NC}"
echo -e "  ${GREEN}${BOLD}ᚱ $(t3 'Ready.' 'Готово.' 'Готово.')${NC}"
