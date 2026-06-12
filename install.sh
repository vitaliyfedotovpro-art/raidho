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
  echo "  ⚠  Installation interrupted / Установка прервана."
  echo "     Re-run the same command — the installer is idempotent."
  echo "     Запусти ту же команду снова — установщик идемпотентен."
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
echo "   algebraic VSA memory, context-first mode, council debates."
echo -e "   Один установщик, всё объяснит по ходу. ~5 минут.${NC}"

# ── 1/6 system check ─────────────────────────────────────────────────────────
step "[1/6] System check / Проверка системы"
command -v python3 >/dev/null 2>&1 || fail "python3 not found — install Python 3.11+ first"
PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' \
  || fail "Python $PYV found, need 3.11+ / нужен Python 3.11+"
ok "Python $PYV"
command -v git >/dev/null 2>&1 && ok "git" || warn "git not found (optional)"

# ── 2/6 virtualenv + package ─────────────────────────────────────────────────
step "[2/6] Virtual environment & package / Окружение и пакет"
info "Everything goes into ./.venv — your system Python stays untouched."
info "Всё ставится в ./.venv — системный Python не трогаем."
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip -q install --upgrade pip
pip -q install -e .
ok "raidho installed into .venv"

# ── 3/6 provider choice ──────────────────────────────────────────────────────
step "[3/6] AI provider / Провайдер модели"
info "Raidho's flagship trick: think on a SMART model, execute on a CHEAP one."
info "Фишка Raidho: думать умной моделью, исполнять — дешёвой."
echo ""
echo "    1) DeepSeek      — cheapest (~\$2/month real use), отличный старт"
echo "    2) Anthropic     — Claude, самый умный execution"
echo "    3) Both / Оба    — reasoning на Claude + execution на DeepSeek (split)"
echo ""
PROVIDER="${RAIDHO_PROVIDER:-}"
if [ -z "$PROVIDER" ]; then
  read -r -p "  Choose / Выбери [1/2/3] (default 1): " PROVIDER
  PROVIDER="${PROVIDER:-1}"
fi

need_key() {  # $1 env-name  $2 label  $3 signup-url
  local cur="${!1:-}"
  if [ -z "$cur" ] && [ -f .env ]; then
    cur=$(grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2-) || true
  fi
  if [ -z "$cur" ]; then
    show_url "Get a $2 API key here / Ключ берётся тут:" "$3"
    read -r -s -p "  Paste your $2 API key (hidden / ввод скрыт): " cur
    echo ""
  else
    ok "$2 key found (env or .env) — reusing / найден, переиспользую"
  fi
  [ -n "$cur" ] || fail "$2 key is empty"
  printf -v "$1" '%s' "$cur"
}

verify_deepseek() {
  info "Verifying the key with a live 1-token call… / Проверяю ключ живым вызовом…"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 20 https://api.deepseek.com/chat/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
    -d '{"model":"deepseek-chat","max_tokens":1,"messages":[{"role":"user","content":"ping"}]}')
  [ "$code" = "200" ] && ok "DeepSeek key works" || fail "DeepSeek key check failed (HTTP $code)"
}
verify_anthropic() {
  info "Verifying the key (models list, costs nothing)… / Проверяю ключ (бесплатный вызов)…"
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
     info "Split: Claude plans (reasoning), DeepSeek executes (cheap tool-loop)."
     ENV_LINES+=("CODER_PROVIDER=deepseek" "DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY"
                 "CODER_REASON_PROVIDER=anthropic" "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY") ;;
  *) fail "unknown choice: $PROVIDER" ;;
esac

# ── 4/6 semantic memory (optional) ───────────────────────────────────────────
step "[4/6] Semantic memory / Семантическая память (optional)"
info "Without it memory recalls EXACT keywords only. With it — meaning:"
info "a Russian paraphrase finds an English fact. Costs ~400MB of models."
info "Без неё память ищет только точные слова; с ней — по смыслу (~400МБ)."
EMB="${RAIDHO_EMBED:-}"
if [ -z "$EMB" ]; then
  read -r -p "  Install semantic embedder? / Ставить? [y/N]: " EMB
fi
case "$EMB" in
  y|Y|yes|1) pip -q install -e '.[embed]' && ok "sentence-transformers installed" ;;
  *) info "Skipped — re-run installer any time to add it. / Пропущено." ;;
esac

# ── 5/6 .env + smoke test ────────────────────────────────────────────────────
step "[5/6] Config & smoke test / Конфиг и проверка боем"
{ echo "# Raidho config (created by install.sh — re-run to change)"
  for l in "${ENV_LINES[@]}"; do echo "$l"; done; } > .env
chmod 600 .env
ok ".env written (chmod 600, gitignored)"
info "Asking the agent one real question end-to-end… / Один живой вопрос агенту…"
SMOKE=$(set -a && source .env && set +a && \
  .venv/bin/python -W ignore -c "
import asyncio
from agent.cli import _make_session
s = _make_session('.')
print(asyncio.run(s.chat('Reply with exactly: RAIDHO OK')))" 2>/dev/null | tail -1)
case "$SMOKE" in
  *"RAIDHO OK"*) ok "Live answer received — the agent works / агент отвечает" ;;
  *) warn "Smoke test answered unexpectedly: '$SMOKE' — check the key/balance." ;;
esac

# ── 6/6 how to use ───────────────────────────────────────────────────────────
step "[6/6] Done — how to use / Готово — как пользоваться"
cat <<'GUIDE'

    Start / Запуск:
      source .venv/bin/activate && set -a && source .env && set +a
      coder                       # interactive REPL
      coder "fix the bug in x.py" # one-shot task

    REPL modes / Режимы:
      /code     agentic coding (tools: bash, read, write)  ← default
      /text     discuss & plan, no tools
      /ctx      toggle context-first (workspace handed to the 1st call —
                fewer iterations, measured ×2.6 cheaper on audit tasks)
      /council  two providers debate → consensus

    ⚠ Security / Безопасность: the bash tool is UNSANDBOXED in your workdir —
      run in a project folder you trust, not in $HOME. См. SECURITY.md.

    Memory: the agent stores durable facts (remember) and recalls them into
    the prompt. Uninstall = delete this folder; nothing else is touched.

GUIDE
echo -e "  ${DIM}Installer concept: MavKa by Oles Lytvyn (MozgAI) — github.com/MozgAI/MavKa${NC}"
echo -e "  ${GREEN}${BOLD}ᚱ Ready. / Готово.${NC}"
