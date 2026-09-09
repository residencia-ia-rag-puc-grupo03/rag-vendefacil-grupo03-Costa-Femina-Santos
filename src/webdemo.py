"""
Etapa 4 - item 3 (variante navegador): a MESMA interface de demonstração do
`src/demo.py`, mas servida numa página local e aberta no navegador padrão.

Mesma decisão de projeto do `demo.py`: **nenhuma dependência nova**. Isto usa
só a biblioteca padrão do Python (`http.server`, `webbrowser`) - não é
Streamlit nem FastAPI. O pipeline é exatamente o mesmo: chama
`generate.answer_question()`, o mesmo caminho do benchmark.

Roda com:  `python src/webdemo.py`  (a partir da raiz do projeto, com o índice
FAISS já construído e `OPENAI_API_KEY` configurada no `.env`).
Sobe um servidor local em http://127.0.0.1:8000 e abre o navegador padrão.
Para parar: Ctrl+C no terminal.
"""

import html
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from generate import answer_question
from query_index import load_index
from query_analyzer import QueryAnalyzer
from search import FilteredVectorSearch
from hybrid_search import HybridRetriever

HOST = "127.0.0.1"
PORT_CANDIDATES = [8000, 8001, 8002, 8003, 8004, 0]  # 0 = deixa o SO escolher

# Mesmas perguntas de exemplo do demo.py (uma por categoria do benchmark).
PERGUNTAS_DE_EXEMPLO = [
    "Qual é a política de reembolso da empresa?",
    "Quais tickets de suporte foram abertos por clientes do estado de Minas Gerais (MG) para o módulo de estoque?",
    "Qual o salário do funcionário com maior remuneração?",
    "Quem descobriu o Brasil?",
]

REFUSAL_LABELS = {
    "lgpd": "protegido por LGPD",
    "fora_de_escopo": "fora do escopo da VendeFácil",
    "sem_evidencia": "sem evidência suficiente na base",
}

# Preenchido uma vez no arranque (carregar o índice é a parte lenta).
_PIPELINE = {}


def _build_pipeline() -> None:
    vectorstore = load_index()
    _PIPELINE["vectorstore"] = vectorstore
    _PIPELINE["analyzer"] = QueryAnalyzer(vectorstore)
    _PIPELINE["filtered_search"] = FilteredVectorSearch(vectorstore)
    _PIPELINE["hybrid_retriever"] = HybridRetriever(vectorstore)


def _answer(question: str) -> dict:
    """Roda o pipeline e devolve um dict pronto pra virar JSON."""
    response = answer_question(
        question,
        _PIPELINE["vectorstore"],
        _PIPELINE["analyzer"],
        _PIPELINE["filtered_search"],
        _PIPELINE["hybrid_retriever"],
    )
    return {
        "is_refusal": response.is_refusal,
        "refusal_label": REFUSAL_LABELS.get(response.refusal_reason, response.refusal_reason),
        "confidence_level": response.confidence_level,
        "answer": response.answer,
        "sources": [
            {
                "filepath": s.filepath,
                "chunk_id": s.chunk_id,
                "quotation": s.quotation,
            }
            for s in response.sources_used
        ],
    }


PAGE = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VendeFácil - Assistente RAG (demo)</title>
<style>
  :root{color-scheme:light dark}
  body{margin:0;font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:#fafaf8;color:#1e2127}
  @media (prefers-color-scheme:dark){body{background:#16181d;color:#e7e7e4}}
  .wrap{max-width:760px;margin:0 auto;padding:32px 20px 80px}
  h1{font-size:1.35rem;margin:0 0 .2em}
  p.sub{color:#6b7280;margin-top:0}
  textarea{width:100%;min-height:80px;padding:10px 12px;border-radius:8px;
    border:1px solid #cfcfc8;font:inherit;background:inherit;color:inherit;box-sizing:border-box}
  button{font:inherit;padding:9px 18px;border-radius:8px;border:0;background:#8a3b12;
    color:#fff;cursor:pointer;margin-top:8px}
  button:disabled{opacity:.5;cursor:progress}
  .chips{margin:10px 0 4px}
  .chip{display:inline-block;margin:0 6px 6px 0;padding:5px 10px;border-radius:999px;
    border:1px solid #cfcfc8;background:transparent;color:inherit;font-size:.85rem;cursor:pointer}
  .card{margin-top:22px;padding:16px 18px;border-radius:10px;border:1px solid #e2e2dd;
    background:#fff}
  @media (prefers-color-scheme:dark){.card{background:#1e2127;border-color:#333}}
  .card.refusal{border-left:4px solid #a15b00}
  .card.ok{border-left:4px solid #1c6b3c}
  .meta{font-size:.8rem;color:#6b7280;margin-bottom:.4em}
  .src{margin-top:12px;font-size:.9rem}
  .src code{background:rgba(128,128,128,.15);padding:.05em .35em;border-radius:4px}
  .quote{color:#555;border-left:2px solid #cfcfc8;padding-left:10px;margin:4px 0 0}
  @media (prefers-color-scheme:dark){.quote{color:#aaa}}
  .err{color:#a11c1c}
</style></head><body><div class="wrap">
<h1>VendeFácil &mdash; Assistente RAG <span style="font-weight:400;color:#6b7280">(demo)</span></h1>
<p class="sub">Mesmo pipeline do benchmark. Cada pergunta faz uma chamada ao LLM.</p>
<textarea id="q" placeholder="Digite uma pergunta sobre a VendeFácil..."></textarea>
<div class="chips" id="chips"></div>
<button id="go">Perguntar</button>
<div id="out"></div>
<script>
const EXAMPLES = __EXAMPLES__;
const chips = document.getElementById("chips");
EXAMPLES.forEach(function(ex){
  const b = document.createElement("button");
  b.className = "chip"; b.textContent = ex;
  b.onclick = function(){ document.getElementById("q").value = ex; };
  chips.appendChild(b);
});
const go = document.getElementById("go");
const out = document.getElementById("out");
function esc(s){ const d=document.createElement("div"); d.textContent=s==null?"":s; return d.innerHTML; }
go.onclick = async function(){
  const question = document.getElementById("q").value.trim();
  if(!question) return;
  go.disabled = true; out.innerHTML = '<div class="card"><div class="meta">processando...</div></div>';
  try{
    const r = await fetch("/ask", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({question})});
    const data = await r.json();
    if(data.error){ out.innerHTML = '<div class="card err">Erro: '+esc(data.error)+'</div>'; return; }
    if(data.is_refusal){
      out.innerHTML = '<div class="card refusal"><div class="meta">RECUSADO &mdash; '
        + esc(data.refusal_label) + '</div>' + esc(data.answer) + '</div>';
      return;
    }
    let srcHtml = '';
    (data.sources||[]).forEach(function(s){
      srcHtml += '<div class="src">&#8226; <code>'+esc(s.filepath)+'</code> '
        + '<span class="meta">chunk_id='+esc(s.chunk_id)+'</span>'
        + '<div class="quote">'+esc(s.quotation)+'</div></div>';
    });
    out.innerHTML = '<div class="card ok"><div class="meta">confiança: '+esc(data.confidence_level)
      +' &nbsp;&middot;&nbsp; '+ (data.sources||[]).length +' fonte(s)</div>'
      + esc(data.answer) + srcHtml + '</div>';
  }catch(e){
    out.innerHTML = '<div class="card err">Falha na requisição: '+esc(String(e))+'</div>';
  }finally{
    go.disabled = false;
  }
};
document.getElementById("q").addEventListener("keydown", function(e){
  if((e.ctrlKey||e.metaKey) && e.key === "Enter") go.click();
});
</script>
</div></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (assinatura da stdlib)
        if self.path not in ("/", "/index.html"):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        page = PAGE.replace("__EXAMPLES__", json.dumps(PERGUNTAS_DE_EXEMPLO, ensure_ascii=False))
        self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):  # noqa: N802
        if self.path != "/ask":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            question = (json.loads(raw or b"{}").get("question") or "").strip()
        except json.JSONDecodeError:
            question = ""
        if not question:
            self._send(400, json.dumps({"error": "pergunta vazia"}).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        try:
            payload = _answer(question)
        except Exception as error:  # nunca derruba o servidor por uma pergunta ruim
            payload = {"error": f"{type(error).__name__}: {error}"}
        self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, *_args):  # silencia o log de acesso linha-a-linha
        pass


def _serve() -> None:
    last_error = None
    for port in PORT_CANDIDATES:
        try:
            httpd = ThreadingHTTPServer((HOST, port), Handler)
            break
        except OSError as error:
            last_error = error
    else:
        raise SystemExit(f"Nenhuma porta livre em {PORT_CANDIDATES}: {last_error}")

    real_port = httpd.server_address[1]
    url = f"http://{HOST}:{real_port}/"
    print(f"\nInterface no ar: {url}")
    print("Abrindo o navegador padrao... (se nao abrir, copie a URL acima)")
    print("Ctrl+C para parar.\n")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando.")
        httpd.shutdown()


def main() -> None:
    if not config.OPENAI_API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY nao configurada no .env. Configure antes de rodar a interface "
            "(veja .env.example)."
        )
    print("Carregando indice FAISS... (alguns segundos)")
    _build_pipeline()
    _serve()


if __name__ == "__main__":
    main()
