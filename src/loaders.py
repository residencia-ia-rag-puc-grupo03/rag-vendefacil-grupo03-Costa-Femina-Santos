
import json
import os

import re

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader

from metadata_schema import ChunkMetadata


def _make_id(prefix: str, idx) -> str:
    return f"{prefix}-{idx}"


# Correção (Etapa 4 - item 3): customers.csv e stores.json guardavam o nome
# completo do produto no campo `module` (ex.: "VendeFácil Estoque"), enquanto
# tickets.jsonl e os manuais usam a chave curta (ex.: "estoque"). Isso fazia
# o Query Analyzer, em alguns casos, extrair o valor "errado" (o nome
# completo) e filtrar só por customer/store, excluindo tickets e manuais que
# usam a chave curta - mesmo already havendo uma correção parcial anterior
# (Etapa 2) só para tickets. Esta tabela unifica os dois lados para o mesmo
# padrão, usando só valores que já existem nos dados reais.
_MODULE_NAME_TO_KEY = {
    "vendefacil pdv": "pdv",
    "vendefacil estoque": "estoque",
    "vendefacil loja": "ecommerce",
    "vendefacil analytics": "analytics",
    "vendefacil pay": "pay",
}


def _strip_accents(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_module(value: str | None) -> str | None:
    """Converte nome completo de produto para a chave curta canônica,
    quando reconhecido. Caso não reconheça, devolve o valor original
    (sem inventar chave nova)."""
    if not value or value == "nan":
        return value
    key = _strip_accents(value.strip().lower())
    return _MODULE_NAME_TO_KEY.get(key, value)

def load_customers_csv(path: str) -> list[Document]:
    df = pd.read_csv(path)
    docs = []
    for i, row in df.iterrows():
        text = (
            f"Cliente {row.get('customer_id', i)}: {row.get('company_name', '')} "
            f"({row.get('segment', '')}), CNPJ {row.get('cnpj', '')}, "
            f"localizado em {row.get('city', '')}/{row.get('state', '')}, "
            f"plano {row.get('plan', '')}, produto principal {row.get('main_product', '')}, "
            f"MRR de R$ {row.get('mrr', '')}, situação {row.get('status', '')}, "
            f"contato {row.get('contact_email', '')}."
        )
        meta = ChunkMetadata(
            source_file="structured/customers.csv",
            doc_type="customer",
            chunk_id=_make_id("customer", i),
            sensitivity="interno",
            customer_id=str(row.get("customer_id", "")),
            state=str(row.get("state", "")),
            module=normalize_module(str(row.get("main_product", ""))),
            status=str(row.get("status", "")),
            company_name=str(row.get("company_name", "")) or None,
        )
        docs.append(Document(page_content=text, metadata=meta.to_dict()))
    return docs


def load_employees_csv(path: str) -> list[Document]:

    df = pd.read_csv(path)
    docs = []
    for i, row in df.iterrows():
        text = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        meta = ChunkMetadata(
            source_file="structured/employees.csv",
            doc_type="employee",
            chunk_id=_make_id("employee", i),
            sensitivity="restrito",
        )
        docs.append(Document(page_content=text, metadata=meta.to_dict()))
    return docs

def load_generic_csv(path: str, doc_type: str, sensitivity: str, source_label: str) -> list[Document]:
    """
    Carrega um CSV genérico (logs, vendas, etc.) para chunks de texto.

    Correção (Etapa 4 - item 2): o arquivo original só extraía `date`/`timestamp`
    para os metadados, mesmo quando o CSV já tinha colunas reais como
    `customer_id`, `module`, `state` e `status` (caso de system_logs.csv e
    sales.csv). Sem isso, o Query Analyzer nunca conseguia filtrar logs/vendas
    por cliente ou módulo, mesmo que a pergunta pedisse exatamente isso -
    o dado existia na fonte, só não chegava ao metadado do chunk.
    Agora extraímos essas colunas quando existem, sem inventar nada.
    """
    df = pd.read_csv(path)
    docs = []
    for i, row in df.iterrows():
        text = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        date_value = None
        if "date" in df.columns:
            date_value = str(row.get("date"))
        elif "timestamp" in df.columns:
            date_value = str(row.get("timestamp"))

        meta = ChunkMetadata(
            source_file=source_label,
            doc_type=doc_type,
            chunk_id=_make_id(doc_type, i),
            sensitivity=sensitivity,
            date=date_value,
            customer_id=str(row.get("customer_id")) if "customer_id" in df.columns else None,
            module=normalize_module(str(row.get("module"))) if "module" in df.columns else None,
            state=str(row.get("state")) if "state" in df.columns else None,
            status=str(row.get("status")) if "status" in df.columns else None,
        )
        docs.append(Document(page_content=text, metadata=meta.to_dict()))
    return docs


def load_system_logs_csv(path: str, sensitivity: str, source_label: str) -> list[Document]:
    """
    Carrega system_logs.csv (doc_type="log" fixo).

    Correção (Etapa 4 - item 2): antes só extraía `date`/`timestamp`. O CSV
    real tem colunas `customer_id`, `module` e `state`/`status` que eram
    descartadas - sem elas, o Query Analyzer nunca conseguia filtrar logs
    por cliente/módulo mesmo quando a pergunta pedia exatamente isso.
    Reaproveita a mesma extração de load_generic_csv, só fixando doc_type.
    """
    return load_generic_csv(
        path, doc_type="log", sensitivity=sensitivity, source_label=source_label
    )

def load_products_json(path: str) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("products", [])
    docs = []
    for i, item in enumerate(items):
        features = ", ".join(item.get("features", []))
        # Correção (Etapa 4 - item 24): o texto serializado do produto omitia
        # `tech_lead` e `product_manager`, que estão no JSON de origem. A Q02
        # ("quem é o Tech Lead / PM do VendeFácil Estoque?") era irrespondível
        # não por falha de recuperação, mas porque a resposta NÃO ESTAVA no
        # índice - o chunk certo, mesmo recuperado, não tinha os nomes. Agora
        # entram na frase, com o rótulo que a pergunta usa ("Tech Lead"/"PM").
        price = item.get("standalone_monthly_price_brl")
        price_str = (
            f"Preço avulso: R$ {price}/mês. " if price not in (None, "") else ""
        )
        text = (
            f"Produto {item.get('product_id', i)}: {item.get('name', '')}, "
            f"categoria {item.get('category', '')}. {item.get('description', '')} "
            f"Responsável técnico (Tech Lead): {item.get('tech_lead', 'não informado')}. "
            f"Gerente de produto (PM / Product Manager): {item.get('product_manager', 'não informado')}. "
            f"{price_str}"
            f"Funcionalidades: {features}. SLA de uptime: {item.get('sla_uptime', '')}."
        )
        meta = ChunkMetadata(
            source_file="structured/products.json",
            doc_type="product",
            chunk_id=_make_id("product", i),
            sensitivity="publico",
            # Liga o produto ao módulo canônico (ex.: "VendeFácil Estoque" ->
            # "estoque"), pelo mesmo mapa usado em customers/stores. Assim uma
            # pergunta sobre um módulo específico pode ancorar no produto.
            module=normalize_module(item.get("name")),
        )
        docs.append(Document(page_content=text, metadata=meta.to_dict()))
    return docs

def load_stores_json(path: str) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("network_stores", [])
    docs = []
    for i, item in enumerate(items):
        modules = ", ".join(item.get("active_modules", []))
        text = (
            f"Loja {item.get('store_id', i)}: {item.get('store_name', '')}, da rede "
            f"{item.get('company_name', '')} (cliente {item.get('customer_id', '')}), "
            f"localizada em {item.get('city', '')}/{item.get('state', '')}, com "
            f"{item.get('pos_terminals_count', '')} terminais PDV. "
            f"Módulos ativos: {modules}."
        )
        meta = ChunkMetadata(
            source_file="structured/stores.json",
            doc_type="store",
            chunk_id=_make_id("store", i),
            sensitivity="publico",
            state=str(item.get("state", "")),
            customer_id=str(item.get("customer_id", "")),
        )
        docs.append(Document(page_content=text, metadata=meta.to_dict()))
    return docs

def load_tickets_jsonl(path: str, max_body_chars: int = 1000) -> list[Document]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            ticket = json.loads(line)
            header = (
                f"Ticket {ticket.get('ticket_id', i)} | "
                f"cliente {ticket.get('customer_id', '')} | "
                f"estado {ticket.get('state', '')} | "
                f"módulo {ticket.get('module', '')} | "
                f"prioridade {ticket.get('priority', '')} | "
                f"status {ticket.get('status', '')} | "
                f"título: {ticket.get('title', '')}.\n"
            )
            body = ticket.get("body", ticket.get("description", ""))
            resolution = ticket.get("resolution")
            if resolution:
                body = f"{body}\nResolução: {resolution}"

            if len(body) <= max_body_chars:
                text = header + body
                meta = ChunkMetadata(
                    source_file="semi_structured/tickets.jsonl",
                    doc_type="ticket",
                    chunk_id=_make_id("ticket", i),
                    sensitivity="interno",
                    customer_id=str(ticket.get("customer_id", "")),
                    state=str(ticket.get("state", "")),
                    module=str(ticket.get("module", "")),
                    priority=str(ticket.get("priority", "")),
                    status=str(ticket.get("status", "")),
                    date=str(ticket.get("created_at", "")),
                )
                docs.append(Document(page_content=text, metadata=meta.to_dict()))
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_body_chars, chunk_overlap=100
                )
                parts = splitter.split_text(body)
                for j, part in enumerate(parts):
                    meta = ChunkMetadata(
                        source_file="semi_structured/tickets.jsonl",
                        doc_type="ticket",
                        chunk_id=_make_id(f"ticket-{i}", j),
                        sensitivity="interno",
                        customer_id=str(ticket.get("customer_id", "")),
                        state=str(ticket.get("state", "")),
                        module=str(ticket.get("module", "")),
                        priority=str(ticket.get("priority", "")),
                        status=str(ticket.get("status", "")),
                        date=str(ticket.get("created_at", "")),
                    )
                    docs.append(
                        Document(page_content=header + part, metadata=meta.to_dict())
                    )
    return docs


def load_markdown(path: str, doc_type: str, source_label: str, sensitivity: str = "interno",
                   extra_meta: dict | None = None) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sections = md_splitter.split_text(text)

    fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    docs = []
    for i, section in enumerate(sections):
        section_title = " / ".join(section.metadata.values()) or "sem_titulo"
        if len(section.page_content) > 1200:
            sub_chunks = fallback_splitter.split_text(section.page_content)
        else:
            sub_chunks = [section.page_content]

        for j, chunk_text in enumerate(sub_chunks):
            meta = ChunkMetadata(
                source_file=source_label,
                doc_type=doc_type,
                chunk_id=_make_id(f"md-{os.path.basename(path)}-{i}", j),
                sensitivity=sensitivity,
                section=section_title,
                **(extra_meta or {}),
            )
            docs.append(Document(page_content=chunk_text, metadata=meta.to_dict()))
    return docs

def load_pdf(path: str, sensitivity: str, source_label: str) -> list[Document]:
    reader = PdfReader(path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(full_text)

    docs = []
    for i, chunk_text in enumerate(chunks):
        meta = ChunkMetadata(
            source_file=source_label,
            doc_type="policy",
            chunk_id=_make_id(f"pdf-{os.path.basename(path)}", i),
            sensitivity=sensitivity,
        )
        docs.append(Document(page_content=chunk_text, metadata=meta.to_dict()))
    return docs

def load_email_txt(
    path: str,
    source_label: str,
    max_chars: int = 1000
) -> list[Document]:

    with open(path, encoding="utf-8") as f:
        text = f.read().strip()

    # Primeiro separa o thread em mensagens.
    # Cada nova mensagem começa por "De:" no início de uma linha.
    messages = re.split(
        r"(?=^De:\s*)",
        text,
        flags=re.MULTILINE
    )

    # Remove pedaços vazios
    messages = [
        message.strip()
        for message in messages
        if message.strip()
    ]

    docs = []

    for message_idx, message in enumerate(messages):

        # Só depois de separar as mensagens é que aplicamos
        # o limite de tamanho.
        if len(message) <= max_chars:
            parts = [message]
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chars,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " "]
            )
            parts = splitter.split_text(message)

        for part_idx, chunk_text in enumerate(parts):
            meta = ChunkMetadata(
                source_file=source_label,
                doc_type="email",
                chunk_id=_make_id(
                    f"email-{os.path.basename(path)}-{message_idx}",
                    part_idx
                ),
                sensitivity="restrito",
            )

            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata=meta.to_dict()
                )
            )

    return docs
