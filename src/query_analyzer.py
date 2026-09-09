import re
import unicodedata
from pydantic import BaseModel
from query_index import load_index


class QueryFilters(BaseModel):
    doc_type: str | None = None
    state: str | None = None
    module: str | None = None
    customer_id: str | None = None
    priority: str | None = None
    status: str | None = None
    date: str | None = None
    source_file: str | None = None

    def to_dict(self):
        return self.model_dump(exclude_none=True)


def normalize(text: str) -> str:
    """Normaliza caixa, acentos e espaços."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.lower().strip().split())


STATE_ALIASES = {
    "acre": "AC",
    "alagoas": "AL",
    "amapa": "AP",
    "amazonas": "AM",
    "bahia": "BA",
    "ceara": "CE",
    "distrito federal": "DF",
    "espirito santo": "ES",
    "goias": "GO",
    "maranhao": "MA",
    "mato grosso": "MT",
    "mato grosso do sul": "MS",
    "minas gerais": "MG",
    "para": "PA",
    "paraiba": "PB",
    "parana": "PR",
    "pernambuco": "PE",
    "piaui": "PI",
    "rio de janeiro": "RJ",
    "rio grande do norte": "RN",
    "rio grande do sul": "RS",
    "rondonia": "RO",
    "roraima": "RR",
    "santa catarina": "SC",
    "sao paulo": "SP",
    "sergipe": "SE",
    "tocantins": "TO",
}


DOC_TYPE_ALIASES = {
    "ticket": ["ticket", "tickets", "chamado", "chamados"],
    "customer": ["cliente", "clientes"],
    "employee": ["funcionario", "funcionarios", "colaborador", "colaboradores"],
    "product": ["produto", "produtos"],
    "store": ["loja", "lojas", "filial", "filiais"],
    "log": ["log", "logs"],
    "manual": ["manual", "manuais"],
    "ata": ["ata", "atas", "reuniao", "reunioes", "retrospectiva"],
    "policy": ["politica", "politicas"],
    "email": ["email", "emails"],
    "sale": ["venda", "vendas"],
}

# Correção (Etapa 4 - item 4): tipos de CONTEÚDO (o documento que de fato
# responde à pergunta) têm prioridade sobre tipos de ENTIDADE (palavras que
# só nomeiam de quem se fala, sem pedir o cadastro dela). Sem isso, uma
# pergunta como "logs ... para o cliente CUST008" perdia o filtro doc_type=log
# porque a palavra "cliente" (entidade incidental) era checada antes de "log"
# (o conteúdo real pedido), no dicionário acima.
_CONTENT_DOC_TYPES = ["ticket", "log", "manual", "ata", "policy", "email", "sale"]
_ENTITY_DOC_TYPES = ["customer", "employee", "product", "store"]

# Correção (Etapa 4 - item 5): termos que aparecem nos nomes/conteúdo real
# das políticas internas (conferido em data/unstructured/policies/*.md) e que,
# quando citados, indicam com bastante segurança que a resposta está numa
# política - mesmo que a pergunta também mencione "funcionários",
# "colaboradores" ou outra palavra de entidade que hoje sequestra o filtro
# para doc_type=employee/customer por engano.
_PRODUCT_FULL_NAME_TO_KEY = {
    "vendefacil pdv": "pdv",
    "vendefacil estoque": "estoque",
    "vendefacil loja": "ecommerce",
    "vendefacil analytics": "analytics",
    "vendefacil pay": "pay",
}

_MONTH_NAME_TO_NUM = {
    "janeiro": "01", "fevereiro": "02", "marco": "03", "abril": "04",
    "maio": "05", "junho": "06", "julho": "07", "agosto": "08",
    "setembro": "09", "outubro": "10", "novembro": "11", "dezembro": "12",
}

_POLICY_TOPIC_HINTS = [
    "home office", "reembolso", "beneficio", "beneficios", "certificacao",
    "certificacoes", "treinamento", "treinamentos", "atendimento ao cliente",
    "lgpd", "seguranca da informacao", "seguranca cibernetica",
    "codigo de conduta",
]

# Correção (Etapa 4 - item 13): quando o tópico é inequívoco (só existe UM
# arquivo de política sobre aquilo), apontamos direto pro arquivo certo via
# `source_file`, em vez de confiar só na similaridade semântica dentro de
# ~20 chunks de política. Isso importa porque um trecho novo pode ser
# topicamente certo mas não repetir as mesmas palavras da pergunta (ex.:
# a seção de "conectividade e disponibilidade" do home office não repete a
# palavra "home office"), perdendo pra outro chunk que repete mais palavras
# da pergunta só por coincidência lexical.
_POLICY_HINT_TO_FILE = {
    # Termos mais específicos primeiro (Etapa 4 - item 13.1): "reembolso"
    # sozinho é ambíguo - existe reembolso a CLIENTE (reembolso.pdf) e
    # reembolso de CURSO a colaborador (beneficios_e_viagens.md). Checar
    # certificação/treinamento/curso antes evita que "reembolso de cursos e
    # certificações" (Q21) resolva pro arquivo errado só por causa da
    # palavra "reembolso" aparecer também aí.
    "certificacao": "unstructured/policies/beneficios_e_viagens.md",
    "certificacoes": "unstructured/policies/beneficios_e_viagens.md",
    "treinamento": "unstructured/policies/beneficios_e_viagens.md",
    "treinamentos": "unstructured/policies/beneficios_e_viagens.md",
    "beneficio": "unstructured/policies/beneficios_e_viagens.md",
    "beneficios": "unstructured/policies/beneficios_e_viagens.md",
    "home office": "unstructured/policies/home_office.md",
    "atendimento ao cliente": "unstructured/policies/atendimento_sla.md",
    "lgpd": "unstructured/policies/seguranca_lgpd.pdf",
    "seguranca da informacao": "unstructured/policies/seguranca_lgpd.pdf",
    "seguranca cibernetica": "unstructured/policies/seguranca_lgpd.pdf",
    "codigo de conduta": "unstructured/policies/codigo_de_conduta.md",
    "reembolso": "unstructured/policies/reembolso.pdf",
}


def contains(query: str, value: str) -> bool:
    """Evita encontrar palavras apenas por coincidência parcial."""
    return re.search(rf"\b{re.escape(value)}\b", query) is not None


class QueryAnalyzer:

    FIELDS = [
        "doc_type",
        "state",
        "module",
        "customer_id",
        "priority",
        "status",
        "date",
        "source_file",
    ]

    def __init__(self, vectorstore):
        self.valid_values = self._load_valid_values(vectorstore)
        self.company_to_customer_id = self._load_company_names(vectorstore)

    def _load_valid_values(self, vectorstore):
        """
        Descobre quais valores realmente existem nos metadados
        do índice FAISS.
        """
        values = {field: {} for field in self.FIELDS}

        for doc_id in vectorstore.index_to_docstore_id.values():
            doc = vectorstore.docstore.search(doc_id)

            if not hasattr(doc, "metadata"):
                continue

            for field in self.FIELDS:
                value = doc.metadata.get(field)

                if value is not None and str(value).strip():
                    values[field][normalize(str(value))] = str(value)

        return values

    def _load_company_names(self, vectorstore):
        """
        Correção (Etapa 4 - item 6): antes, customer_id só era extraído
        quando a pergunta citava o CÓDIGO literal (ex.: "CUST001"). Perguntas
        que citam o NOME da empresa (ex.: "Supermercado Boa Compra"), que é
        como a maioria das perguntas reais do benchmark se refere ao cliente,
        nunca resolviam nenhum filtro de customer_id. Aqui construímos um
        mapa nome-da-empresa -> customer_id a partir do metadado real
        `company_name` (adicionado aos chunks de customers.csv), sem inventar
        nenhum nome.
        """
        mapping = {}
        for doc_id in vectorstore.index_to_docstore_id.values():
            doc = vectorstore.docstore.search(doc_id)
            if not hasattr(doc, "metadata"):
                continue
            company_name = doc.metadata.get("company_name")
            customer_id = doc.metadata.get("customer_id")
            if company_name and customer_id:
                mapping[normalize(str(company_name))] = str(customer_id)
        return mapping

    def _validated_value(self, field: str, value: str):
        """Só devolve o valor se ele existir no índice."""
        return self.valid_values[field].get(normalize(value))

    def analyze(self, question: str) -> QueryFilters:
        query = normalize(question)

        filters = QueryFilters()

        # Tipo do documento.
        # Ordem de prioridade (Etapa 4 - item 4/5, ver comentários acima):
        # 1) tipos de CONTEÚDO explícitos (ticket/log/manual/ata/policy/
        #    email/sale) - se a pergunta já nomeia o tipo certo (ex.:
        #    "chamados"), essa é a pista mais confiável e não deve ser
        #    sobrescrita;
        # 2) só se nenhum tipo de conteúdo bateu, pistas de política (topic
        #    hints reais, tipo "home office"/"reembolso"), que cobrem
        #    perguntas de política que não usam a palavra "política";
        # 3) só por último, tipos de ENTIDADE (customer/employee/product/
        #    store), que são os mais propensos a aparecer como palavra
        #    incidental (ex.: "cliente", "lojas") sem pedir de fato aquele
        #    cadastro.
        for doc_type in _CONTENT_DOC_TYPES:
            aliases = DOC_TYPE_ALIASES[doc_type]
            if any(contains(query, normalize(alias)) for alias in aliases):
                value = self._validated_value("doc_type", doc_type)
                if value:
                    filters.doc_type = value
                    break

        if not filters.doc_type and any(
            contains(query, normalize(hint)) for hint in _POLICY_TOPIC_HINTS
        ):
            value = self._validated_value("doc_type", "policy")
            if value:
                filters.doc_type = value

        # Aponta direto pro arquivo quando o tópico é inequívoco (ver
        # comentário de _POLICY_HINT_TO_FILE acima). Só usa se o valor
        # realmente existir no índice - nunca inventa um caminho de arquivo.
        for hint, file_path in _POLICY_HINT_TO_FILE.items():
            if contains(query, normalize(hint)):
                value = self._validated_value("source_file", file_path)
                if value:
                    filters.source_file = value
                    break

        # Guarda técnica (Etapa 4 - item 7): se a pergunta cita um código de
        # erro real (ex.: "STK-409", "PAY-504") ou palavras de troubleshooting
        # ("erro", "conflito", "falha"), a resposta está num manual/log, não
        # num cadastro de loja/cliente/funcionário/produto - mesmo que uma
        # dessas palavras apareça incidentalmente (ex.: "entre lojas"). Nesse
        # caso, não deixamos um alias de ENTIDADE decidir o doc_type.
        has_error_code = re.search(r"\b[a-z]{2,5}-\d{2,5}\b", query) is not None
        has_troubleshoot_word = any(
            contains(query, w) for w in ["erro", "conflito", "falha", "timeout"]
        )
        # "Quem é ...": pergunta pede uma PESSOA (nome), então uma palavra de
        # entidade não-humana (produto/loja/cliente) que apareça incidentalmente
        # não deve travar o filtro - quem responde isso é employees.csv, achado
        # por busca semântica + o filtro de module (que continua sendo extraído
        # normalmente).
        is_who_question = contains(query, "quem")
        suppress_entity_doc_type = has_error_code or has_troubleshoot_word or is_who_question

        if not filters.doc_type and not suppress_entity_doc_type:
            for doc_type in _ENTITY_DOC_TYPES:
                aliases = DOC_TYPE_ALIASES[doc_type]
                if any(contains(query, normalize(alias)) for alias in aliases):
                    value = self._validated_value("doc_type", doc_type)
                    if value:
                        filters.doc_type = value
                        break

        # Estado por nome completo
        for state_name, acronym in STATE_ALIASES.items():
            if contains(query, state_name):
                value = self._validated_value("state", acronym)

                if value:
                    filters.state = value
                    break

        # Estado informado diretamente como UF
        if not filters.state:
            for acronym in STATE_ALIASES.values():
                if contains(query, acronym.lower()):
                    value = self._validated_value("state", acronym)

                    if value:
                        filters.state = value
                        break

        # Módulo.
        # Correção (Etapa 4 - item 10): perguntas "quem é" (Q02) nunca acham
        # a resposta em employees.csv se um filtro de module for aplicado,
        # porque os chunks de funcionários NUNCA têm campo `module` nos
        # metadados (só existe em customer/log/ticket/manual). Sem isso, o
        # filtro sozinho já exclui employees.csv mesmo com doc_type certo.
        if is_who_question:
            return filters

        for full_name, module_key in _PRODUCT_FULL_NAME_TO_KEY.items():
            if contains(query, full_name):
                value = self._validated_value("module", module_key)
                if value:
                    filters.module = value
                    break

        module_matches = []

        for normalized_value, original_value in self.valid_values["module"].items():

            clean_value = normalized_value.replace("vendefacil", "").strip()

            # Prioridade maior quando o valor real aparece diretamente na pergunta.
            if contains(query, normalized_value):
                module_matches.append((3, normalized_value, original_value))

            # Ex.: "VendeFácil Estoque" -> "estoque"
            elif clean_value and contains(query, clean_value):
                module_matches.append((2, normalized_value, original_value))

            # Fallback por palavras relevantes
            else:
                words = [
                    word
                    for word in normalized_value.split()
                    if len(word) >= 4 and word != "vendefacil"
                ]

                if any(contains(query, word) for word in words):
                    module_matches.append((1, normalized_value, original_value))

        if not filters.module and module_matches:
            # Maior prioridade primeiro.
            module_matches.sort(key=lambda item: item[0], reverse=True)
            filters.module = module_matches[0][2]

        # Customer ID, prioridade e status
        for field in ["customer_id", "priority", "status"]:
            for normalized_value, original_value in self.valid_values[field].items():

                if contains(query, normalized_value):
                    setattr(filters, field, original_value)
                    break

        # Cliente citado pelo NOME da empresa (não só pelo código CUST0xx).
        # Só tenta se ainda não achou customer_id pelo código.
        if not filters.customer_id:
            for company_normalized, customer_id in self.company_to_customer_id.items():
                if contains(query, company_normalized):
                    filters.customer_id = customer_id
                    break

        # Mês/ano citados na pergunta (Etapa 4 - item 12): perguntas sobre
        # atas de reunião (ex.: "retrospectiva ... de Fevereiro de 2026")
        # citam quando o evento aconteceu, e o metadado `date` das atas já
        # guarda isso no formato "AAAA-MM" (extraído do nome do arquivo em
        # loaders.py). Sem esse filtro, a busca precisa escolher a reunião
        # certa entre ~38 chunks de 18 reuniões diferentes só por
        # similaridade semântica, o que erra quando há temas parecidos
        # (várias reuniões falam de incidentes/TEF/PDV). Só usamos o valor
        # se ele realmente existir no índice - não inventamos combinação.
        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match and filters.doc_type == self._validated_value("doc_type", "ata"):
            month_num = next(
                (num for name, num in _MONTH_NAME_TO_NUM.items() if contains(query, name)),
                None,
            )
            if month_num:
                candidate = f"{year_match.group(1)}-{month_num}"
                value = self._validated_value("date", candidate)
                if value:
                    filters.date = value

        # Quando um cliente específico foi resolvido (por código ou por
        # nome), um doc_type=customer é redundante e prejudicial: a
        # informação sobre aquele cliente pode estar em tickets, e-mails,
        # atas ou lojas, não só no cadastro. Mantemos doc_type só quando ele
        # aponta pra um tipo de CONTEÚDO específico (ex.: doc_type=ticket),
        # que ainda é uma pista válida mesmo com customer_id já definido.
        if filters.customer_id and filters.doc_type == "customer":
            filters.doc_type = None

        return filters


if __name__ == "__main__":

    vectorstore = load_index()
    analyzer = QueryAnalyzer(vectorstore)

    questions = [
        "Quais tickets de clientes de Minas Gerais estão relacionados ao módulo de estoque?",
        "Mostre os chamados de São Paulo",
        "Quais tickets possuem prioridade alta?",
        "Mostre os tickets de Wakanda do módulo abacaxi",
    ]

    for question in questions:
        filters = analyzer.analyze(question)

        print(f"\nPergunta: {question}")
        print("Filtros:", filters.to_dict())
