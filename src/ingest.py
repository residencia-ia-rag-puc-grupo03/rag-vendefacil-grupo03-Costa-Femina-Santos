import glob
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from loaders import (
    load_customers_csv,
    load_employees_csv,
    load_generic_csv,
    load_system_logs_csv,
    load_products_json,
    load_stores_json,
    load_tickets_jsonl,
    load_markdown,
    load_pdf,
    load_email_txt,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

POLICIES_WITH_PDF_VERSION = {"reembolso", "seguranca_lgpd"}


def load_all_documents():
    docs = []


    structured = os.path.join(DATA_DIR, "structured")

    customers_path = os.path.join(structured, "customers.csv")
    if os.path.exists(customers_path):
        docs += load_customers_csv(customers_path)

    employees_path = os.path.join(structured, "employees.csv")
    if os.path.exists(employees_path):
        docs += load_employees_csv(employees_path)

    sales_path = os.path.join(structured, "sales.csv")
    if os.path.exists(sales_path):
        docs += load_generic_csv(sales_path, doc_type="sale", sensitivity="interno",
                                  source_label="structured/sales.csv")

    products_path = os.path.join(structured, "products.json")
    if os.path.exists(products_path):
        docs += load_products_json(products_path)

    stores_path = os.path.join(structured, "stores.json")
    if os.path.exists(stores_path):
        docs += load_stores_json(stores_path)


    semi = os.path.join(DATA_DIR, "semi_structured")

    logs_path = os.path.join(semi, "system_logs.csv")
    if os.path.exists(logs_path):
        docs += load_system_logs_csv(logs_path, sensitivity="interno",
                                      source_label="semi_structured/system_logs.csv")

    tickets_path = os.path.join(semi, "tickets.jsonl")
    if os.path.exists(tickets_path):
        docs += load_tickets_jsonl(tickets_path)


    documentation_dir = os.path.join(DATA_DIR, "unstructured", "documentation")
    if os.path.isdir(documentation_dir):
        for module_name in os.listdir(documentation_dir):
            module_dir = os.path.join(documentation_dir, module_name)
            if not os.path.isdir(module_dir):
                continue
            for md_path in glob.glob(os.path.join(module_dir, "*.md")):
                source_label = f"unstructured/documentation/{module_name}/{os.path.basename(md_path)}"
                docs += load_markdown(
                    md_path,
                    doc_type="manual",
                    source_label=source_label,
                    sensitivity="interno",
                    extra_meta={"module": module_name},
                )


    meetings_dir = os.path.join(DATA_DIR, "unstructured", "meetings")
    if os.path.isdir(meetings_dir):
        for md_path in glob.glob(os.path.join(meetings_dir, "*.md")):
            filename = os.path.basename(md_path)
            # nomes tipo 2026-01-product_roadmap.md -> extrai a data
            date_part = filename[:7] if len(filename) > 7 else None
            source_label = f"unstructured/meetings/{filename}"
            docs += load_markdown(
                md_path,
                doc_type="ata",
                source_label=source_label,
                sensitivity="interno",
                extra_meta={"date": date_part} if date_part else None,
            )


    policies_dir = os.path.join(DATA_DIR, "unstructured", "policies")
    if os.path.isdir(policies_dir):
        for md_path in glob.glob(os.path.join(policies_dir, "*.md")):
            base_name = os.path.splitext(os.path.basename(md_path))[0]
            if base_name in POLICIES_WITH_PDF_VERSION:
                continue  # a versão em PDF já cobre esse conteúdo
            sensitivity = "restrito" if base_name == "codigo_de_conduta" else "interno"
            source_label = f"unstructured/policies/{os.path.basename(md_path)}"
            docs += load_markdown(
                md_path, doc_type="policy", source_label=source_label, sensitivity=sensitivity
            )

        for pdf_path in glob.glob(os.path.join(policies_dir, "*.pdf")):
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            sensitivity = "restrito" if base_name == "seguranca_lgpd" else "interno"
            source_label = f"unstructured/policies/{os.path.basename(pdf_path)}"
            docs += load_pdf(pdf_path, sensitivity=sensitivity, source_label=source_label)


    emails_dir = os.path.join(DATA_DIR, "unstructured", "emails")
    if os.path.isdir(emails_dir):
        for txt_path in glob.glob(os.path.join(emails_dir, "*.txt")):
            source_label = f"unstructured/emails/{os.path.basename(txt_path)}"
            docs += load_email_txt(txt_path, source_label=source_label)

    return docs


def main():
    print("Carregando e processando documentos de ./data ...")
    documents = load_all_documents()
    print(f"Total de chunks gerados: {len(documents)}")

    if not documents:
        print("Nenhum documento encontrado. Confira se está rodando este script a partir da raiz do projeto.")
        return

    print("Gerando embeddings e indexando no FAISS (pode levar alguns minutos na primeira vez)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"Índice salvo em: {INDEX_DIR}")


if __name__ == "__main__":
    main()
