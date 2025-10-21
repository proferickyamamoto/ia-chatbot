# 🩺 Chatbot de Manutenção de Equipamentos Médicos — RAG + Comparação de Modelos

## 📘 Descrição

Este projeto tem como objetivo construir um **chatbot técnico** que auxilia profissionais de manutenção hospitalar na **diagnóstico e resolução de falhas** em equipamentos médicos, utilizando **RAG (Retrieval-Augmented Generation)**.  

Os alunos deverão:
1. Criar uma **base de conhecimento** em formato `.json` com dados de manutenção;
2. Implementar um **chatbot** que consulta a base;
3. **Comparar dois modelos de linguagem** (ex.: *Falcon*, *LLaMA*, *Mistral*, *Gemma*, etc.);
4. **Justificar diferenças** nos resultados, analisando precisão, clareza e segurança.

---

## 🎯 Objetivos de Aprendizagem

- Estruturar um **corpus técnico** (manual de manutenção) em JSON;
- Implementar um **mini-RAG** com embeddings e busca semântica;
- Avaliar e comparar **modelos de linguagem** em contexto técnico;
- Discutir **ética e segurança** no uso de IA em aplicações médicas.

---

## 🧩 Estrutura do Projeto

project/ ├── kb/ │   ├── med_kb_example.json │   ├── sample_tickets.json │ ├── scripts/ │   ├── build_index.py │   ├── rag_chatbot.py │   └── compare_models.py │ ├── results/ │   └── runs/ │ └── README.md

---

## 📁 Exemplo de Base de Conhecimento (`med_kb_example.json`)

```json
{
  "metadata": {
    "source": "Manual interno de manutenção",
    "version": "1.0",
    "last_update": "2025-10-01",
    "safety_notice": "Somente técnicos qualificados devem executar procedimentos. Desconecte da rede elétrica antes de abrir o equipamento."
  },
  "equipments": [
    {
      "type": "Ventilador Pulmonar",
      "brand": "RespiraTech",
      "model": "VT-500",
      "common_faults": [
        {
          "code": "AL01",
          "symptom": "Fluxo de ar insuficiente",
          "likely_causes": ["Filtro HEPA obstruído", "Vazamento na tubulação", "Sensor de fluxo defeituoso"],
          "diagnostics_steps": [
            "1) Verificar integridade das traqueias e conexões.",
            "2) Inspecionar e substituir filtro HEPA se diferencial de pressão > 250 Pa.",
            "3) Rodar autoteste de sensores (menu > manutenção > autoteste)."
          ],
          "fix_steps": [
            "Substituir filtro HEPA (código peça: HEPA-RT-500).",
            "Apertar conexões e substituir traqueia se rachada.",
            "Se autoteste falhar em sensor, solicitar troca do sensor de fluxo (cód. SF-22)."
          ],
          "parts": ["HEPA-RT-500", "Traqueia-PVC-22", "SF-22"]
        }
      ],
      "preventive": [
        "Trocar filtro HEPA a cada 1000h ou 6 meses (o que ocorrer primeiro).",
        "Limpeza externa semanal com álcool isopropílico 70% (não usar cloro)."
      ]
    }
  ]
}


---

📄 Exemplo de Consulta (sample_tickets.json)

[
  {
    "id": "TCK-001",
    "equipment": {"type": "Ventilador Pulmonar", "model": "VT-500"},
    "question": "Alarme AL01 com fluxo insuficiente. O que devo checar e qual peça provavelmente será trocada?"
  },
  {
    "id": "TCK-002",
    "equipment": {"type": "Desfibrilador", "model": "DF-10"},
    "question": "Bateria descarregando rapidamente, o que devo fazer?"
  }
]


---

⚙️ Passo a Passo da Implementação

1️⃣ Instalar Dependências

python -m venv .venv
source .venv/bin/activate      # (Windows: .venv\Scripts\activate)
pip install -U transformers sentence-transformers faiss-cpu pandas numpy scikit-learn torch


---

2️⃣ Construir o Índice de Conhecimento

> 📁 scripts/build_index.py



import json, os, glob
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

KB_DIR = "kb"
INDEX_OUT = "kb/faiss_index.bin"
DOCS_OUT = "kb/docs.npy"
META_OUT = "kb/meta.json"

def load_kb():
    chunks, meta = [], []
    for fp in glob.glob(os.path.join(KB_DIR, "*.json")):
        if fp.endswith("sample_tickets.json"):
            continue
        data = json.load(open(fp, "r", encoding="utf-8"))
        for eq in data["equipments"]:
            header = f'{eq["type"]} | {eq["brand"]} {eq["model"]}'
            for f in eq.get("common_faults", []):
                chunk = (
                    f"{header}\n"
                    f"Falha {f['code']}: {f['symptom']}\n"
                    f"Causas: {', '.join(f['likely_causes'])}\n"
                    f"Diagnóstico: {' '.join(f['diagnostics_steps'])}\n"
                    f"Correção: {' '.join(f['fix_steps'])}"
                )
                chunks.append(chunk)
                meta.append({"model": eq["model"], "code": f["code"]})
    return chunks, meta

def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    chunks, meta = load_kb()
    emb = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, INDEX_OUT)
    np.save(DOCS_OUT, np.array(chunks, dtype=object))
    json.dump(meta, open(META_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Indexados {len(chunks)} documentos.")

if __name__ == "__main__":
    main()

Execute:

python scripts/build_index.py


---

3️⃣ Criar o Chatbot RAG

> 📁 scripts/rag_chatbot.py



import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

PROMPT = """Você é um assistente técnico de manutenção hospitalar.
Use SOMENTE o contexto fornecido. 
Se não houver informação suficiente, oriente o técnico a escalar o caso.

Pergunta: {question}

Contexto:
{context}

Responda de forma técnica e segura.
"""

def retrieve(question, k=3):
    index = faiss.read_index("kb/faiss_index.bin")
    docs = np.load("kb/docs.npy", allow_pickle=True)
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q = emb_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q, k)
    return [docs[i] for i in idx[0]]

def generate_answer(model_name, question, context_blocks):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    prompt = PROMPT.format(question=question, context="\n---\n".join(context_blocks))
    inputs = tok(prompt, return_tensors="pt")
    out = mdl.generate(**inputs, max_new_tokens=250)
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    question = "Alarme AL01 com fluxo insuficiente no VT-500."
    ctx = retrieve(question)
    answer = generate_answer("MODEL_NAME_HERE", question, ctx)
    print("=== Contexto ===")
    print(ctx)
    print("\n=== Resposta ===")
    print(answer)


---

4️⃣ Comparar Dois Modelos

> 📁 scripts/compare_models.py



import json, pandas as pd, time
from rag_chatbot import retrieve, generate_answer

def run_eval(model_a, model_b):
    tickets = json.load(open("kb/sample_tickets.json", "r", encoding="utf-8"))
    results = []
    for t in tickets:
        for model in [model_a, model_b]:
            q = t["question"]
            ctx = retrieve(q)
            start = time.time()
            ans = generate_answer(model, q, ctx)
            latency = round(time.time() - start, 2)
            results.append({"id": t["id"], "model": model, "question": q, "answer": ans, "latency_s": latency})
    df = pd.DataFrame(results)
    df.to_csv("results/compare_results.csv", index=False)
    print("Resultados salvos em results/compare_results.csv")

if __name__ == "__main__":
    run_eval("MODEL_A_NAME", "MODEL_B_NAME")


---

5️⃣ Executar os Testes

python scripts/rag_chatbot.py
python scripts/compare_models.py


---

📊 Avaliação e Relatório

Cada grupo deverá produzir um relatório técnico (2–3 páginas) com:

Metodologia utilizada;

Modelos comparados;

Exemplos de perguntas e respostas;

Análise de coerência, segurança e precisão;

Casos de alucinação (respostas fora do contexto);

Tabela de comparação:


Critério	Modelo A	Modelo B

Exatidão técnica		
Clareza		
Segurança		
Latência (s)		
Alucinações		
Preferência geral		



---

🧠 Reflexão Final

Os alunos devem responder:

1. O modelo mais rápido foi o mais confiável?


2. O chatbot manteve a segurança técnica nas respostas?


3. Quais limitações e riscos existem em automatizar instruções médicas?


4. Como melhorar o sistema (dados, embeddings, RAG, filtros)?




---

⚠️ Observações de Segurança

> Este chatbot é apenas para fins educacionais.
Não substitui o julgamento técnico nem autoriza qualquer ação em equipamentos reais.
Sempre validar as instruções com manuais oficiais e profissionais qualificados.
