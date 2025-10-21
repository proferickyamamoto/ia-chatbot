Roteiro da Atividade — Chatbot de Manutenção de Equipamentos Médicos (com comparação de modelos)

1) Objetivos de aprendizagem

Estruturar um corpus técnico (manuais/procedimentos) em JSON.

Implementar um mini-RAG (Retrieval-Augmented Generation) simples com embeddings + busca semântica.

Comparar a qualidade das respostas entre dois modelos LLM distintos e justificar resultados.

Documentar decisões, erros e limitações (responsabilidade e segurança em contexto médico).


2) Entregáveis

1. Repositório com:

kb/ (JSONs da base de conhecimento).

scripts/ (código do chatbot).

notebooks/ (opcional, experimentos).

results/ (respostas dos modelos, métricas e justificativas).



2. Relatório curto (2–3 páginas):

Metodologia, modelos escolhidos, avaliação, comparação e justificativas.



3. Tabela de avaliação preenchida (ver Seção 8).




---

3) Exemplo de documento JSON (base de consulta)

> Salve como kb/med_kb_example.json



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
    },
    {
      "type": "Bomba de Infusão",
      "brand": "InfuCare",
      "model": "IC-200",
      "common_faults": [
        {
          "code": "ERR-PRESS",
          "symptom": "Alarme de pressão na linha",
          "likely_causes": ["Linha dobrada/oclusa", "Bolhas no equipo", "Calibração de pressão vencida"],
          "diagnostics_steps": [
            "1) Verificar oclusão/dobras na linha.",
            "2) Executar purga para remover bolhas.",
            "3) Conferir etiqueta de calibração: validade 12 meses."
          ],
          "fix_steps": [
            "Reposicionar linha, remover dobras.",
            "Repetir purga; se persistir, substituir equipo.",
            "Se calibração vencida, encaminhar para metrologia."
          ],
          "parts": ["Equipo-IC", "Kit-Calib-P"]
        }
      ],
      "preventive": [
        "Calibração anual do transdutor de pressão.",
        "Teste funcional mensal (fluxo nominal ±5%)."
      ]
    },
    {
      "type": "Desfibrilador",
      "brand": "CardioPlus",
      "model": "DF-10",
      "common_faults": [
        {
          "code": "LOW-BATT",
          "symptom": "Bateria descarregando rapidamente",
          "likely_causes": ["Bateria fim de vida", "Carregador interno com falha"],
          "diagnostics_steps": [
            "1) Executar autoteste (segurar TEST por 5s).",
            "2) Checar ciclos de carga (>500 ciclos sugere troca)."
          ],
          "fix_steps": [
            "Substituir bateria (cód. BAT-DF10).",
            "Se persistir, verificar carregador interno (board CHG-10)."
          ],
          "parts": ["BAT-DF10", "CHG-10"]
        }
      ],
      "preventive": [
        "Ensaios de carga mensal e ciclo completo trimestral.",
        "Armazenar entre 20–25 °C, umidade < 60%."
      ]
    }
  ]
}

> Opcional: criem mais arquivos JSON (um por fabricante) para ampliar a base.




---

4) Cenário de consulta (ticket do técnico)

> Salve como kb/sample_tickets.json



[
  {
    "id": "TCK-001",
    "equipment": {"type": "Ventilador Pulmonar", "model": "VT-500"},
    "question": "Alarme AL01 com fluxo insuficiente. O que devo checar e qual peça provavelmente será trocada?"
  },
  {
    "id": "TCK-002",
    "equipment": {"type": "Bomba de Infusão", "model": "IC-200"},
    "question": "Alarme de pressão na linha. Já reposicionei a linha, o que mais posso fazer?"
  },
  {
    "id": "TCK-003",
    "equipment": {"type": "Desfibrilador", "model": "DF-10"},
    "question": "Bateria cai muito rápido mesmo após recarga. Próximos passos?"
  }
]


---

5) Implementação (tutorial passo-a-passo)

5.1. Ambiente

# criar venv (opcional)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -U transformers sentence-transformers faiss-cpu pandas numpy scikit-learn

> Observação: vocês podem usar qualquer dois modelos compatíveis com transformers (ex.: “Modelo A” e “Modelo B”). Se preferirem usar APIs externas, adaptem a função generate_with_model().



5.2. Estrutura sugerida

project/
  kb/
    med_kb_example.json
    sample_tickets.json
  scripts/
    build_index.py
    rag_chatbot.py
    compare_models.py
  results/
    runs/
  README.md

5.3. Código — construção do índice (FAISS + MiniLM)

> scripts/build_index.py



import json, os, glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

KB_DIR = "kb"
INDEX_OUT = "kb/faiss_index.bin"
DOCS_OUT = "kb/docs.npy"
META_OUT = "kb/meta.json"

def load_kb():
    chunks, meta = [], []
    for fp in glob.glob(os.path.join(KB_DIR, "*.json")):
        if fp.endswith("sample_tickets.json"):  # não indexar tickets
            continue
        data = json.load(open(fp, "r", encoding="utf-8"))
        for eq in data["equipments"]:
            header = f'{eq["type"]} | {eq["brand"]} {eq["model"]}'
            # gerar chunks simples (poderiam usar chunking mais sofisticado)
            if "common_faults" in eq:
                for f in eq["common_faults"]:
                    chunk = (
                        f"{header}\n"
                        f"Falha {f.get('code','')}: {f['symptom']}\n"
                        f"Causas prováveis: {', '.join(f['likely_causes'])}\n"
                        f"Diagnóstico: {' '.join(f['diagnostics_steps'])}\n"
                        f"Correção: {' '.join(f['fix_steps'])}\n"
                        f"Peças: {', '.join(f['parts'])}"
                    )
                    chunks.append(chunk)
                    meta.append({"model": eq["model"], "type": eq["type"], "code": f.get("code","")})
            if "preventive" in eq:
                chunks.append(f"{header}\nPreventiva: " + " ".join(eq["preventive"]))
                meta.append({"model": eq["model"], "type": eq["type"], "section": "preventive"})
    return chunks, meta

def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    chunks, meta = load_kb()
    emb = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # dot product com vetores normalizados ~ cos
    index.add(emb)
    faiss.write_index(index, INDEX_OUT)
    np.save(DOCS_OUT, np.array(chunks, dtype=object))
    json.dump(meta, open(META_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Indexados {len(chunks)} chunks.")

if __name__ == "__main__":
    main()

5.4. Código — função RAG e geração com modelo

> scripts/rag_chatbot.py



import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

KB_DIR = "kb"
INDEX_FP = "kb/faiss_index.bin"
DOCS_FP = "kb/docs.npy"

def load_retriever():
    index = faiss.read_index(INDEX_FP)
    docs = np.load(DOCS_FP, allow_pickle=True)
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return index, docs, emb_model

def retrieve(question, index, docs, emb_model, k=4):
    q = emb_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q, k)
    ctx = [docs[i] for i in idx[0]]
    return ctx, scores[0].tolist()

def load_model(model_name_or_path):
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    mdl = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto")
    return tok, mdl

PROMPT = """Você é um assistente técnico de manutenção hospitalar.
Use SOMENTE as informações do CONTEXTO. 
Se não houver informação suficiente, diga claramente o que falta e recomende escalar para um técnico qualificado.

PERGUNTA: {question}

CONTEXTO:
{context}

Responda de forma passo a passo e inclua (quando houver) código de falha, diagnóstico e peças.
"""

@torch.inference_mode()
def generate_with_model(model_name, question, context_blocks, max_new_tokens=250):
    tok, mdl = load_model(model_name)
    context = "\n---\n".join(context_blocks)
    prompt = PROMPT.format(question=question, context=context)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    # Remover o prompt do início (depende do modelo)
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text

def answer(question, model_name, k=4):
    index, docs, emb_model = load_retriever()
    ctx, scores = retrieve(question, index, docs, emb_model, k=k)
    ans = generate_with_model(model_name, question, ctx)
    return ans, ctx, scores

if __name__ == "__main__":
    q = "Alarme AL01 no VT-500: o que checar e qual peça provavelmente será trocada?"
    ans, ctx, scores = answer(q, "MODEL_NAME_HERE")
    print("== CONTEXTO ==")
    for c in ctx: print("-", c[:200].replace("\n"," ") + "...")
    print("\n== RESPOSTA ==")
    print(ans)

5.5. Código — comparação entre dois modelos

> scripts/compare_models.py



import json, os, time, pandas as pd
from rag_chatbot.py import answer  # ou: from rag_chatbot import answer

TICKETS_FP = "kb/sample_tickets.json"
RESULTS_DIR = "results/runs"

def run_eval(model_a, model_b, topk=4):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tickets = json.load(open(TICKETS_FP, "r", encoding="utf-8"))
    rows = []
    for t in tickets:
        q = t["question"]
        for model in [model_a, model_b]:
            start = time.time()
            ans, ctx, scores = answer(q, model, k=topk)
            lat = time.time() - start
            rows.append({
                "ticket_id": t["id"],
                "model": model,
                "question": q,
                "answer": ans,
                "latency_s": round(lat, 2),
                "top_docs": ctx,
                "top_scores": scores
            })
    df = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_csv = os.path.join(RESULTS_DIR, f"compare_{ts}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Resultados salvos em {out_csv}")
    return df

if __name__ == "__main__":
    # substituam pelos nomes dos modelos escolhidos (ex.: "Modelo A", "Modelo B")
    df = run_eval("MODEL_A_NAME", "MODEL_B_NAME", topk=4)
    print(df[["ticket_id","model","latency_s"]])

> Importante: coloquem nomes reais dos modelos no lugar de MODEL_A_NAME e MODEL_B_NAME (qualquer dois que vocês tenham acesso). Se usarem modelos instruídos (instruct), melhores resultados.




---

6) Como executar (passo a passo)

1. Montar a base: coloquem os JSONs em kb/.


2. Construir índice:

python scripts/build_index.py


3. Testar manualmente o RAG:

python scripts/rag_chatbot.py

Troquem MODEL_NAME_HERE pelo modelo escolhido e observem contexto recuperado vs resposta.



4. Rodar comparação:

python scripts/compare_models.py

Preencham MODEL_A_NAME e MODEL_B_NAME. O script salva um CSV com as respostas e latências.





---

7) Como justificar resultados (guia para o relatório)

Cobertura do contexto: a resposta citou passos/peças presentes no contexto recuperado?

Cadeia de raciocínio técnico: os diagnósticos e correções seguem uma ordem segura (diagnóstico → correção → peças)?

Aderência a segurança: menciona desligamento, qualificação do técnico, escalação quando faltar info?

Consistência: entre tickets similares, o modelo mantém coerência?

Latência: tempo de resposta vs qualidade.

Falhas: onde o modelo alucinou? (ex.: peça que não existe no contexto).


Peçam aos alunos para apontar trechos do contexto que embasam a resposta. Se o modelo sugeriu algo fora do contexto, marcar como alucinação e explicar.


---

8) Tabela de avaliação (preencher)

Critério	Medida sugerida	Modelo A	Modelo B

Exatidão factual	% de respostas com passos/peças corretos (validados no JSON)		
Cobertura do contexto	% de respostas que usam ao menos 1–2 trechos recuperados		
Segurança	% de respostas que incluem aviso/limite/escalação quando necessário		
Clareza	Nota 1–5 (rubrica do time)		
Latência	s (média)		
Alucinações	# por 10 respostas		
Preferência Humana	Votos de avaliadores (A vs B)		


> Rubrica rápida (1–5): 1 = confuso/incompleto; 3 = adequado com pequenos problemas; 5 = claro, completo, seguro.




---

9) Dicas & variações

Sem RAG vs com RAG: peçam um teste extra desativando a recuperação (contexto vazio) para ver o efeito do RAG.

Top-K: experimentem k=2,4,6 e observem mudanças (cobertura vs ruído).

Chunking: troquem o chunking simples por um baseado em títulos ou tabelas.

Embeddings: comparem all-MiniLM-L6-v2 com outro encoder (impacto na recuperação).

Avaliação adicional: façam uma lista de perguntas fora da base para checar se o bot recusa corretamente.



---

10) Observações de segurança e ética (contexto médico)

O chatbot não substitui o manual, nem um técnico qualificado.

Sempre incluir avisos de segurança; se faltar informação, não inventar: orientar a escalar.

Registrem no relatório limitações conhecidas.
