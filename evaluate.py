# evaluate.py
import json
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np

# Load predictions
with open("artifacts/predictions_longragsum.jsonl") as f:
    data = [json.loads(line) for line in f]

refs = [d["reference_summary"] for d in data]
gens = [d["generated_summary"] for d in data]

# ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
r1, r2, rl = [], [], []
for ref, gen in zip(refs, gens):
    scores = scorer.score(ref, gen)
    r1.append(scores['rouge1'].fmeasure)
    r2.append(scores['rouge2'].fmeasure)
    rl.append(scores['rougeL'].fmeasure)

# BERTScore
P, R, F1 = bert_score(gens, refs, lang="en", verbose=False)

print("LONG-RAG-SUM RESULTS (GovReport Test)")
print("="*60)
print(f"ROUGE-1:  {np.mean(r1):.3f}")
print(f"ROUGE-2:  {np.mean(r2):.3f}")
print(f"ROUGE-L:  {np.mean(rl):.3f}")
print(f"BERTScore: {F1.mean().item():.4f}")