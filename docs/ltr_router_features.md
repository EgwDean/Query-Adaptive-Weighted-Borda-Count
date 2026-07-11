# Learning-to-Rank Router Features: Lexical vs. Semantic Retrieval

Compiled feature list for a learning-to-rank router that chooses between lexical and semantic (dense) information retrieval, based on query performance prediction (QPP) and hybrid retrieval literature.

**Two status columns.**
- **Lex/sem routing?** — whether the signal has been used to decide/weight
  lexical vs. dense retrieval, *in any form* (a trained feature, a supervision
  label, or a plain number fed into a static function such as a sigmoid).
  **Yes** = there is such precedent (see "How used" + the Related-methods table);
  **No (QPP)** / **No (dense QPP)** = so far only a general query-difficulty
  predictor, which this project *repurposes* per retriever.
- **How used** — the mechanism: `Feature` (input to a trained model),
  `Label` (supervision target, needs qrels), `Static-function input` (a number
  plugged into a fixed formula to set the weight/decision), or `QPP score`
  (computed as a standalone difficulty estimate).

## Pre-retrieval features (query-only, no ranker run needed)

| Feature | Explanation | Formula | Reference | Lex/sem routing? | How used |
|---|---|---|---|---|---|
| Query length (ql) | The number of content (non-stopword) tokens in the query, \|Q\|. | Number of non-stop-word terms in Q | He & Ounis, "Query Performance Prediction," Inf. Process. Manage. 2006 | No (QPP) | QPP score / feature |
| Average IDF (avgIDF) | Sum over the query terms of idf(t)=log(N/f_t), divided by the number of query terms \|Q\|. | avgIDF = (1/\|Q\|) Σ_{t∈Q} idf(t), idf(t) = log(N/f_t) | Hauff, Hiemstra & de Jong, CIKM 2008; routing use: Query-Adaptive Hybrid Search, MAKE 2026 (doi:10.3390/make8040091) | **Yes** | Static-function input (sigmoid(avgIDF) → α); also QPP feature |
| Max IDF (maxIDF) | The largest idf(t)=log(N/f_t) among the query terms. | maxIDF = max_{t∈Q} idf(t) | Hauff, Hiemstra & de Jong, CIKM 2008 | No (QPP) | QPP score / feature |
| Std. dev. of IDF (γ1) | The standard deviation of the query terms' idf values, using idf(t)=log2((N+0.5)/N_t)/log2(N+1). | γ1 = σ_idf over {idf(t): t∈Q}, idf(t) = log2((N+0.5)/N_t) / log2(N+1) | He & Ounis, SPIRE 2004 | No (QPP) | QPP score / feature |
| Max/min IDF ratio (γ2) | The largest idf among the query terms divided by the smallest idf among them. | γ2 = idf_max / idf_min | He & Ounis, SPIRE 2004 | No (QPP) | QPP score / feature |
| Query scope (ω) | Minus the log of (n_Q/N), where n_Q = number of docs containing ≥1 query term and N = number of docs. | ω = −log(n_Q / N), n_Q = # docs containing ≥1 query term | Plachouras et al., TREC 2003; He & Ounis, SPIRE 2004 | No (QPP) | QPP score / feature |
| Simplified Clarity Score (SCS) | Sum over query words w of P_ml(w\|Q)·log2[P_ml(w\|Q)/P_coll(w)], with P_ml(w\|Q)=qtf/\|Q\| and P_coll(w)=tf_coll/tokens_coll. | SCS = Σ_{w∈Q} P_ml(w\|Q)·log2[P_ml(w\|Q)/P_coll(w)], P_ml(w\|Q)=qtf/ql, P_coll(w)=tf_coll/tokens_coll | He & Ounis, SPIRE 2004 | No (QPP) | QPP score / feature |
| Average Inverse Collection Term Frequency (AvICTF) | Average over query terms of log2(tokens_coll/tf_coll(t)) — i.e. log2 of the product of (tokens_coll/tf_coll(t)) over terms, divided by \|Q\|. | AvICTF = log2[ Π_{t∈Q}(tokens_coll/tf_coll(t)) ] / ql | Kwok, SIGIR 1996; He & Ounis, SPIRE 2004 | No (QPP) | QPP score / feature |
| SCQ per term | For a term t: (1+ln(f_{t,C}))·ln(1+N/f_t), where f_{t,C} = total occurrences of t in the collection and f_t = docs containing t. | SCQ(t) = (1+ln(f_{t,C}))·ln(1+N/f_t), f_{t,C}=collection term freq, f_t=doc frequency | Zhao, Scholer & Tsegay, ECIR 2008 | No (QPP) | QPP score / feature |
| SumSCQ / AvgSCQ / MaxSCQ | The sum, the sum÷\|Q\|, and the maximum of SCQ(t) over the query terms. | SumSCQ=Σ_{t∈Q}SCQ(t); AvgSCQ=SumSCQ/\|Q\|; MaxSCQ=max_t SCQ(t) | Zhao, Scholer & Tsegay, ECIR 2008 | No (QPP) | QPP score / feature |
| Term weight variance, VAR(t) | For a term t: sqrt[(1/f_t)·Σ_{d∋t}(w_{t,d}−w̄_t)²], with w_{t,d}=(1+ln tf_{t,d})·idf(t) and w̄_t the mean of w_{t,d} over docs containing t. | VAR(t) = sqrt[(1/f_t)Σ_{d∋t}(w_{t,d}−w̄_t)²], w_{t,d}=(1+ln tf_{t,d})·idf(t) | Zhao, Scholer & Tsegay, ECIR 2008 | No (QPP) | QPP score / feature |
| SumVAR / AvgVAR / MaxVAR | The sum, average (÷\|Q\|), and maximum of VAR(t) over the query terms. | Sum/Avg/Max of VAR(t) over query terms | Zhao, Scholer & Tsegay, ECIR 2008 | No (QPP) | QPP score / feature |
| Query term co-occurrence (PMI-based) | Average over all query-term pairs (t_i,t_j) of log[P(t_i,t_j)/(P(t_i)·P(t_j))], with probabilities from collection co-occurrence counts. | Avg. pointwise mutual information between all query term pairs in the collection | He, Ounis & Zhou et al., "Co-occurrence Based Predictors for Estimating Query Difficulty" | No (QPP) | QPP score / feature |
| Embedding-based query specificity | Average pairwise cosine similarity cos(e(t_i),e(t_j)) of the query terms' embedding vectors; optionally also avg/max cosine distance of each term embedding to the collection mean embedding. | Avg. pairwise cosine similarity of query-term word embeddings; also avg./max distance to collection centroid | Arabzadeh, Zarrinkalam, Jovanovic & Bagheri, ECIR 2020 | No (QPP) | QPP score / feature |

## Post-retrieval / score-distribution features (require running the ranker)

| Feature | Explanation | Formula | Reference | Lex/sem routing? | How used |
|---|---|---|---|---|---|
| Clarity Score (CS) | Sum over words w of P(w\|Q)·log2[P(w\|Q)/P(w\|Coll)], where P(w\|Q) is a query model estimated from the top docs and P(w\|Coll) the collection probability. | CS(Q) = Σ_w P(w\|Q)·log2[P(w\|Q)/P(w\|Coll)] | Cronen-Townsend, Zhou & Croft, SIGIR 2002 | No (QPP) | QPP score / feature |
| Weighted Information Gain (WIG) | (1/(k·√\|Q\|))·Σ over top-k docs of [score(Q,d)−score(Q,Corpus)] — the mean amount top-k scores exceed the corpus score, scaled by √\|Q\|. | WIG = (1/(k·√\|Q\|)) Σ_{d∈Top_k} [score(Q,d) − score(Q,Corpus)] | Zhou & Croft, SIGIR 2007 | No (QPP) | QPP score / feature |
| Normalized Query Commitment (NQC) | σ_k divided by \|score(Q,Corpus)\|, where σ_k=sqrt[(1/k)Σ(score(Q,d)−μ_k)²] over top-k and μ_k is the mean top-k score. | NQC = σ_k / \|score(Q,Corpus)\|, σ_k = sqrt[(1/k)Σ_{d∈Top_k}(score(Q,d)−μ_k)²] | Shtok, Kurland, Carmel, Raiber & Markovits, ACM TOIS 2012 | No (QPP) | QPP score / feature |
| Top (max) retrieval score | score(Q,d₁), the score of the rank-1 document. | score(Q, d₁), d₁ = top-ranked document | Zhou & Croft 2007; Shtok et al. 2012 | No (QPP) | QPP score / feature |
| Std. dev. of top-k scores (σ_k) | sqrt[(1/k)·Σ over top-k docs of (score(Q,d)−μ_k)²], the std. dev. of the top-k scores about their mean μ_k. | σ_k = sqrt[(1/k)Σ_{d∈Top_k}(score(Q,d)−μ_k)²] | Pérez-Iglesias & Araujo, ICTIR 2009; Cummins et al., SIGIR 2011 | No (QPP) | QPP score / feature |
| Score Magnitude and Variance (SMV) | (1/k)·Σ over top-k docs of (score(Q,d)/μ_k)·\|ln(score(Q,d)/μ_k)\|, with μ_k the mean top-k score. | SMV = (1/k) Σ_{d∈Top_k} (score(Q,d)/μ_k)·\|ln(score(Q,d)/μ_k)\| | Tao & Wu, CIKM 2014 | No (QPP) | QPP score / feature |
| Entropy of score distribution | −Σ over top-k docs of P(d\|Q)·log P(d\|Q), where P(d\|Q) = softmax of the top-k scores. | H(Q) = −Σ_{d∈D_k} P(d\|Q)·log P(d\|Q), P(d\|Q)=softmax(score(Q,d)) | Zendel, Liu, Culpepper & Scholer, QPP++ @ ECIR 2023 | No (QPP) | QPP score / feature |
| Query Feedback (QF) | \|Top_k(Q) ∩ Top_k(Q′)\| ÷ k — the overlap between the top-k list of Q and that of an expanded query Q′, divided by k. | QF = overlap@k(Top_k(Q), Top_k(Q′)) / k | Zhou & Croft, SIGIR 2007 | No (QPP) | QPP score / feature |
| Ranking Robustness | Rank correlation between the original top-k ranking and the ranking after adding noise to the document representations (averaged over noise draws). | Correlation between original and noise-perturbed top-k rankings | Zhou & Croft, CIKM 2006 | No (QPP) | QPP score / feature |
| Utility Estimation Framework (UEF) | ξ·φ(Q): a base predictor φ (e.g. NQC/WIG/Clarity) multiplied by ξ = Pearson correlation between the ranking and its relevance-model re-ranking. | UEF(Q,φ) = ξ(RM(Q), RM(θ_Q))·φ(Q) | Shtok, Kurland & Carmel, SIGIR 2010 | No (QPP) | QPP score / feature |
| Score autocorrelation (cluster-hypothesis measure) | A Moran's-I–like quantity Σ_{i,j} W_{ij}(s_i−s̄)(s_j−s̄) (normalized), where W_{ij} is the similarity between docs i,j and s_i their scores. | Spatial autocorrelation (Moran's I–like) of scores over a doc-similarity graph | Diaz, SIGIR 2007 | No (QPP) | QPP score / feature |
| Reference-list based estimation | A weighted average of the similarities between Q's score distribution and those of a set of reference queries with known effectiveness. | Weighted combination vs. reference queries of known effectiveness | Roitman et al., ICTIR 2017; Zendel et al., SIGIR 2019 | No (QPP) | QPP score / feature |
| Robust standard deviation estimation | σ_k recomputed after trimming/discarding the most extreme top-k scores (an outlier-resistant std. dev.). | Trimmed/robust variant of σ_k | Roitman, Erera & Weiner, ICTIR 2017 | No (QPP) | QPP score / feature |

## Neural / dense-retrieval–specific features

| Feature | Explanation | Formula | Reference | Lex/sem routing? | How used |
|---|---|---|---|---|---|
| BERT-QPP score | Linear(BERT([CLS] Q [SEP] d₁ [SEP])) — a linear layer on BERT's embedding of query+top doc d₁, trained to regress AP/nDCG. | φ(Q) = Linear(BERT([CLS] Q [SEP] d₁ [SEP])) | Arabzadeh, Khodabakhsh & Bagheri, CIKM 2021 | No (dense QPP) | Feature (trained regressor) |
| NeuralQPP (learned combination) | f_NN(φ₁(Q),…,φ_m(Q)) — a neural network mapping a vector of base QPP signals (WIG, NQC, Clarity, …) to a single score. | φ(Q) = f_NN(φ₁(Q),...,φ_m(Q)) | Zamani, Croft & Culpepper, SIGIR 2018 | No (QPP) | Feature-combiner (trained) |
| Relative Information Gain (RIG) | score(Q) minus the retrieval scores of automatically generated variant queries of Q (a relative gain). | Retrieval-score gain of Q vs. generated query variants | Datta, Ganguly, Mitra & Greene, ACM TOIS 2022 | No (QPP) | QPP score |
| Coherence-based dense predictor (A-Pair-Ratio) | [avg pairwise cosine among the top-ranked doc embeddings] ÷ [avg pairwise cosine among the bottom-ranked doc embeddings]. | Ratio of avg pairwise cosine (top docs) to avg pairwise cosine (bottom docs) | Vlachou & Macdonald, arXiv:2310.11405 / SIGIR-ICTIR 2024 | No (dense QPP) | QPP score |
| Embedding-based query specificity (dense) | Average (or maximum) cosine distance between the query embedding and its nearest document embeddings. | Avg./max cosine distance of query embedding to nearest doc embeddings | "QPP Using Neural Query Space Proximity," ACM TIST 2024 | No (dense QPP) | QPP score |

## Hybrid lexical/semantic routing–specific features

| Feature | Explanation | Formula | Reference | Lex/sem routing? | How used |
|---|---|---|---|---|---|
| Sparse–dense score margin (top-1 effectiveness gap) | α from the normalized gap between quality scores s_sparse, s_dense of the top-1 sparse and top-1 dense docs (e.g. α = s_sparse/(s_sparse+s_dense)); final = α·BM25 + (1−α)·dense. | Effectiveness scores of top-1 sparse & top-1 dense (via LLM judge), α = normalized gap; final = α·BM25 + (1−α)·dense | Hsu & Tzeng, "DAT," arXiv:2503.23013, 2025 | **Yes** | Static-function input (normalized gap → α; via LLM judge) |
| Sparse-retriever rank of first relevant doc (routing label) | F_q = the rank position of the first relevant passage in the top-K sparse list; route to sparse if F_q ≤ T, else dense. **Needs qrels → train-time label only.** | F_q = rank of first relevant passage in top-K sparse list; route sparse if F_q ≤ T | Arabzadeh, Yan & Clarke, CIKM 2021, arXiv:2109.10739 | **Yes** | Label (train-time target; not available at inference) |
| Query-only cross-encoder routing score | The output probability of a classifier applied to the contextual embedding of Q alone (no document), predicting sparse / dense / hybrid. | Classifier over contextual embedding of Q (no doc) → sparse/dense/hybrid | Arabzadeh, Yan & Clarke, CIKM 2021, arXiv:2109.10739 | **Yes** | Feature → trained classifier (inference-time predictor for the label above) |
| Rank-list agreement between retrievers | Kendall's τ between the sparse and dense rankings, or Jaccard@k = \|S_k ∩ D_k\| / \|S_k ∪ D_k\| for the two top-k sets S_k (sparse), D_k (dense). | Kendall's τ, or Jaccard@k = \|S_k ∩ D_k\| / \|S_k ∪ D_k\| | Kuzi, Zhang, Lin, Metzler & Nogueira, 2020 | **Yes** | Feature / hybrid-fusion diagnostic |

## Related routing / adaptive-weighting methods (how the signals are consumed)

Papers that select or weight lexical vs. dense retrieval per query, and the mechanism they use. Note scope: the last two rows select among *retrievers/models* generally, which is broader than lexical-vs-dense.

| Method | Query signal(s) used | How used | Reference |
|---|---|---|---|
| Query-Adaptive Hybrid Search | mean/avg IDF of query terms | Number → sigmoid → per-query (adaptive-RRF) weight (static function) | MAKE 2026, doi:10.3390/make8040091 |
| DAT: Dynamic Alpha Tuning | LLM-judged effectiveness of top-1 sparse & top-1 dense results | Number → normalized gap → α(q) (static function, LLM judge) | Hsu & Tzeng, arXiv:2503.23013, 2025 |
| Dense-vs-Sparse Strategy Selection | rank of first relevant in sparse list; query-only cross-encoder | Label (from qrels) trains a classifier used as the inference-time predictor | Arabzadeh, Yan & Clarke, CIKM 2021, arXiv:2109.10739 |
| LTRR: Learning to Rank Retrievers for LLMs | per-query features incl. QPP signals | Features → learning-to-rank model ordering/selecting retrievers | arXiv:2506.13743, 2025 |
| Selective Query Processing (QPP study) | QPP scores | Scores → per-query selection of retrieval pipeline | arXiv:2504.01101, 2025 |
| DenseQuest / dense-retriever selection | unsupervised QPP signals | Scores → pick best dense retriever (model selection) | arXiv:2407.06685; arXiv:2309.09403 |

---

**Notes on use:** Several of these features are highly correlated — avgIDF, SCS, AvICTF, and γ1 all capture "specificity/informativeness," while NQC, WIG, SMV, and entropy all capture "score dispersion." An ablation study will likely collapse each family down to one or two representatives. The post-retrieval group requires a chosen retrieval model and top-k results already computed, which fits naturally with a lexical/semantic router since both candidate lists are being scored anyway — this also allows computing each post-retrieval feature separately per ranker (e.g., NQC_lexical vs. NQC_semantic) as router inputs. A **"No (QPP)"** in the routing column means only that no cited work has *yet* used the signal to route lexical vs. dense — many of these (score margins, entropy, agreement, specificity) are strong candidate router inputs and are exactly what a Phase-2 ablation would test.
