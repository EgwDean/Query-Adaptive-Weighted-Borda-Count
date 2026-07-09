# Learning-to-Rank Router Features: Lexical vs. Semantic Retrieval

Compiled feature list for a learning-to-rank router that chooses between lexical and semantic (dense) information retrieval, based on query performance prediction (QPP) and hybrid retrieval literature.

## Pre-retrieval features (query-only, no ranker run needed)

| Feature | Explanation | Formula | Reference |
|---|---|---|---|
| Query length (ql) | Count of content (non-stopword) terms; short keyword queries lean lexical, long natural-language ones lean dense. | Number of non-stop-word terms in Q | He & Ounis, "Query Performance Prediction," Inf. Process. Manage. 2006 |
| Average IDF (avgIDF) | Mean rarity of the query terms; high means specific, discriminative vocabulary that rewards exact lexical match. | avgIDF = (1/\|Q\|) Σ_{t∈Q} idf(t), idf(t) = log(N/f_t) | Hauff, Hiemstra & de Jong, "A Survey of Pre-retrieval Query Performance Predictors," CIKM 2008 |
| Max IDF (maxIDF) | Rarity of the single rarest term; flags queries hinging on one highly specific word. | maxIDF = max_{t∈Q} idf(t) | Hauff, Hiemstra & de Jong, CIKM 2008 |
| Std. dev. of IDF (γ1) | How uneven the term rarities are across the query (mix of rare and common vs uniform). | γ1 = σ_idf over {idf(t): t∈Q}, idf(t) = log2((N+0.5)/N_t) / log2(N+1) | He & Ounis, "Inferring Query Performance Using Pre-retrieval Predictors," SPIRE 2004 |
| Max/min IDF ratio (γ2) | How dominant the rarest term is relative to the commonest one. | γ2 = idf_max / idf_min | He & Ounis, SPIRE 2004 |
| Query scope (ω) | Fraction of the collection touched by the query terms; small scope = a specific, narrowing query. | ω = −log(n_Q / N), n_Q = # docs containing ≥1 query term | Plachouras et al., TREC 2003; used as predictor in He & Ounis, SPIRE 2004 |
| Simplified Clarity Score (SCS) | Divergence of the query's word distribution from the collection's; high = a focused, specific query. | SCS = Σ_{w∈Q} P_ml(w\|Q)·log2[P_ml(w\|Q)/P_coll(w)], P_ml(w\|Q)=qtf/ql, P_coll(w)=tf_coll/tokens_coll | He & Ounis, SPIRE 2004 |
| Average Inverse Collection Term Frequency (AvICTF) | Average term informativeness using collection token frequencies; another specificity signal. | AvICTF = log2[ Π_{t∈Q}(tokens_coll/tf_coll(t)) ] / ql | Kwok, "A New Method of Weighting Query Terms," SIGIR 1996; He & Ounis, SPIRE 2004 |
| SCQ per term | Per-term collection-query similarity combining term frequency and document rarity. | SCQ(t) = (1+ln(f_{t,C}))·ln(1+N/f_t), f_{t,C}=collection term freq, f_t=doc frequency | Zhao, Scholer & Tsegay, "Effective Pre-retrieval QPP Using Similarity and Variability Evidence," ECIR 2008 |
| SumSCQ / AvgSCQ / MaxSCQ | Total, average and peak SCQ across the query terms. | SumSCQ=Σ_{t∈Q}SCQ(t); AvgSCQ=SumSCQ/\|Q\|; MaxSCQ=max_t SCQ(t) | Zhao, Scholer & Tsegay, ECIR 2008 |
| Term weight variance, VAR(t) | How much a term's TF-IDF weight varies across the docs containing it; high = strongly discriminating term. | VAR(t) = sqrt[(1/f_t)Σ_{d∋t}(w_{t,d}−w̄_t)²], w_{t,d}=(1+ln tf_{t,d})·idf(t) | Zhao, Scholer & Tsegay, ECIR 2008 |
| SumVAR / AvgVAR / MaxVAR | Total, average and peak term-weight variance over the query terms. | Sum/Avg/Max of VAR(t) over query terms | Zhao, Scholer & Tsegay, ECIR 2008 |
| Query term co-occurrence (PMI-based) | Whether the query terms naturally co-occur (coherent phrase) vs an unusual combination. | Avg. pointwise mutual information between all query term pairs in the collection | He, Ounis & Zhou et al., "Co-occurrence Based Predictors for Estimating Query Difficulty" |
| Embedding-based query specificity | Embedding-space specificity: how tightly query terms cluster and how far they sit from the collection centroid. | Avg. pairwise cosine similarity of query-term word embeddings; also avg./max distance to collection centroid | Arabzadeh, Zarrinkalam, Jovanovic & Bagheri, "Neural Embedding-Based Metrics for Pre-retrieval QPP," ECIR 2020 |

## Post-retrieval / score-distribution features (require running the ranker)

| Feature | Explanation | Formula | Reference |
|---|---|---|---|
| Clarity Score (CS) | Topic coherence of the top results vs the collection; high = the retriever locked onto a clear topic. | CS(Q) = Σ_w P(w\|Q)·log2[P(w\|Q)/P(w\|Coll)], P(w\|Q) from a relevance/query model over top docs | Cronen-Townsend, Zhou & Croft, "Predicting Query Performance," SIGIR 2002 |
| Weighted Information Gain (WIG) | How far top-k scores exceed a whole-corpus baseline; a large gap signals confident separation. | WIG = (1/(k·√\|Q\|)) Σ_{d∈Top_k} [score(Q,d) − score(Q,Corpus)] | Zhou & Croft, "Query Performance Prediction in Web Search Environments," SIGIR 2007 |
| Normalized Query Commitment (NQC) | Top-k score spread normalized by the corpus score; a clear standout signals a good retrieval. | NQC = σ_k / \|score(Q,Corpus)\|, σ_k = sqrt[(1/k)Σ_{d∈Top_k}(score(Q,d)−μ_k)²] | Shtok, Kurland, Carmel, Raiber & Markovits, "Predicting Query Performance by Query-Drift Estimation," ACM TOIS 2012 |
| Top (max) retrieval score | Raw score of the #1 document; a simple confidence proxy. | score(Q, d₁), d₁ = top-ranked document | Used as base signal in Zhou & Croft 2007; Shtok et al. 2012 |
| Std. dev. of top-k scores (σ_k) | Dispersion of the top-k scores; a wide spread means a more decisive ranking. | σ_k = sqrt[(1/k)Σ_{d∈Top_k}(score(Q,d)−μ_k)²] | Pérez-Iglesias & Araujo, "Ranking List Dispersion as a QPP," ICTIR 2009; "SD as a Query Hardness Estimator," SPIRE 2010; Cummins, Jose & O'Riordan, SIGIR 2011 |
| Score Magnitude and Variance (SMV) | Combines top-score magnitude and variance into one hardness estimate. | SMV = (1/k) Σ_{d∈Top_k} (score(Q,d)/μ_k)·\|ln(score(Q,d)/μ_k)\| | Tao & Wu, "Query Performance Prediction by Considering Score Magnitude and Variance Together," CIKM 2014 |
| Entropy of score distribution | Softmax entropy of top-k scores; low = confident (mass on few docs), high = uncertain. | H(Q) = −Σ_{d∈D_k} P(d\|Q)·log P(d\|Q), P(d\|Q)=softmax(score(Q,d)) | Zendel, Liu, Culpepper & Scholer, "Entropy-Based QPP for Neural IR," QPP++ Workshop @ ECIR 2023 |
| Query Feedback (QF) | Stability of results when the query is expanded from its own top docs and re-run. | QF = overlap@k(Top_k(Q), Top_k(Q′)) / k, Q′ = pseudo-relevance-expanded query | Zhou & Croft, SIGIR 2007 |
| Ranking Robustness | How little the top-k ranking changes when document representations are perturbed with noise. | Correlation between original top-k ranking and ranking after perturbing document representations with noise | Zhou & Croft, "Ranking Robustness: A Novel Framework to Predict Query Performance," CIKM 2006 |
| Utility Estimation Framework (UEF) | Meta-predictor weighting a base signal by how consistent the ranking is with its relevance-model re-ranking. | UEF(Q,φ) = ξ(RM(Q), RM(θ_Q))·φ(Q), ξ = Pearson correlation between ranking and its RM-feedback re-ranking, φ = base predictor (e.g., NQC/WIG/Clarity) | Shtok, Kurland & Carmel, "Using Statistical Decision Theory and Relevance Models for QPP," SIGIR 2010 |
| Score autocorrelation (cluster-hypothesis measure) | Whether topically similar documents receive similar scores (local score consistency). | Spatial-autocorrelation (Moran's I–like) of scores over a document similarity graph; local consistency of scores among topically related docs | Diaz, "Performance Prediction Using Spatial Autocorrelation," SIGIR 2007; "Autocorrelation and Regularization of Query-Based IR Scores" (PhD thesis) |
| Reference-list based estimation | Calibrates the query's score distribution against reference queries of known effectiveness. | Weighted combination of a query's score distribution against a set of reference queries with known effectiveness | Roitman, Erera, Sar-Shalom & Weiner, "Enhanced Mean Retrieval Score Estimation for QPP," ICTIR 2017; Zendel, Shtok, Raiber, Kurland & Culpepper, SIGIR 2019 |
| Robust standard deviation estimation | Outlier-resistant version of σ_k, less distorted by freak scores. | Trimmed/robust variant of σ_k, less sensitive to score outliers | Roitman, Erera & Weiner, "Robust Standard Deviation Estimation for QPP," ICTIR 2017 |

## Neural / dense-retrieval–specific features

| Feature | Explanation | Formula | Reference |
|---|---|---|---|
| BERT-QPP score | Learned model reading query + top doc to directly predict effectiveness (AP/nDCG). | φ(Q) = Linear(BERT([CLS] Q [SEP] d₁ [SEP])), trained to regress AP/nDCG of top doc d₁ | Arabzadeh, Khodabakhsh & Bagheri, "BERT-QPP: Contextualized Pre-trained Transformers for QPP," CIKM 2021 |
| NeuralQPP (learned combination) | Neural combiner that fuses many base QPP signals into one predictor. | φ(Q) = f_NN(φ₁(Q),...,φ_m(Q)), weak-supervision-trained combiner over base QPP signals (WIG, NQC, Clarity, etc.) | Zamani, Croft & Culpepper, "Neural QPP Using Weak Supervision from Multiple Signals," SIGIR 2018 |
| Relative Information Gain (RIG) | Compares the query's retrieval-score gain against automatically generated query variants. | Compares retrieval-score gain of Q vs. automatically generated query variants | Datta, Ganguly, Mitra & Greene, "A Relative Information Gain-Based QPP Framework With Generated Query Variants," ACM TOIS 2022 |
| Coherence-based dense predictor (A-Pair-Ratio) | Embedding similarity among top docs vs bottom docs; a tight top cluster signals a confident dense match. | Ratio of avg. pairwise cosine similarity among top-ranked doc embeddings to that among bottom-ranked doc embeddings | Vlachou & Macdonald, "On Coherence-based Predictors for Dense Query Performance Prediction," arXiv:2310.11405 / SIGIR-ICTIR 2024 |
| Embedding-based query specificity (dense) | How close the query embedding sits to its nearest document embeddings in vector space. | Avg./max cosine distance of query embedding to nearest document embeddings ("neural query space proximity") | "Query Performance Prediction Using Neural Query Space Proximity," ACM TIST 2024 |

## Hybrid lexical/semantic routing–specific features

| Feature | Explanation | Formula | Reference |
|---|---|---|---|
| Sparse–dense score margin (top-1 effectiveness gap) | Quality gap between the top-1 sparse and top-1 dense results, used to set α directly. | Effectiveness scores assigned to top-1 sparse and top-1 dense results (via LLM judge), α set from normalized gap: final = α·BM25 + (1−α)·dense | Hsu & Tzeng, "DAT: Dynamic Alpha Tuning for Hybrid Retrieval in RAG," arXiv:2503.23013, 2025 |
| Sparse-retriever rank of first relevant doc (routing label/feature) | Rank of the first relevant doc in the sparse list; a high rank means route lexical, otherwise dense. | F_q = rank of first relevant passage in top-K sparse list; route to sparse if F_q ≤ threshold T, else dense | Arabzadeh, Yan & Clarke, "Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection," CIKM 2021 |
| Query-only cross-encoder routing score | Classifier over the query representation alone predicting sparse vs dense vs hybrid (no document needed). | Binary classifier over contextualized query representation (no document needed) predicting sparse vs. dense vs. hybrid | Arabzadeh, Yan & Clarke, CIKM 2021 |
| Rank-list agreement between retrievers | Agreement between the sparse and dense top-k lists; strong disagreement is where fusion has the most to gain. | Rank correlation (Kendall's τ) or top-k overlap (Jaccard@k) between sparse and dense result lists | General hybrid-fusion diagnostic, e.g., Kuzi, Zhang, Lin, Metzler & Nogueira, "Leveraging Semantic and Lexical Matching to Improve the Recall of Document Retrieval Systems," 2020 |

---

**Notes on use:** Several of these features are highly correlated — avgIDF, SCS, AvICTF, and γ1 all capture "specificity/informativeness," while NQC, WIG, SMV, and entropy all capture "score dispersion." An ablation study will likely collapse each family down to one or two representatives. The post-retrieval group requires a chosen retrieval model and top-k results already computed, which fits naturally with a lexical/semantic router since both candidate lists are being scored anyway — this also allows computing each post-retrieval feature separately per ranker (e.g., NQC_lexical vs. NQC_semantic) as router inputs.
