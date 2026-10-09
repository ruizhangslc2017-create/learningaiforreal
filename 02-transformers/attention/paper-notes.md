# Attention Is All You Need — focused reading

[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).
Scope: sections 3.2, 4, and the translation results in section 6.1.

## Method

Queries and keys produce scaled dot products. Softmax converts them to value
weights. Multiple heads use separate learned projections; their outputs are
concatenated and projected. Decoder self-attention masks future positions.
[Sections 3.2.1–3.2.3](https://arxiv.org/html/1706.03762v7#S3.SS2).

## Evidence

Table 2 reports 28.4 English–German BLEU for Transformer (big), versus 27.3
for Transformer (base), on WMT 2014. These evaluate full trained translation
systems, including components beyond attention.
[Section 6.1, Table 2](https://arxiv.org/html/1706.03762v7#S6.SS1).

## Limitations and interpretation

Dense self-attention has quadratic sequence-length cost in the paper's
complexity comparison. Translation benchmarks provide evidence in those
settings; they do not establish performance for every task or isolate the
contribution of a single component. These latter points are my interpretation.
[Section 4](https://arxiv.org/html/1706.03762v7#S4).

## Questions for the next read

- What does Table 3 tell us about changing the number of heads?
- How would I design an experiment separating attention from the feed-forward block?
- Which part of the architecture gives token positions distinct information?

## Connection to my implementation

Trace the formula through `row_scores`, `row_weights`, and `output`.
Explain how `project()` changes the inputs to that calculation.
