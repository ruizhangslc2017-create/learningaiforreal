"""A single attention head, written with Python's standard library.

Run: python3 implementation.py
Rows are vectors. The demo uses Q = K = V = X (identity projections).
The demo compares unmasked attention with causal attention.
"""

from math import exp, isclose, isfinite, sqrt


def softmax(scores):
    """Normalize a row; -inf marks masked positions with zero weight."""
    if not scores or any(not isfinite(s) and s != float("-inf") for s in scores):
        raise ValueError("Scores must be finite numbers or negative infinity.")
    largest = max(scores)
    if largest == float("-inf"):
        raise ValueError("A row needs at least one unmasked score.")
    exponentials = [exp(score - largest) for score in scores]
    total = sum(exponentials)
    return [number / total for number in exponentials]


def attention(queries, keys, values, *, causal=False):
    """Compute softmax(Q K^T / sqrt(d_k)) V.

    Return (scores, weights, outputs) so each step can be inspected.
    Each key has one corresponding value. Value width can differ from d_k.
    Causal mode assumes an aligned sequence: query i may use keys j <= i.
    """
    if not queries or not keys or not values:
        raise ValueError("Queries, keys, and values must be nonempty.")
    if len(keys) != len(values):
        raise ValueError("Each key needs one corresponding value.")
    if causal and len(queries) != len(keys):
        raise ValueError("Causal mode needs aligned queries and keys of equal length.")
    d_k = len(keys[0])
    d_v = len(values[0])
    if d_k == 0 or any(len(row) != d_k for row in queries + keys):
        raise ValueError("Queries and keys must share a positive width.")
    if d_v == 0 or any(len(row) != d_v for row in values):
        raise ValueError("Values must share a positive width.")

    scores = []
    weights = []
    outputs = []
    for i, query in enumerate(queries):
        # One score for each key: this is one row of Q @ K.T / sqrt(d_k).
        row_scores = [
            sum(q * k for q, k in zip(query, key)) / sqrt(d_k)
            for key in keys
        ]
        if causal:
            # Remove future positions BEFORE normalization. exp(-inf) = 0.
            row_scores = [score if j <= i else float("-inf")
                          for j, score in enumerate(row_scores)]
        row_weights = softmax(row_scores)
        # Combine values separately for each output coordinate.
        output = [
            sum(weight * value[d] for weight, value in zip(row_weights, values))
            for d in range(d_v)
        ]
        scores.append(row_scores)
        weights.append(row_weights)
        outputs.append(output)
    return scores, weights, outputs


def project(vectors, matrix):
    """Multiply X @ W; W has shape (input width, output width).

    These weights are provided by the caller. This function does not train them.
    """
    if not vectors or not matrix or not matrix[0]:
        raise ValueError("Vectors and projection matrix must be nonempty.")
    d_in, d_out = len(matrix), len(matrix[0])
    if any(len(row) != d_in for row in vectors):
        raise ValueError("Vector width must equal the projection's row count.")
    if any(len(row) != d_out for row in matrix):
        raise ValueError("Projection matrix must be rectangular.")
    return [[sum(vector[k] * matrix[k][d] for k in range(d_in))
             for d in range(d_out)] for vector in vectors]


def print_matrix(label, matrix):
    print(label)
    for row in matrix:
        print("  [" + ", ".join(f"{number:.6f}" for number in row) + "]")


def main():
    tokens = ["I", "like", "tea"]
    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    scores, weights, outputs = attention(x, x, x)

    print("Token order:", ", ".join(tokens))
    print_matrix("Input vectors:", x)
    print_matrix("Scaled scores:", scores)
    print_matrix("Attention weights (rows sum to 1):", weights)
    print_matrix("Output vectors:", outputs)
    print("Weight row sums:", [round(sum(row), 6) for row in weights])

    assert all(isclose(sum(row), 1.0, abs_tol=1e-12) for row in weights)
    # Independent calculation for the first token: scores are [a, 0, a].
    numerator = exp(1.0 / sqrt(2.0))
    denominator = 2.0 * numerator + 1.0
    expected = [2.0 * numerator / denominator, (1.0 + numerator) / denominator]
    assert all(isclose(actual, wanted) for actual, wanted in zip(outputs[0], expected))

    masked_scores, masked_weights, masked_outputs = attention(x, x, x, causal=True)
    print_matrix("Causal scores (-inf blocks future tokens):", masked_scores)
    print_matrix("Causal weights:", masked_weights)
    print_matrix("Causal outputs:", masked_outputs)
    assert masked_weights[0] == [1.0, 0.0, 0.0]
    assert masked_outputs[0] == x[0]
    assert all(isclose(sum(row), 1.0, abs_tol=1e-12) for row in masked_weights)
    assert all(masked_weights[i][j] == 0.0
               for i in range(len(x)) for j in range(i + 1, len(x)))

    # Change the third token in Q, K, and V, then recompute both modes.
    changed_x = [x[0][:], x[1][:], [10.0, 10.0]]
    _, _, changed_causal = attention(changed_x, changed_x, changed_x, causal=True)
    _, _, changed_unmasked = attention(changed_x, changed_x, changed_x)
    assert changed_causal[:2] == masked_outputs[:2]
    assert changed_unmasked[0] != outputs[0]
    print("Checks passed: normalization, worked output, masking, and future-token isolation.")


if __name__ == "__main__":
    main()
