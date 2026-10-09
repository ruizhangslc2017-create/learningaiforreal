# Learning AI for real

Learn a concept, build it in a small example, change one thing, and explain
what happened. Keep notes and code in the same topic folder.

## Start here

The active topic is [attention](02-transformers/attention/README.md).
It includes a standard-library implementation, worked calculations,
causal masking, and a notebook of experiments.

```bash
python3 02-transformers/attention/implementation.py
```

Run that command from the AI root. The program prints intermediate matrices
and checks that changing a future token cannot affect earlier causal outputs.

## Subjects and useful milestones

| Folder | First learning milestone |
| --- | --- |
| `01-foundations/backpropagation/` | Build scalar autodiff and verify gradients with finite differences. |
| `01-foundations/optimization/` | Compare learning rates on the same small loss function. |
| `02-transformers/attention/` | Reconstruct scores, weights, outputs, and causal masking. |
| `02-transformers/tokenization/` | Implement a small byte-pair tokenizer and inspect its splits. |
| `02-transformers/language-modeling/` | Train next-token prediction with shifted targets. |
| `03-memory-and-agents/` | Retrieve relevant notes, then inspect a simple tool-use loop. |
| `04-multimodal/` | Compare text and image embeddings in a shared representation. |
| `projects/` | Combine working concepts into a small trained language model. |

The other subject folders are places for future work. Add their files when
you begin an actual example.

## A topic's files

| File | What belongs in it |
| --- | --- |
| `README.md` | Goal, sources, current state, next action. |
| `notes.md` | Your explanation, equations, worked examples, questions. |
| `implementation.py` | Small code you can reconstruct and explain. |
| `experiments.ipynb` | Predictions, changed variables, actual outputs, mistakes, follow-up. |
| `paper-notes.md` | Method, evidence, limitations, questions, source links. |

## Session routine

1. Predict one result before running the code.
2. Make one controlled change and inspect the result.
3. Explain the result in your own words, including anything surprising.
4. Append three lines to [progress.md](progress.md): `Learned:`, `Built:`, `Next:`.

Treat generated explanations as starting material: rewrite a paragraph from
memory, then check it against the calculation. Understanding a topic means
you can rebuild its core example and explain a failed experiment.
