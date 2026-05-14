# Temperature Sweep Notes

Prompt: "Explain unlearning in one sentence."

## Temperature = 0.2
The output was more conservative and definition-like. It stayed close to a general explanation and produced a structured continuation.

## Temperature = 0.7
The output became slightly more interpretive and less rigid, adding language about perspectives and relationships.

## Temperature = 1.2
The output became more variable and began shifting the interaction by asking a follow-up question. This suggests higher temperature can make generation branch into less predictable trajectories.

## Key Concept
Temperature does not make the model "think harder." It rescales logits before sampling. Lower temperature sharpens the probability distribution, making high-probability tokens dominate. Higher temperature flattens the distribution, allowing lower-probability tokens to be sampled more often.

## Main Observation
The runtime stayed almost constant across temperatures, but the behavioral trajectory changed. This makes sense because sampling temperature changes token selection, not the core model computation.
