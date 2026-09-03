## Single Agent World Model Design

### Architecture
- causal Diffusion Forcing Transformer
    - causal mask, except observation tokens for the same observation can attend to each other
    - multimodal rope (time, height, width) positional embeddings for all tokens
    - additive diffusion tau embedding for observation tokens
- input sequence of (observation tokens, action_token, observation tokens) and so on
- action tokens are just conditioning
- observation tokens learned with diffusion forcing objective
    - recommend trivial patch tokenizer, 3x3 patches through a learned projection up into the hidden space.
    - diffusion head outputs predicted noise from the last hidden over the patch tokens

### Decisions
- **No pose conditioning.** Absolute pose would break the translation and
  rotation equivariance the head-frame observation is built on, duplicate the
  heading the background gradient already encodes, and go stale the moment
  dead reckoning outlives the agent. Leaving it out also forces the model to
  piece together views taken from different angles, which is the interesting
  part.
- **Death is a single black frame** appended after the last living frame.
  Terminated episodes are not trainable past that point anyway, so the marker
  gives the model something to predict rather than a silent truncation. Only
  appended when the agent actually died; an episode cut by the step limit just
  ends.
- **Single agent is a baseline, not the target.** Another snake entering view
  is exogenous and unpredictable from one agent's partial view, so grading
  reports map consistency and self consistency separately from other-agent
  error rather than as one number.
