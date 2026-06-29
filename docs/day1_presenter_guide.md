# Day 1 — Presenter prep guide

Skim this the night before. It is *not* the slide script (that's
`day1_neural_networks_slides.md`). This is for fielding questions and
pre-empting confusion. Format per block: **teaching goals**, **likely
questions** (with crisp answers), **common confusions** (with the fix).

A mixed room will ask "what is X" more than "prove X." Keep answers intuitive;
offer the math only if they push.

---

## Block 1 — From neuron to perceptron

**Teaching goals**
- A neuron = weighted sum + bias + threshold. The dot product and bias are the
  whole story.
- A single perceptron draws one straight decision boundary.
- AND/OR are easy; XOR is not — and that failure motivates depth.

**Likely questions**
- **"Is this how the brain actually works?"** No — it's a loose analogy. Real
  neurons are far more complex (timing, chemistry, spiking). We borrow only the
  "sum inputs, fire past a threshold" shape.
- **"AI vs ML vs DL — what's the difference?"** AI is the broad goal; ML is
  learning from data; deep learning is the slice of ML that uses deep
  (many-layered) neural nets. Not every neural net is "deep."
- **"Why is the decision boundary a straight line?"** Because the score
  `w·x + b` is linear in the inputs; the set where it equals 0 is a line (or
  hyperplane). Curving it requires hidden layers + non-linearity (this
  afternoon).
- **"What does the bias do?"** It shifts the threshold. AND needs a high bar
  (b = −1.5, both inputs required); OR needs a low bar (b = −0.5, one is enough).
  Same weights, different bias.
- **"What's the dot product doing?"** Measuring alignment: how much the input
  points in the direction the node cares about (its weight vector).

**Common confusions**
- **Weights vs inputs.** Inputs `x` are the data; weights `w` are what the model
  learns. Point at each on the diagram.
- **"It failed on XOR, so neural nets don't work."** No — *one* node fails. The
  whole point of the afternoon is that two nodes fix it.
- **Linearly separable.** Means "a single straight line can split the classes."
  XOR's two true corners sit diagonally, so no line works. Draw it if asked.

---

## Block 2 — Make it deep, make it learn

**Teaching goals**
- Hidden layers build features; XOR = combine OR and AND, then separate.
- Activations must be non-linear (or depth collapses) and differentiable.
- Training = define a loss, then descend its gradient; backprop is just the
  chain rule.

**Likely questions**
- **"Why do we even need activation functions?"** Without them, stacking layers
  collapses to one linear layer: `W₂(W₁x) = (W₂W₁)x`. The non-linearity is what
  lets depth bend the boundary.
- **"Sigmoid vs ReLU — why did everyone switch to ReLU?"** Sigmoid/tanh
  saturate: their slopes flatten to ~0 at the tails, and backprop multiplies
  slopes across layers, so gradients vanish and early layers stop learning.
  ReLU's slope is exactly 1 when active, so the signal survives depth.
- **"Is ReLU even differentiable? It has a corner."** Everywhere except the
  single point z = 0. In practice we just declare the derivative there to be 0
  (sometimes 1); it's a measure-zero case that never matters.
- **"Softmax vs sigmoid?"** Sigmoid squashes one number to (0,1); softmax turns a
  *vector* of scores into probabilities that sum to 1 across classes.
- **"Why log in cross-entropy?"** It's the negative log-probability of the
  correct class: confident-and-wrong is punished heavily, confident-and-right is
  cheap. Minimizing it = maximizing the probability you assign to the truth.
- **"What exactly is a gradient?"** The vector of slopes of the loss w.r.t. every
  weight — it points uphill, so we step the opposite way.
- **"Do we compute backprop by hand every time?"** No. We do it once by hand to
  understand it (NB2); after that the framework's autodiff does it.

**Common confusions**
- **Logits vs probabilities.** Raw scores (logits) → softmax → probabilities.
  People mix these up; name them explicitly.
- **Learning rate.** Too big overshoots/diverges; too small crawls. It's a
  *hyperparameter* (you set it), not learned.
- **Loss vs accuracy.** Loss is what we optimize (smooth, differentiable);
  accuracy is what we report (not directly differentiable).
- **"Backprop is a different algorithm from gradient descent."** No — backprop
  *computes* the gradient; gradient descent *uses* it to update weights.

---

## Block 3 — Do it for real: PyTorch

**Teaching goals**
- A framework gives autodiff, GPU speed, composable layers, and pretrained
  models to build on.
- The five-line training loop is the pattern reused all week.

**Likely questions**
- **"Why not just use our NumPy version?"** Three reasons: autodiff (no
  hand-written backward), speed (optimized GPU kernels), and ecosystem (stack
  layers freely, load pretrained networks). Our version was for understanding,
  not production.
- **"Tensor vs NumPy array?"** Almost the same API, but a tensor can live on a
  GPU and can track gradients (`requires_grad`).
- **"What does `requires_grad` do?"** Tells PyTorch to record operations on that
  tensor so it can compute gradients later via `.backward()`.
- **"Why `optimizer.zero_grad()` every step?"** Gradients *accumulate* by
  default; if you don't clear them, this step's gradient adds to the last one's.
  Most common beginner bug.
- **"Adam vs SGD?"** Adam adapts the step size per-parameter and usually trains
  faster out of the box; SGD is simpler. We default to Adam.
- **"What's Fashion-MNIST?"** 28×28 grayscale clothing images in 10 classes — a
  drop-in, slightly harder replacement for digit-MNIST.
- **"CPU or GPU?"** Works on CPU for this small model; GPU just makes it faster.

**Common confusions**
- **Parameters vs hyperparameters.** Parameters (weights/biases) are learned;
  hyperparameters (learning rate, batch size, #layers) are chosen by you.
- **Epoch vs batch vs step.** A *batch* is a handful of examples; a *step* is one
  update on one batch; an *epoch* is one full pass over the data.
- **`nn.Module` magic.** Defining layers in `__init__` auto-registers their
  parameters; `forward()` just describes the data flow. They don't call
  `backward` themselves — `loss.backward()` does it.
- **Overfitting.** Train loss down + val loss up = memorizing, not learning.
  That's what dropout / more data / regularization fight.

---

## Quick reference (have these at your fingertips)

**Gate weights (the demos)**
- AND: w₁ = 1, w₂ = 1, b = −1.5 (fires only when both = 1).
- OR:  w₁ = 1, w₂ = 1, b = −0.5 (fires when at least one = 1).
- XOR (MLP): h₁ = OR, h₂ = AND, output y = step(h₁ − h₂ − 0.5).
- XOR remap: (0,0)→(0,0) F · (0,1),(1,0)→(1,0) T · (1,1)→(1,1) F → one line separates.

**Key equations**
- Node: z = w·x + b, then y = f(z).
- Softmax: pᵢ = e^{zᵢ} / Σⱼ e^{zⱼ}.
- Cross-entropy: L = −log p_correct.
- MSE: L = (1/n) Σ (yᵢ − ŷᵢ)².
- Update: θ ← θ − η ∇θ L.
- Backprop tagline: "gradient descent + the chain rule."

**Numbers worth knowing**
- Sigmoid's max slope is 0.25 → why deep sigmoid nets vanish gradients.
- ReLU slope = 1 when active, 0 otherwise.

**History (for the "why now" slide)** — all **[VERIFY]** before quoting:
- Perceptron: Rosenblatt, 1958. Gradient descent: Cauchy, 1847. Backprop
  popularized: Rumelhart, Hinton & Williams, 1986 (roots in the 1970s).
- ImageNet/AlexNet inflection: 2012, ~15% top-5 error vs ~26% runner-up.

**Throughline to keep saying**
- Today's training loop returns on Day 3 (CNNs) and Day 4 (fine-tuning).
- Cross-entropy + "classification over a vocabulary" is how language modeling
  works → sets up NLP and transformers.

---

## Pacing reminders
- Block 2 is the time crunch — if behind, make NB2 a read-along and trim the
  optimizer/overfitting asides.
- Protect the XOR cliffhanger: end the morning on the *failure*, open the
  afternoon by fixing it.
- Don't over-explain the [VERIFY] items live; if unsure of a date/figure, say
  "roughly" rather than commit to a wrong number.

---

## Deep-dive explanations (your questions)

### 1. The decision boundary (perceptron) — how to say it accurately

A perceptron computes a score `z = w·x + b` and predicts the class from its
sign. The **decision boundary** is the set of points where it's exactly
undecided, `z = 0`, i.e. `w·x + b = 0`. In 2D with `x = (x₁, x₂)`:

```
w₁x₁ + w₂x₂ + b = 0   ->   x₂ = -(w₁/w₂) x₁ - b/w₂
```

That's the equation of a straight line. Four accurate things to say:

- **It's always a hyperplane (a line in 2D, a flat plane in higher D)** because
  `z` is *linear* in the inputs. A single perceptron can only ever draw a
  straight boundary — this is exactly why XOR defeats it.
- **`w` is the normal vector** — it points perpendicular to the boundary, toward
  the positive-class side. So changing `w` *rotates* the boundary.
- **`b` only shifts it.** The line's offset from the origin is `-b/‖w‖`; changing
  `b` *translates* the boundary without rotating it. (This is exactly the AND vs
  OR demo: same `w`, the bias slides the line.)
- **The magnitude `|z|` is the margin.** `|z| = ‖w‖ × (distance from the point to
  the boundary)`. So points far on the correct side have large `z` (confident);
  points near the line have small `z` (unsure). Learning = move the line so every
  point sits on its correct side.

If someone asks "why a line and not a curve?" — because there's no non-linearity
yet. Curving the boundary is precisely what the hidden layer buys us next.

### 2. How the hidden layer "shifts the points" (XOR) — accurately

The honest framing: **the hidden layer doesn't bend one line in the original
space — it re-represents the data in a new space where a straight line works
again.** Steps to say out loud:

1. In the original `(x₁, x₂)` space, XOR's classes are **not linearly separable** —
   the two "true" corners sit diagonally, so no single line splits them.
2. The hidden layer applies a non-linear map. In our build, `h₁` acts like OR and
   `h₂` like AND, so each input point gets new coordinates `(h₁, h₂)` — a learned
   **feature space / representation**.
3. In that new space the points are rearranged:
   ```
   (0,0) -> (0,0)  False
   (0,1) -> (1,0)  True
   (1,0) -> (1,0)  True
   (1,1) -> (1,1)  False
   ```
   The two "true" inputs collapse onto the same point `(1,0)`, which is now
   cleanly separable from `(0,0)` and `(1,1)` by the single line `h₁ − h₂ = 0.5`.
4. The **output node is still just a perceptron** — a linear boundary — but now
   it operates on the transformed features, so one line suffices.

Two precise caveats worth keeping in your back pocket:
- Equivalently, viewed *back* in the original `(x₁, x₂)` space, the composition is
  a non-linear boundary. With step/ReLU nodes it's **piecewise-linear** (the
  decision region is a union of half-planes / a polygon), not a smooth curve.
- This is the one-sentence thesis of deep learning: **early layers learn a
  representation in which the final linear layer's job becomes easy.** Say that —
  it's the throughline for Days 3–4 too.

### 3. How to explain backprop (nested function -> chain rule -> δ caching)

Use the tiny scalar network `x -> a₁ -> a₂` with one hidden node. Forward
equations: `z₁ = w₁x + b₁`, `a₁ = σ(z₁)`, `z₂ = w₂a₁ + b₂`, `a₂ = σ(z₂)`,
`L = ½(a₂ − y)²`.

**(a) The network is one nested function.** Substitute each equation into the one
above it until the whole net is a single expression:

```
L = ½(a₂ − y)²
  = ½( σ(z₂) − y )²
  = ½( σ(w₂a₁ + b₂) − y )²
  = ½( σ(w₂·σ(z₁) + b₂) − y )²
  = ½( σ(w₂·σ(w₁x + b₁) + b₂) − y )²
```

Reading the parentheses inside-out *is* the forward pass. Each layer is a shell
wrapped around the previous one.

**(b) Differentiating the nest = chain rule peeling shells.** `w₁` is in the
innermost shell, so reaching it multiplies one local derivative per shell:

```
dL/du   = (a₂ − y)        (u = σ(z₂) − y)
du/dz₂  = σ'(z₂)
dz₂/da₁ = w₂
da₁/dz₁ = σ'(z₁)
dz₁/dw₁ = x

∂L/∂w₁ = (a₂ − y)·σ'(z₂)·w₂·σ'(z₁)·x
          └ output shells ┘  link  hidden  input
```

That product is just calculus on the big formula — no "backprop" yet.

**(c) Backprop = don't recompute the shared shells.** Put the two gradients side
by side:

```
∂L/∂w₂ = (a₂ − y)·σ'(z₂) · a₁
         └──── δ₂ ────┘
∂L/∂w₁ = (a₂ − y)·σ'(z₂) · w₂·σ'(z₁) · x
         └──── δ₂ ────┘
```

The whole `(a₂−y)·σ'(z₂)` bundle appears in *both*. So compute it once, call it
`δ₂`, and pass it backward:

```
δ₂ = (a₂ − y)·σ'(z₂)          (peel the outer shells once)
δ₁ = δ₂ · w₂ · σ'(z₁)         (reuse δ₂, peel one more shell)
```

Each `δ` is "all the outer shells already peeled," ready for the next layer to
extend by one local factor. That caching is the entire trick.

**(d) The worked numbers** (use `x = 1, y = 0, w₁ = w₂ = 0.5, b₁ = b₂ = 0`):

```
Forward:  z₁ = 0.5,  a₁ = σ(0.5) = 0.6225
          z₂ = 0.5·0.6225 = 0.3113,  a₂ = σ(0.3113) = 0.5772
          L  = ½(0.5772)² = 0.1666

Backward: a₂ − y = 0.5772
          σ'(z₂) = a₂(1−a₂) = 0.5772·0.4228 = 0.2440
          δ₂ = 0.5772·0.2440 = 0.1409
          ∂L/∂w₂ = δ₂·a₁ = 0.1409·0.6225 = 0.0877
          σ'(z₁) = a₁(1−a₁) = 0.6225·0.3775 = 0.2350
          δ₁ = δ₂·w₂·σ'(z₁) = 0.1409·0.5·0.2350 = 0.01655
          ∂L/∂w₁ = δ₁·x = 0.01655
```

Landing line for the room: "We peeled the outer shells **once** (δ₂), reused them
for `w₂`, then extended by one shell to get `w₁`. That reuse, across a whole
network, is what makes training tractable."
