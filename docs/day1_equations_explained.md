# Day 1 — Every on-slide equation, explained

A rehearsal companion: for each equation on the slides, what it says, how to read
it out loud, and the one line to land. Symbols: **x** = input, **w** = weights,
**b** = bias, **z** = pre-activation (weighted sum), **a** = activation (output of
a node), **σ / f** = sigmoid, **η** = learning rate, **L** = loss, **ŷ** =
prediction, **δ** = a node's "error" term.

---

## Perceptron

**z = w·x + b = Σᵢ wᵢ xᵢ + b**
Read: "z equals w dot x plus b — multiply each input by its weight, add them up,
then add the bias." The dot product `w·x` is a similarity/alignment score between
the input and the weights. Say: "A neuron is just a weighted vote plus a bias."

**y = f(z), with f = Sign (or step)**
Read: "the output is a function of z; for a perceptron that function is the sign."
Say: "If the weighted vote clears zero, fire a 1; otherwise 0."

**h(x) = Sign(b + w₁x₁ + w₂x₂)**
The same neuron written out for two inputs. `b` is the bias (you'll also see it as
`w₀`). Say: "Two weights and a bias decide a yes/no."

**w₀ + w₁x₁ + ⋯ + w_k x_k = 0  (the decision boundary)**
Read: "the boundary is where the weighted sum equals zero." Everything on one side
is predicted True, the other side False. Say: "Setting the score to zero traces a
straight line (a hyperplane) — that line is the classifier."

**AND: w₁=0.5, w₂=0.5, b=−0.8  ·  OR: b≈−0.3  ·  threshold T = −b**
Say: "Same machine; the bias sets how many inputs must be ON. AND needs both (very
negative bias); OR needs just one (less negative). The threshold is minus the
bias."

---

## Why a perceptron fails on XOR
XOR's true cases sit on opposite corners, so no single straight line separates
them. Say: "A perceptron can only draw one straight line — XOR needs more."

---

## Activation functions

**σ(z) = 1 / (1 + e^(−z))** → squashes any number into (0, 1).
**tanh(z) = (eᶻ − e^(−z)) / (eᶻ + e^(−z))** → squashes into (−1, 1), zero-centered.
**ReLU(z) = max(0, z)** → zero for negatives, identity for positives.
Say: "Three ways to bend a number; ReLU is the modern default because it doesn't
saturate."

**W₂(W₁x) = (W₂W₁)x   vs.   W₂·σ(W₁x)**
Read: "two linear layers collapse into one linear layer — still just a line. Put a
sigmoid in between and it stops being linear." Say: "The non-linearity is what lets
depth bend the boundary; without it, layers add nothing."

---

## Feed-forward (matrix form)

**h = X·W_xh** then **O = h·W_ho** then **σ(O)**
Read: "multiply the input row vector by the first weight matrix to get the hidden
vector; multiply that by the second weight matrix to get the output; squash."
Say: "A forward pass is just two matrix multiplies and a squash."

**With bias: h = X·W_xh + h_bias, O = h·W_ho + O_bias**
Say: "Each layer also adds a constant — that's what the extra '1' node does."

---

## Loss functions

**softmax: pᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)**
Read: "exponentiate each score and normalize so they sum to one." Say: "Turns raw
scores into probabilities."

**cross-entropy: L = −Σᵢ yᵢ log pᵢ = −log p_correct**
Read: "minus the log probability the model gave the correct class." Say: "Confident
and wrong is punished hard; confident and right is nearly free."

**MSE = (1/n) Σᵢ (yᵢ − ŷᵢ)²**
Read: "average squared gap between prediction and truth." Say: "The regression
loss; cross-entropy is its classification cousin."

---

## Gradient descent

**θ ← θ − η ∇_θ L**
Read: "update the parameters by stepping against the gradient, scaled by the
learning rate." `∇_θ L` points uphill (toward larger loss); the minus sign walks
downhill. Say: "Nudge every weight a little in the direction that lowers the loss."

---

## Two-layer backprop (the general form)

**Objective:  min_{W,v}  Σₙ ½ ( yₙ − Σᵢ vᵢ f(wᵢ·xₙ) )²**
Read: "minimize, over all weights, the squared error summed over observations." The
inner sum `Σᵢ vᵢ f(wᵢ·x)` is the prediction ŷ: each hidden unit i computes
`f(wᵢ·x)`, then the output weights `vᵢ` combine them. Say: "Find the weights that
make the prediction match y."

**Loss:  L(W) = ½ ( y − Σᵢ vᵢ f(wᵢ·x) )²**  — same thing for one example.

**Chain rule:  ∂L/∂wᵢ = (∂L/∂fᵢ)(∂fᵢ/∂wᵢ)**
Read: "to change a first-layer weight, go through the hidden unit it feeds." Two
pieces:
- **∂L/∂fᵢ = −( y − ŷ ) vᵢ = −e·vᵢ** — how much the loss cares about hidden unit i:
  the error `e = y − ŷ` times that unit's output weight `vᵢ`.
- **∂fᵢ/∂wᵢ = f'(wᵢ·x)·x** — how the hidden unit responds to its weight: the
  slope of the sigmoid times the input.

**Result:  ∇_{wᵢ} = −e·vᵢ·f'(wᵢ·x)·x**
Say: "The gradient is the error, scaled by how much this unit matters (vᵢ), by how
responsive it is (f'), and by the input (x)."

**Sanity checks**
- Small error e → small step.
- Small vᵢ (unit barely used) → small gradient.
- Flip the sign of e or vᵢ → the update flips too.
Say: "The formula behaves exactly how intuition says it should."

---

## Concrete scalar example (x → a₁ → a₂ → L)

**Forward:** z₁ = w₁x + b₁ ; a₁ = σ(z₁) ; z₂ = w₂a₁ + b₂ ; a₂ = σ(z₂) ;
L = ½(a₂ − y)².
Read each as "weighted sum, then squash," twice, then "halve the squared error."

**Nested form:  L = ½( σ( w₂·σ(w₁x + b₁) + b₂ ) − y )²**
Say: "The whole network is one function — read the parentheses inside-out and you
are doing the forward pass."

**Backward (the four to remember):**
- **δ₂ = (a₂ − y)·σ'(z₂)** — the output node's error: how wrong it is, times how
  responsive it is.
- **δ₁ = δ₂·w₂·σ'(z₁)** — pass δ₂ back through the weight and the hidden node's
  slope.
- **∂L/∂w₂ = δ₂·a₁** — output-weight gradient = its delta times its input.
- **∂L/∂w₁ = δ₁·x** — first-weight gradient = its delta times its input.
Say: "Compute δ at a node, then every weight into it gets δ × (its input). δ's
flow backward; that reuse is what makes it cheap."

**The numbers (x=1, y=0, w₁=w₂=0.5, b₁=b₂=0):**
a₁ = σ(0.5) = 0.6225 ; a₂ = σ(0.3113) = 0.5772 ; L = 0.1666 ;
δ₂ = 0.5772·0.2440 = 0.1409 ; ∂L/∂w₂ = 0.1409·0.6225 = 0.0877 ;
δ₁ = 0.1409·0.5·0.2350 = 0.01655 ; ∂L/∂w₁ = 0.01655·1 = 0.01655.
(Note σ'(z) = a(1−a): σ'(z₂)=0.5772·0.4228=0.2440, σ'(z₁)=0.6225·0.3775=0.2350.)
Say: "Watch each red number on the right be one of the left-hand formulas, filled
in with the green forward numbers."

---

## Generalizing backprop

**Three steps:** 1) forward → outputs; 2) backward → deltas; 3) update → gradient
descent. Say: "Same loop at any depth."

**Output-layer delta:  δ_jK = Y_jK(1 − Y_jK)(Ŷ_jK − Y_jK)**
Read: "for an output node: sigmoid-slope `Y(1−Y)` times the residual `(Ŷ − Y)`."
Say: "How wrong this output node is, weighted by how responsive it is."

**Hidden delta:  δ₅ = w₅₆·δ**
Read: "a hidden node's delta is the downstream delta times the weight connecting
them." Say: "Blame flows back along the wires, scaled by the wire's weight."

**Cost (cross-entropy form):  Cost = Y·log Ŷ + (1−Y)·log(1−Ŷ)**
Say: "Another way to measure how close the output is to the truth."

**Weight update:  w += η·δ_jk·x_i**
Say: "Every weight moves by learning-rate × its node's delta × its input — that's
gradient descent again."

---

## XOR with an MLP
**y = step(h₁ − h₂ − 0.5)**, with h₁ = OR, h₂ = AND.
Say: "OR fires for any input, AND only for both; OR-minus-AND isolates the
'exactly one' cases — that's XOR."

---

## How to use this
- The right-hand "Say:" lines are your spoken version — lead with those.
- If a room wants the derivation, the chain-rule block and the four delta
  equations are the whole story; everything else is arithmetic.
- The single sentence that ties it together: **"Backprop = the chain rule applied
  to the loss, reusing shared pieces (the δ's) as it walks backward."**
