# Day 1 — Neural Networks: slide content + speaker notes (DRAFT v2 for review)

Each slide has two parts:
- **On slide** — the literal bullets / layout / notation the audience sees.
- **Say** — terse talking-point cues for you (not paragraphs).
- **Cite** — anchor tags; full list in **Sources** at the bottom.

Anchors: Berkeley **CS189 Spring 2025** (Shewchuk lecture notes) and Jurafsky &
Martin **SLP3, Jan 6 2026 release, Ch.6 Neural Networks**. **[VERIFY]** = check
before it goes on a slide. Notation philosophy: show the math, point at each
piece, never make them grind it. Terminology: we call a neuron a **node** (not
"unit").

**Animated scenes (to build in Manim later, in one batch):** decision boundary
(Slide 11), the logic-gate arc AND → OR → XOR (Slides 13–15), MLP-solves-XOR
(Slide 17), backprop picture (Slide 25). Natural lunch split: after Slide 15 (XOR
fails); Slide 17 (MLP solves) opens the afternoon.

---

## Timing pressure-test (does Day 1 fit ~5 hrs?)

| Time | Block | Content | Mins |
|---|---|---|---|
| 10:00–10:30 | — | Breakfast / settle | — |
| 10:30–12:30 | **Block 1** (S1–16) | Intro/about-me/agenda → DL framing → why now → UAT → perceptron → **AND/OR/XOR animation** → **NB1** | ~120 |
| 12:30–1:30 | — | Lunch (cliffhanger: XOR just broke the perceptron) | — |
| 1:30–4:00 | **Block 2** (S17–26) | **MLP-solves-XOR animation** → formalize → activations → loss → gradient descent → backprop (3-layer + manim) → **NB2** | ~150 |
| 4:00–4:15 | — | Break | — |
| 4:15–5:30 | **Block 3** (S27–32) | PyTorch → **NB3 (Fashion-MNIST)** → light mentions → wrap | ~75 |
| 5:30–6:00 | — | Wrap, Q&A, bridge to NLP | — |

≈5 hr of real content. **Block 2 is the crunch.** Safety valves: keep NB2 a
guided read-along; collapse optimizer + overfitting asides.

---

# BLOCK 1 — From neuron to perceptron (10:30–12:30)

## Slide 1 — Title
**On slide**
- "Neural Networks" (large, centered; charcoal bg, white text).

**Say**
- Land the title, set the tone. Move quickly to who you are.

**Cite** — none.

## Slide 2 — About me
**On slide**
- Your name + the lab/group you're part of (logo or name).
- The problems you work on (2–3 bullets).
- `# TODO image:` photo of you.

**Say**
- 2–3 min, human and concrete: what your lab studies, what you find exciting.
- Why you're qualified to walk them through this — lightly, no résumé.

**Cite** — your content.

## Slide 3 — Where this week is going
**On slide**
- Day 1 Neural Networks
- Day 2 NLP
- Day 3 Computer Vision
- Day 4 Transformers
- (No caption.)

**Say**
- Each day stands alone but compounds; today is the foundation.
- Plant the callback: attention (Day 4) fixes a problem we surface in NLP.

**Cite** — none.

## Slide 4 — Today's agenda
**On slide**
- Morning: from the neuron to the perceptron; the XOR problem.
- Afternoon: multilayer networks, training, and backprop.
- Late afternoon: PyTorch + hands-on notebooks.
- (Meals/breaks noted lightly. No caption.)

**Say**
- Set the rhythm: slides to motivate, notebooks to apply, back to slides to generalize.
- Tell them notebooks are follow-along; they'll write/run code with you.

**Cite** — course design.

## Slide 5 — What is deep learning? (AI ⊃ ML ⊃ DL)
**On slide**
- Nested circles: AI ⊃ Machine Learning ⊃ Deep Learning.
- ML: learn a function from data instead of hand-coding rules.
- DL: also learn the *features*, via layered (deep) networks.
- Caption (descriptive): "Deep learning is the part of machine learning that uses deep, many-layered neural networks. Not every neural network is 'deep' — a single perceptron is still a neural network and still machine learning — but nearly all of deep learning is built from neural networks."

**Say**
- Classic ML: human designs features, model fits weights.
- DL's move: learn the features too — that's the "deep" part.
- The nuance: neural network = the model family; "deep learning" = using deep ones. Don't overclaim the Venn.

**Cite** — [JM-6] (NNs as learned non-linear function approximators). **[VERIFY line]**

## Slide 6 — Where neural nets show up
**On slide**
- Two case panels (reserve right third for an image each — `# TODO image:`):
  - **AlphaFold** (DeepMind): predicts a protein's 3D structure from its amino-acid sequence.
  - **Poverty & aid mapping** (your colleague's work): neural nets to study poverty and target aid.

**Say**
- AlphaFold: a decades-old biology problem (structure from sequence); at CASP14 (2020) it reached near-experimental accuracy — point at the impact, not the architecture.
- Poverty/aid: your colleague's work; the broader line (satellite imagery + deep learning → poverty maps) is a well-known result to anchor it.
- Keep energy up; this is the "why care" beat.

**Cite** — [AlphaFold] Jumper et al. 2021 (CASP14, 2020); [Poverty] Jean et al. 2016 (Science) as anchor. Colleague specifics **[VERIFY]**.

## Slide 7 — Why *now*? (the ideas are old)
**On slide**
- Caption (descriptive): "The perceptron dates to 1958, gradient descent even earlier to 1847, and backprop was popularized in 1986 — the core ideas are decades old. So what changed?"
- Bullets:
  - Compute
  - Data
  - Software
- `# TODO image:` your ImageNet/inflection-point image.

**Say**
- The math isn't new — say the three dates out loud.
- What changed: **Compute** (GPUs make big matrix math cheap), **Data** (large labeled datasets), **Software** (autodiff frameworks like PyTorch).
- The inflection: for years neural nets were sidelined; other methods (e.g. SVMs) led. Then the **ImageNet competition (ILSVRC)** — a large image-recognition contest — was won decisively by a deep network, **AlexNet (2012)**: ~15% top-5 error vs ~26% for the next best. That convinced the field DL works at scale → the modern boom.

**Cite** — [CS189-L19] (Krizhevsky et al. 2012). Dates: perceptron 1958, GD 1847, backprop 1986 **[VERIFY figures]**.

## Slide 8 — One property worth stating: universal approximation
**On slide**
- Boxed: **"Any sufficiently wide two-layer neural network is enough to approximate any given function."**
- Cartoon: sum of ReLU "bumps" tracing a curve.
- Caveat: existence ≠ findable; says nothing about *how many* nodes.

**Say**
- Intuition: enough little pieces trace any shape; width = number of pieces.
- This is *why* nets are expressive.
- Be honest: promises a net exists, not that training finds it. Don't prove it.

**Cite** — [UAT] Cybenko (1989); Hornik et al. (1989). **[VERIFY wording]**; text Bishop (2024) [CS189-TEXT].

## Slide 9 — The biological neuron (the analogy)
**On slide**
- Sketch: dendrites (inputs) → soma (sums) → axon (fires if strong enough).
- Caption: "Perceptrons are analogies to neurons, not one-to-one models."

**Say**
- Real neuron: collects signals, fires past a threshold.
- We borrow the *shape* of the idea, nothing more — say this plainly.
- Bridge: "let's write that down as math."

**Cite** — [CS189-L18] (neuron biology); [JM-6.1] (the artificial node).

## Slide 10 — The perceptron: the math
**On slide**
- Inputs **x**, weights **w**, bias b.
- z = w·x + b = Σᵢ wᵢxᵢ + b
- y = f(z), perceptron uses f = step / sign.
- Highlight "w·x" (dot product) and "b" (threshold shift).

**Say**
- Left to right: multiply each input by its weight, sum, add bias, threshold.
- Dot product = "how aligned is the input with what this node cares about."
- Bias = where the threshold sits. That's the whole node.

**Cite** — [CS189-L2]; [JM-6.1].

## Slide 11 — Geometry: the decision boundary (ANIMATED)
**On slide**
- w·x + b = 0 is a line/hyperplane; w = normal; b = offset.
- Animation: sweep w and b → boundary rotates/translates, points switch sides.

**Say**
- Boundary = where the node is undecided (z = 0).
- w points perpendicular to the line; b slides it.
- "Learning = finding the right line." (Sets up the gate demos next.)

**Cite** — [CS189-L2][CS189-L3]. Manim: `next_slide(loop=True)`.

## Slide 12 — How a perceptron learns (intro to weights + error)
**On slide**
- "The weights **w** and bias b are the knobs we adjust."
- Learning loop (plain): make a prediction → measure the error → nudge the weights to reduce it → repeat.
- Perceptron rule: on a misclassified point, shift w toward fixing it.
- Tag: "a first taste of gradient descent — formalized this afternoon."

**Say**
- This is the gentle intro to two ideas we'll formalize later: **weights** (what we learn) and **error/loss** (what we reduce).
- Keep it conceptual now — the full loss function is Slide 21, gradient descent is Slide 22. Here just plant: "learning = adjust weights to lower error."
- The perceptron rule: each mistake pushes the boundary a little; repeat until it separates the data (when it can).

**Cite** — [CS189-L3] (perceptron learning / gradient descent); loss + GD formalized later [JM-6.6].

## Slide 13 — Perceptron computes AND (ANIMATED) → scene `LogicGates`
**On slide**
- Recall banner: "z = w·x + b, then step(z)" (callback to Slide 10).
- Perceptron diagram with weights **w₁ = 1, w₂ = 1, b = −1.5**.
- Truth table (inputs 0/1, output T/F): (0,0)→F, (0,1)→F, (1,0)→F, (1,1)→T.
- Per-row reveal: z = 1·x₁ + 1·x₂ − 1.5, then step(z), then ✓ against the table.

**Say**
- Recall the perceptron equation first (point at the banner).
- "A single perceptron already computes logic. Watch AND."
- Walk (1,1): z = 0.5 ≥ 0 → fires → T. Only both-on fires; the bias sets that threshold.
- Let the animation tick through the other rows.

**Cite** — [CS189-L2] (perceptron as linear classifier); AND/OR are classic gate constructions.

## Slide 14 — Perceptron computes OR (ANIMATED + ask the room) → scene `LogicGates`
**On slide**
- Same diagram, weights shown as **w₁ = ?, w₂ = ?, b = ?**.
- Truth table for OR: (0,0)→F, (0,1)→T, (1,0)→T, (1,1)→T.
- Prompt: "What weights make this OR?" → (pause) → reveal **w₁=1, w₂=1, b=−0.5**, verify rows.

**Say**
- Interactive beat — stop and ask: "what bias makes a *single* on enough?"
- Take guesses; the key insight: the bias moves from −1.5 (AND) to −0.5 (OR).
- Then reveal and verify. Same machine, different threshold.

**Cite** — [CS189-L2]. (Uses `next_slide()` to hold for answers.)

## Slide 15 — XOR breaks the perceptron (ANIMATED) → scene `LogicGates`
**On slide**
- Truth table for XOR: (0,0)→F, (0,1)→T, (1,0)→T, (1,1)→F.
- 2D plot of the 4 points colored by class; T on opposite corners, F on the other two.
- Animation: a candidate line rotates through a few angles — always misclassifies a point (flash red).
- Conclusion banner: "Not linearly separable — one line can't do it."

**Say**
- AND and OR were easy. Now XOR.
- The two T's sit on opposite corners — no straight line peels them from the F's.
- Try a couple of lines; one point always lands wrong.
- Land it: "the limit of a single perceptron" (Minsky & Papert, 1969).
- **Lunch cliffhanger:** "After lunch — how two of these fix it."

**Cite** — [JM-6.2]; historical limit Minsky & Papert (1969) **[VERIFY]**.

## Slide 16 — Notebook checkpoint: NB1 (build together)
**On slide**
- "In NumPy: build a perceptron, set weights for **AND**, then **OR**. Then try **XOR** — and watch it fail."

**Say**
- Guided walk-through — they type/run with you (we build this notebook after the slides are done).
- They re-derive the gate weights in code; confirms the animation.
- End on XOR failing in code too; hold over lunch.

**Cite** — [JM-6.2].
> NB1 → notebooks/01_perceptron_numpy.ipynb (build AFTER slides; outputs → notebooks/outputs/)

---

# BLOCK 2 — Make it deep, make it learn (1:30–4:00)

## Slide 17 — MLP solves XOR (ANIMATED, assemble from AND + OR) → scene `LogicGates`
**On slide**
- Step 1: bring back the **OR** node and the **AND** node from Slides 13–14, side by side (highlight them as the pieces we already built).
- Step 2: animate them **assembling** into one network: x₁, x₂ → **h₁ = OR**, **h₂ = AND** → output **y = step(h₁ − h₂ − 0.5)**.
- Step 3: re-plot the 4 points in the new (h₁, h₂) space:
  - (0,0)→(0,0) F · (0,1)&(1,0)→(1,0) T · (1,1)→(1,1) F
- A single line now separates T from F. Draw it.

**Say**
- The fix: don't use one line — reuse the OR and AND nodes we just built as a hidden layer.
- Make the assembly explicit: "here's OR, here's AND — snap them together, add one output node."
- Key move: the hidden layer **remaps** the points; show the same 4 points land in new positions where one line works.
- "XOR = OR AND-NOT AND." This is the whole idea of depth: build features, then classify them.

**Cite** — [JM-6.2.1] (XOR with hidden nodes, Fig 6.6); [JM-6.3].

## Slide 18 — Feedforward networks, formalized
**On slide**
- Diagram: input → hidden layer → output (generalize the XOR net).
- h = g(W⁽¹⁾x + b⁽¹⁾),  ŷ = g(W⁽²⁾h + b⁽²⁾)
- Note (descriptive): "Feedforward networks indicate that data only flows from input to output."

**Say**
- Same node as before, stacked — point at the repeated z = Wx + b.
- Hidden layer builds intermediate features; output combines them (just like OR+AND → XOR).
- Name it: multilayer perceptron / feedforward network.

**Cite** — [JM-6.3].

## Slide 19 — Activations: why a non-linearity (decision-boundary figure)
**On slide**
- Static figure (need not animate): a scatter of two mixed-color point clouds that are NOT linearly separable.
  - A straight line through them → some points misclassified (mark them).
  - A curved/closed boundary around one cloud → classifies correctly.
- The collapse: W₂(W₁x) = (W₂W₁)x — two linear layers = one linear layer.
- Requirements: **non-linear** (so we can bend the boundary) + **differentiable** (so we can train by gradients).

**Say**
- Tie straight back to XOR: linear = straight boundaries only.
- Show the figure: a straight line can't separate these; a non-linear boundary can — that's what activations buy us.
- Then the algebra in one line: without a non-linearity, stacked layers collapse to one linear layer (depth is wasted).
- Must be differentiable because training rides on gradients (backprop, soon).

**Cite** — [JM-6.1]; decision-boundary framing [CS189-L2].

## Slide 20 — The usual three: sigmoid, tanh, ReLU
**On slide**
- σ(z) = 1/(1+e⁻ᶻ) → (0,1); tanh(z) → (−1,1); ReLU(z) = max(0, z).
- Mini-plots of each.
- Notes: "sigmoid/tanh saturate (flat tails → ~0 gradient)"; "ReLU: gradient 1 for z>0".
- ReLU footnote: "non-differentiable only at z = 0; in practice we just set the derivative there to 0."

**Say**
- Sigmoid: classic, squashes to (0,1), but flattens at the ends.
- tanh: zero-centered, same saturation issue.
- ReLU: dead simple, gradient 1 when active — the modern default.
- **Vanishing gradients (talk about it here):** backprop multiplies gradients layer by layer. Sigmoid/tanh derivatives are < 1 (sigmoid's max is 0.25), so through many layers the product shrinks toward 0 — early layers barely learn. ReLU's gradient is exactly 1 when active, so it doesn't shrink the signal → deep nets train better.
- **ReLU differentiability (your check, tightened):** ReLU is differentiable everywhere except the single kink at z = 0. Frameworks just assign a value there (usually 0, sometimes 1); it's a measure-zero case that never matters. So "differentiable almost everywhere," which is enough.

**Cite** — [JM-6.1] (Fig 6.1, 6.3; saturation); [CS189-L17] (vanishing gradient; ReLUs).

## Slide 21 — What do we minimize? The loss
**On slide**
- softmax: pᵢ = e^{zᵢ} / Σⱼ e^{zⱼ}
- cross-entropy: L = − Σᵢ yᵢ log pᵢ = − log p_correct
- Regression aside: MSE = (1/n) Σᵢ (yᵢ − ŷᵢ)²

**Say**
- Raw scores → softmax → probabilities that sum to 1.
- Cross-entropy = negative log prob of the correct class — punishes confident wrongness.
- MSE is the regression analogue (show the equation), but today we classify.
- **Note to self:** explain language modeling as a classification task (predict the next token out of the vocabulary) — that's why CE loss carries straight into NLP/transformers later in the week.

**Cite** — [JM-6.6.1]; [CS189-L17].

## Slide 22 — How we minimize: gradient descent (review)
**On slide**
- θ ← θ − η ∇θ L; η = learning rate.
- Loss-landscape cartoon (ball downhill).
- "SGD = gradient on a mini-batch."

**Say**
- Gradient points uphill; step opposite to reduce loss.
- Learning rate = step size: too big overshoots, too small crawls.
- Mini-batches = SGD. That's the engine; backprop is how we get the gradient.

**Cite** — [CS189-L3][CS189-L5]; [JM-6.6.2].

## Slide 23 — Backprop, part 1: forward pass with numbers (3-layer net)
**On slide**
- A **three-layer** net: input → hidden 1 → hidden 2 → output.
- Numbers plugged in: compute h⁽¹⁾, then h⁽²⁾, then ŷ, then L. Show the values.

**Say**
- Use the deeper net so the chain has several links — it makes the next slide's chain rule obvious.
- Walk one example all the way through to a loss value; it's just arithmetic, left to right.
- "Now: how should a weight in the *first* layer change to lower this number?" → next.

**Cite** — [JM-6.6.3]; [CS189-L16].

## Slide 24 — Backprop, part 2: it's just the chain rule (3-layer net)
**On slide**
- For a first-layer weight: ∂L/∂w⁽¹⁾ = (∂L/∂ŷ)(∂ŷ/∂h⁽²⁾)(∂h⁽²⁾/∂h⁽¹⁾)(∂h⁽¹⁾/∂w⁽¹⁾)
- Tagline: **"Backprop = gradient descent + the chain rule."**

**Say**
- With two hidden layers, a first-layer weight's effect travels through *both* — the chain has four factors. Point at each.
- That's the whole trick: multiply local derivatives along the path back from the loss.
- Name each factor, don't grind the algebra. Reassure: PyTorch does this automatically next block.

**Cite** — [JM-6.6.3]; [CS189-L16].

## Slide 25 — Backprop, part 3: the picture (ANIMATED)
**On slide**
- Animation: forward lights up left→right; gradients flow right→left through the layers; weights nudge; loss drops.

**Say**
- Narrate two passes: forward computes the loss, backward distributes "blame" layer by layer.
- One gradient step → loss ticks down.
- This loop, repeated, *is* training.

**Cite** — [JM-6.6.3][CS189-L16].

## Slide 26 — Notebook checkpoint: NB2 (build together)
**On slide**
- "Implement the MLP from scratch (NumPy): forward → loss → manual backward → training loop. Solve XOR."

**Say**
- Guided walk-through (notebook built after slides).
- Closes the XOR arc a second way — in code, with real gradient descent.
- "We wrote `backward` by hand — remember this; Block 3 automates it."

**Cite** — [JM-6.3][JM-6.6].
> NB2 → notebooks/02_mlp_from_scratch.ipynb (build AFTER slides)

---

# BLOCK 3 — Do it for real: PyTorch (4:15–5:30)

## Slide 27 — Why a framework
**On slide**
- "We just wrote `backward` by hand." Frameworks give us:
  - **Composability** — stack layers like Lego; build a net as a sequence of layers without re-deriving anything.
  - **Build on others' work** — load pretrained networks and extend/fine-tune them (preview of Day 3 & 4).
  - **Speed** — highly optimized GPU implementations (far faster than our NumPy).
  - **Autodiff** — gradients computed for us, for any architecture.

**Say**
- Callback to NB2's hand-written backward — nobody wants that per model.
- **Composability:** a framework lets you express a network as a stack of layers and swap pieces freely; you don't rewrite forward/backward each time.
- **Standing on shoulders:** you can download a network someone trained on millions of examples and build on top of it — that's transfer learning, central to Days 3–4.
- **Efficiency:** their kernels (e.g. cuDNN) are far more optimized than anything we'd hand-write, and they run on GPU.
- **Autodiff:** it records your computation and differentiates it automatically — no manual chain rule.

**Cite** — software pillar (Slide 7); autodiff = computation graph [JM-6.6.3]; pretrain/finetune previews Day 3 [CS189-L19/L22-L23].

## Slide 28 — PyTorch building blocks
**On slide**
- `Tensor` — an n-dimensional array (like NumPy) that can live on a GPU.
- `autograd` — set `requires_grad=True`; PyTorch records operations into a graph; `.backward()` fills each tensor's `.grad`.
- `nn.Module` — base class for models/layers; you define layers in `__init__` and the forward pass in `forward()`; parameters are tracked automatically.
- `torch.optim` — optimizers (SGD, Adam) that read `.grad` and update the weights via `optimizer.step()`.

**Say**
- Map each to NB2: `Tensor` = our arrays (now GPU-capable), `autograd` = our hand-written backward, `nn.Module` = our layer functions packaged as a class, `optim` = our weight-update rule.
- `requires_grad` is the switch that tells PyTorch to track a tensor for gradients.
- `nn.Module`: define structure once in `__init__`, the data flow in `forward`; PyTorch finds all the parameters for you.
- The optimizer is just our θ ← θ − η∇θL, automated and with better step-size rules (Adam).

**Cite** — mirrors NB2; [JM-6.6.3]; PyTorch specifics → official docs [general].

## Slide 29 — The training loop (reused all week)
**On slide**
- 1. ŷ = model(x) · 2. loss = criterion(ŷ, y) · 3. optimizer.zero_grad() · 4. loss.backward() · 5. optimizer.step()
- Banner: "This loop returns on Day 3 (CNN) and Day 4 (fine-tuning)."

**Say**
- Walk the five lines slowly — heart of every model this week.
- `zero_grad`: gradients accumulate by default; clear them each step. Common gotcha.
- Memorize the shape of this loop.

**Cite** — mirrors [JM-6.6].

## Slide 30 — Notebook checkpoint: NB3 (build together)
**On slide**
- "Rebuild the MLP in PyTorch with `nn.Module`. Train on **Fashion-MNIST**. Plot loss & accuracy."

**Say**
- Guided walk-through (notebook built after slides).
- Same network as NB2, now in PyTorch — less code, faster.
- First "real" model on real images; let the payoff land.

**Cite** — [CS189-L16].
> NB3 → notebooks/03_pytorch_mlp_fashionmnist.ipynb (build AFTER slides)

## Slide 31 — Three things to know exist (light mentions)
**On slide**
- **train/val/test** + overfitting (plot: train loss ↓, val loss ↑).
- **regularization → dropout:** during training, randomly switch off a fraction of nodes each step.
- **optimizers**: SGD → Adam ("we'll just use Adam").

**Say**
- Name them so the words aren't foreign later — don't teach deeply.
- Overfitting: memorizing train, failing on new data — diverging curves.
- **Dropout (expand):** each training step, randomly "drop" some nodes (set them to 0) so the network can't lean on any single node; this spreads the work and reduces overfitting. At test time all nodes are used.
- Adam = smarter, adaptive step sizes; our default.

**Cite** — [CS189-L1]; [CS189-L22] (dropout); [CS189-L23] (Adam).

## Slide 32 — Wrap + bridge to Day 2
**On slide**
- Recap chain: node → layer → loss → gradient descent → backprop → PyTorch loop.
- Bridge: "Tomorrow, we will focus on some ways text has been dealt with computationally, and how text is dealt with in the context of neural nets."

**Say**
- Retrace the day in one breath using the recap chain.
- Pose the open question: text isn't numbers — how do we feed it to a net?
- End on the hook, take Q&A.

**Cite** — throughline (course design).

---

## Sources (verify here)

Berkeley CS189/289A, Introduction to Machine Learning, **Spring 2025**
(J. Shewchuk). Course page: https://people.eecs.berkeley.edu/~jrs/189/
Lecture notes at `…/lec/NN.pdf`:
- [CS189-L1]  Lecture 1 — intro; train/val/test; over/underfitting. https://people.eecs.berkeley.edu/~jrs/189/lec/01.pdf
- [CS189-L2]  Lecture 2 — linear classifiers; decision boundaries; perceptrons. https://people.eecs.berkeley.edu/~jrs/189/lec/02.pdf
- [CS189-L3]  Lecture 3 — gradient descent, SGD, perceptron learning. https://people.eecs.berkeley.edu/~jrs/189/lec/03.pdf
- [CS189-L5]  Lecture 5 — optimization; influence of step size. https://people.eecs.berkeley.edu/~jrs/189/lec/05.pdf
- [CS189-L16] Lecture 16 — neural networks; backpropagation. https://people.eecs.berkeley.edu/~jrs/189/lec/16.pdf
- [CS189-L17] Lecture 17 — vanishing gradient; ReLUs; softmax + cross-entropy. https://people.eecs.berkeley.edu/~jrs/189/lec/17.pdf
- [CS189-L18] Lecture 18 — neuron biology; training heuristics. https://people.eecs.berkeley.edu/~jrs/189/lec/18.pdf
- [CS189-L19] Lecture 19 — CNNs (reading: Krizhevsky et al. 2012, AlexNet/ImageNet). https://people.eecs.berkeley.edu/~jrs/189/lec/19.pdf
- [CS189-L22] Lecture 22 — generalization; augmentation; ℓ2; dropout. https://people.eecs.berkeley.edu/~jrs/189/lec/22.pdf
- [CS189-L23] Lecture 23 — batch norm; ResNets; Adam/AdamW. https://people.eecs.berkeley.edu/~jrs/189/lec/23.pdf
- [CS189-TEXT] Optional text: Bishop & Bishop, *Deep Learning*, Springer 2024. https://www.bishopbook.com

Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft), **Jan 6 2026
release**, **Ch.6 Neural Networks**. PDF: https://web.stanford.edu/~jurafsky/slp3/6.pdf ;
slides: https://web.stanford.edu/~jurafsky/slp3/slides/nn25aug.pdf
- [JM-6]     Ch.6 intro — NNs as learned non-linear function approximators.
- [JM-6.1]   §6.1 Units — z = w·x + b then non-linearity; sigmoid/tanh/ReLU; saturation.
- [JM-6.2]   §6.2 The XOR problem — a single linear unit cannot compute XOR.
- [JM-6.2.1] §6.2.1 The solution — XOR with ReLU hidden nodes (Fig 6.6, after Goodfellow et al. 2016).
- [JM-6.3]   §6.3 Feedforward Neural Networks — h = g(Wx + b).
- [JM-6.6]   §6.6 Training Neural Nets.
- [JM-6.6.1] §6.6.1 Loss function — cross-entropy = −log prob of correct class.
- [JM-6.6.2] §6.6.2 Computing the gradient.
- [JM-6.6.3] §6.6.3 Computation Graphs — forward + backward (backprop).

Applications & history:
- [AlphaFold] Jumper et al., "Highly accurate protein structure prediction with AlphaFold," *Nature* 596 (2021); breakthrough at CASP14 (2020). **[VERIFY before slide]**
- [Poverty] Jean et al., "Combining satellite imagery and machine learning to predict poverty," *Science* 353(6301), 2016 — anchor for the poverty/aid example; confirm your colleague's specific work. **[VERIFY]**
- [CS230-L5] Stanford CS230 — "AI + Healthcare" lecture. https://cs230.stanford.edu/lecture/
- [UAT] Cybenko (1989); Hornik, Stinchcombe & White (1989). **[VERIFY wording]**
- [MinskyPapert] Minsky & Papert, *Perceptrons* (1969) — XOR limitation. **[VERIFY]**
- History dates: perceptron (Rosenblatt, 1958); gradient descent (Cauchy, 1847); backprop popularized (Rumelhart, Hinton & Williams, 1986; earlier roots Werbos 1974 / Linnainmaa 1970). **[VERIFY]**
- [general] PyTorch specifics → official PyTorch docs.

### Items flagged for your verification
1. **[JM-6]** Slide 5 — confirm the intro framing line.
2. **Slide 6** — AlphaFold framing + your colleague's exact poverty/aid work.
3. **Slide 7** — ImageNet/AlexNet figures (~15% vs ~26% top-5) and the three history dates.
4. **Slide 8 / [UAT]** — universal-approximation attribution & wording.
5. **Slide 15 / [MinskyPapert]** — whether to put the 1969 attribution on-slide.
