# CLAUDE.md — Data Science Lecture Week (Project Context)

## What this project is
A 4-day applied data-science lecture series. Each day = one topic, ~5 hours of
lecture **and** lab combined, taught at an applied level with *enough* theory to
make application make sense.

**Day order (intended teaching order, builds cumulatively):**
1. Neural Networks
2. NLP
3. Computer Vision (CV)
4. Transformers

## Hard constraints
- Session window: **10:00am–6:00pm**. Includes breakfast + lunch.
- **~5 hours** of actual lecture+lab content per day (the rest is meals/breaks).
- Deliverables per topic: (a) syllabus, (b) slides built with **Manim Slides**
  (manim-ce for animations), (c) Jupyter notebooks for the applied/lab portion.

## Audience & pedagogy
- **Very mixed** background.
- **Notation philosophy:** USE mathematical notation to explain and showcase
  ideas clearly. SHOW derivatives, losses, dot products, etc. Do **NOT** make
  attendees do proofs, work through derivations by hand, or grind algebra.
  Math is for illumination, not exercises.
- Teaching loop: motivate (slides) → apply immediately (notebook) → return to
  slides to generalize. Favor the "fail then fix" pattern (e.g. perceptron fails
  XOR → motivates MLP).

## Standard daily time skeleton (~5 hrs content)
- 10:00–10:30  Breakfast / settle
- 10:30–12:30  Lecture Block 1 (theory) + short interleaved notebook
- 12:30–1:30   Lunch
- 1:30–4:00    Lecture Block 2 + main lab notebook
- 4:00–4:15    Break
- 4:15–5:30    Lecture Block 3 / capstone lab
- 5:30–6:00    Wrap, Q&A, bridge to next day

## The week's throughline (recurring callbacks — emphasize these)
- **Day 1** gives the machinery: layers, loss, gradient descent, backprop,
  PyTorch + OOP training loop.
- **Day 2** gives representation of discrete data (tokenization, embeddings) and
  surfaces the *context/order problem* (static embeddings have no context).
- **Day 3** gives spatial structure (convolution), hierarchical features,
  residual/skip connections, and the pretrain→finetune (transfer learning) idea.
- **Day 4** unifies everything: attention solves Day 2's order/context problem,
  reuses Day 3's residual connections + pretrain/finetune paradigm, and CLIP
  fuses Day 2 embeddings with Day 3 vision encoders.

## Per-day content (working outline)

### Day 1 — Neural Networks
Intro to me → intro to deep learning (relation to ML/AI; medical example +
faculty example; "why now" = data + compute + autodiff/algorithms) → unique
properties (universal approximation, show + intuition) → perceptron (neuron
analogy → w·x + b, decision boundary, animated graph) → [NB: numpy perceptron,
fails XOR] → MLP / feedforward (XOR solved) → activation functions (nonlinear +
differentiable; sigmoid/tanh/ReLU) → loss (softmax→CE; MSE noted) → gradient
descent review → backprop (2-layer, forward-pass numbers, chain rule, manim
"gradient descent + chain rule") → [NB: MLP from scratch] → PyTorch (tensors,
autograd, nn.Module, training loop) → [NB: PyTorch MLP on MNIST/Fashion-MNIST].
Light mentions: train/val/test + overfitting, optimizers (SGD→Adam).

### Day 2 — NLP
Why text is hard (discrete, variable length, ambiguous, sparse, no native
numeric rep) → tokenization (word/char → subword BPE & WordPiece; tiktokenizer
demo; primes transformers) → [NB: train a BPE tokenizer, inspect merges] →
sparse reps (one-hot, BoW, tf-idf + limitations) → [NB: tf-idf + logistic
regression sentiment] → embeddings + distributional hypothesis (Word2Vec;
analogies/vector arithmetic) → [NB: pretrained embeddings, nearest neighbors,
PCA/t-SNE viz] → static embedding limits (polysemy, no order/context) →
optional 5-min RNN/LSTM historical note → bridge to attention.

### Day 3 — Computer Vision
Image representation (pixels, channels, H×W×C tensors) → why not an MLP on pixels
(param explosion, no spatial structure, not translation invariant) → convolution
(kernels, feature maps, stride, padding; manim sliding-kernel animation; edge
detectors) → pooling + feature hierarchy (edges→textures→parts→objects) → CNN
architecture (conv-relu-pool → FC head; LeNet/AlexNet/ResNet lineage; introduce
residual/skip connections) → [NB: hand-crafted kernels on an image, visualize
feature maps] → [NB: train a small CNN in PyTorch on CIFAR-10/Fashion-MNIST —
callback to Day 1 OOP] → transfer learning (pretrained backbones, feature
extraction vs fine-tuning) → [NB: fine-tune pretrained ResNet on small dataset]
→ bridge (hierarchical features + pretrain/finetune; images as patches → tokens).

### Day 4 — Transformers
Recap motivation (Day 2 context/order problem) → attention from scratch
(Q,K,V; scaled dot-product softmax(QKᵀ/√d)V; manim attention-weight heatmap) →
self vs cross attention; multi-head → positional encodings (solves Day 2 order
problem) → full block (attention + FFN + residuals [callback Day 3] + layernorm;
Jurafsky SLP framing) → [NB: implement multi-head attention, visualize weights]
→ model families (BERT encoder/MLM vs GPT decoder/autoregressive vs T5 enc-dec)
→ pretraining + finetuning paradigm → BERT fine-tuning (why a small fine-tuned
BERT can rival much larger LLMs on a task; foundation models) → [NB: fine-tune
BERT on classification] → LoRA / PEFT (freeze base, learn low-rank ΔW; why it
matters) → [NB: LoRA fine-tune with peft] → multimodal: CLIP (contrastive
image-text shared space; zero-shot) + VLMs → [NB (optional): CLIP zero-shot].
NOTE: Day 4 is content-heavy — prioritize attention + BERT/finetuning + LoRA;
CLIP/VLM is the stretch goal.

## Key reference
- Jurafsky & Martin, *Speech and Language Processing* — core inspiration for the
  transformer/attention framing.

## Tooling notes
- Slides: Manim Slides (interactive slide presentation on top of manim-ce).
- Labs: Jupyter notebooks, PyTorch + Hugging Face (tokenizers, transformers,
  peft), scikit-learn, gensim/GloVe for embeddings.
- tiktokenizer.vercel.app for the live tokenizer demo (Day 2).

## Folder structure
- `notebooks/`        — Jupyter lab notebooks (`.ipynb`)
- `notebooks/outputs/`— generated outputs from notebooks (figures, saved models, etc.)
- `manim_slides/`     — Manim Slides source (`.py` scenes)
- `manim_slides/outputs/` — rendered slide outputs
- `docs/`             — reference notes (manim-slides notes, sources/citations, slide drafts)

## Notebook code style (STRICT)
- NO emojis anywhere.
- NO large separator characters in print output (no `=====`, no `-----`), and
  NO printed indentation for formatting. Keep prints plain and minimal.
- Code should read as natural / human-written, not over-commented or robotic.
- **Functions: camelCase** (e.g. `trainModel`, `computeLoss`).
- **Variables: snake_case** (e.g. `learning_rate`, `train_loader`).
- Comments: **lowercase only**, sparing.
- Function docstrings: short and to the point.
- Markdown cells OK to mark notebook stages and draw explicit slide connections
  (though presenter may just verbalize these).
- Write notebook-generated outputs into `notebooks/outputs/`.

## Slide style (Manim Slides)
- **Minimalism is VISUAL/aesthetic only** (clean charcoal bg, white text, uncluttered
  layout). Text itself can be **verbose** — the presenter does not want sparse text.
- Background: pleasant **dark charcoal**. Text: **natural white**.
- **Lists use bullet points (•), NOT dashes.** Use **proper sentence-case
  capitalization** (start each bullet/sentence with a capital).
- Design so images can be slotted in later (leave space / use placeholders).
- Study manim-slides + manim-ce docs thoroughly; keep a notes doc in `docs/`.
- Render outputs into `manim_slides/outputs/`.

## Presenter-facing docs (the presenter wants these)
- Keep companion docs written FOR the presenter to study/explain from:
  `docs/day1_presenter_guide.md` (Q&A, confusions, the math) and
  `docs/day1_equations_explained.md` (how to read every on-slide equation).
- Title slide credit line: "Materials based on Jurafsky's SLP and Berkeley's
  Applied Machine Learning (INFO 251, Joshua Blumenstock)."
- When the presenter asks for an explanation of slide content, write it out fully
  and clearly (they use these to rehearse).

## Slide CONTENT sourcing rule (STRICT)
All slide content must be anchored in real, recent university lecture slides or
in Jurafsky & Martin SLP3. Recency: **2026 preferred, 2025 acceptable** fallback.
Acceptable sources:
- Berkeley CS189 (ML), Berkeley Intro to Deep Learning, Berkeley Applied NLP
- **Berkeley INFO 251 Applied Machine Learning (Joshua Blumenstock)** — the
  perceptron->MLP->backprop->tuning narrative on Day 1 is based on this deck
  (saved PDFs in `manim_slides/examples/`).
- Stanford CS230 (DL), CS231n (CV), CS224n (NLP), CS229 (ML)
- Jurafsky & Martin, SLP3 (https://web.stanford.edu/~jurafsky/slp3/) — chapter
  PDFs linked there; ch. 7 = Neural Networks & Neural LMs.
Every drafted slide should CITE the specific lecture slide / textbook passage it
draws from, so the presenter can verify quickly.

## Refinements (from presenter)
- **NN "Why now":** frame as compute (GPUs) + data (large labeled datasets) +
  software/algorithms (autodiff frameworks like PyTorch). Keep it concrete.
- **Universal approximation, distilled:** "Any sufficiently wide two-layer neural
  network is enough to approximate any given function." (State + intuition; no proof.)
- **NLP RNN/LSTM note:** do NOT frame as merely "the old way." Note these are
  still actively researched and useful for some tasks (e.g. streaming/long-
  sequence efficiency work). Present as a different design point, then contrast
  with attention.
- **NLP Word2Vec:** emphasize it foreshadows ideas key to transformers — namely
  **pre-training** (learn reusable representations from unlabeled text, transfer
  them downstream).

## Day 1 notebooks (current)
Follow **Notebook code style (STRICT)** above exactly (camelCase functions,
snake_case variables, lowercase sparing comments, short docstrings, NO emojis,
NO separator chars in prints, no printed indentation). Write generated
figures/models to `notebooks/outputs/`.

**Demo notebook (the ONLY in-lecture demo):**
- `notebooks/01_pytorch_mlp_fashionmnist.ipynb` — MLP rebuilt with PyTorch
  `nn.Module`; Fashion-MNIST via torchvision; CrossEntropyLoss + Adam; the
  canonical 5-line training loop; plot loss/accuracy; sample figure to
  `notebooks/outputs/`. (The from-scratch MLP demo was removed so it does not
  spoil the lab assignment below.)

**Lab assignment (post-lecture, ~90–120 min): "Decision boundaries & the power
of depth" (Option A + optional Track B).** 2D synthetic data (moons / circles /
spirals); build an MLP, visualize the decision boundary, sweep width / depth /
activation, induce + fix overfitting. Optional **Track B**: implement forward +
manual backward in NumPy (formulas in markdown) and pass a prewritten gradient
check. Two notebooks live in `notebooks/`:
- `lab_decision_boundaries_starter.ipynb` — shared harness provided; "YOUR TURN"
  blanks for the model + training loop + (optional) NumPy backward; optional-B
  hint markdown cells.
- `lab_decision_boundaries_solution.ipynb` — canonical worked implementation.

## Status
- [x] Syllabus outlines drafted (all 4 days) — in chat + this file
- [x] Day 1 slides (Manim Slides) — `manim_slides/day1_neural_networks.py`
- [x] Day 1 demo notebook (PyTorch) + lab assignment (A + optional B)
- [ ] Slides (Manim Slides) — Days 2–4
