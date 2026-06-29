"""day 1 neural networks - manim slides deck.

five scenes, presented in order:
    Day1Intro -> Day1Perceptron -> Day1LogicGates -> Day1Training -> Day1PyTorch

style: charcoal background, near-white ink, sparse accent colors. image
placeholders are dashed boxes labeled "image: ..." - swap for ImageMobject later.
speaker notes are attached via next_slide(notes=...); press S in the browser
presentation (or use manim-slides present) to see them.
see docs/day1_neural_networks_slides.md and docs/day1_presenter_guide.md.
"""

import textwrap
from pathlib import Path

import numpy as np
from manim import *
from manim_slides import Slide

IMG_DIR = Path(__file__).parent / "images" / "day1"


def mkImage(name, max_w=5.0, max_h=4.0):
    """load an image and scale it to fit within max_w x max_h."""
    img = ImageMobject(str(IMG_DIR / name))
    factor = min(max_w / img.width, max_h / img.height)
    img.scale(factor)
    return img

# palette
CHARCOAL = "#1e1e22"
INK = "#f5f5f5"
MUTED = "#9aa0a6"
BLUE = "#5aa9e6"
AMBER = "#e6b35a"
GREEN = "#7bd88f"
RED = "#e06c75"

config.background_color = CHARCOAL
Text.set_default(color=INK)
MathTex.set_default(color=INK)


# layout helpers

def fitWidth(mob, max_w):
    if mob.width > max_w:
        mob.scale(max_w / mob.width)
    return mob


def fitBox(mob, max_w, max_h):
    factor = min(max_w / mob.width, max_h / mob.height, 1.0)
    if factor < 1.0:
        mob.scale(factor)
    return mob


def mkTitle(text):
    """slide title, pinned top, scaled to fit."""
    t = Text(text, font_size=40, color=INK, weight=BOLD)
    fitWidth(t, 12.0)
    return t.to_edge(UP, buff=0.5)


def mkBullets(items, font_size=26, buff=0.32, width_chars=66):
    """left-aligned, word-wrapped bullet list (sentence-case, bullet points)."""
    lines = []
    for s in items:
        if s and s[0].isalpha():
            s = s[0].upper() + s[1:]
        wrapped = textwrap.fill(s, width=width_chars)
        lines.append(Text("•  " + wrapped, font_size=font_size, color=INK, line_spacing=0.85))
    g = VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=buff)
    return g


def mkParagraph(text, font_size=24, color=INK, width_chars=54):
    wrapped = "\n".join(textwrap.wrap(text, width=width_chars))
    return Text(wrapped, font_size=font_size, color=color, line_spacing=0.85)


def mkImagePlaceholder(label="image", w=4.0, h=3.0):
    box = DashedVMobject(Rectangle(width=w, height=h, color=MUTED, stroke_width=2))
    lbl = Text("image: " + label, font_size=20, color=MUTED).move_to(box.get_center())
    fitWidth(lbl, w - 0.4)
    return VGroup(box, lbl)


def mkTruthTable(headers, rows, font_size=30):
    cols = len(headers)
    cells = [Text(h, font_size=font_size, color=BLUE, weight=BOLD) for h in headers]
    data_groups = []
    for r in rows:
        rowcells = []
        for j, v in enumerate(r):
            if j == cols - 1:
                col = GREEN if v == "T" else MUTED
            else:
                col = INK
            t = Text(str(v), font_size=font_size, color=col)
            cells.append(t)
            rowcells.append(t)
        data_groups.append(rowcells)
    grid = VGroup(*cells).arrange_in_grid(rows=len(rows) + 1, cols=cols, buff=0.45)
    row_groups = [VGroup(*rc) for rc in data_groups]
    return grid, row_groups


def mkPerceptron(w1="1", w2="1", b="-1.5"):
    """two-input perceptron diagram. returns (group, parts_dict)."""
    in1 = Circle(radius=0.32, color=BLUE).move_to(LEFT * 4.2 + UP * 1.3)
    in2 = Circle(radius=0.32, color=BLUE).move_to(LEFT * 4.2 + DOWN * 1.3)
    x1 = MathTex("x_1", font_size=30).move_to(in1)
    x2 = MathTex("x_2", font_size=30).move_to(in2)
    summ = Circle(radius=0.55, color=INK).move_to(LEFT * 0.4)
    sig = MathTex(r"\Sigma", font_size=40).move_to(summ)
    out = Circle(radius=0.4, color=AMBER).move_to(RIGHT * 2.6)
    y = MathTex("y", font_size=30).move_to(out)
    e1 = Line(in1.get_right(), summ.get_left(), buff=0.1, color=MUTED, stroke_width=3)
    e2 = Line(in2.get_right(), summ.get_left(), buff=0.1, color=MUTED, stroke_width=3)
    e3 = Line(summ.get_right(), out.get_left(), buff=0.1, color=MUTED, stroke_width=3)
    w1l = MathTex("w_1=" + w1, font_size=26, color=AMBER).next_to(e1, UP, buff=0.05)
    w2l = MathTex("w_2=" + w2, font_size=26, color=AMBER).next_to(e2, DOWN, buff=0.05)
    bl = MathTex("b=" + b, font_size=26, color=GREEN).next_to(summ, DOWN, buff=0.25)
    group = VGroup(in1, in2, x1, x2, summ, sig, out, y, e1, e2, e3, w1l, w2l, bl)
    parts = dict(in1=in1, in2=in2, summ=summ, out=out, w1=w1l, w2=w2l, b=bl)
    return group, parts


def mkScalarNet():
    """x -> a1 -> a2 fully-labeled scalar net for backprop. returns (group, parts)."""
    x = Circle(radius=0.34, color=BLUE).move_to(LEFT * 4)
    a1 = Circle(radius=0.34, color=GREEN).move_to(ORIGIN)
    a2 = Circle(radius=0.34, color=AMBER).move_to(RIGHT * 4)
    xl = MathTex("x", font_size=28).move_to(x)
    a1l = MathTex("a_1", font_size=26).move_to(a1)
    a2l = MathTex("a_2", font_size=26).move_to(a2)
    e1 = Line(x.get_right(), a1.get_left(), buff=0.1, color=MUTED, stroke_width=3)
    e2 = Line(a1.get_right(), a2.get_left(), buff=0.1, color=MUTED, stroke_width=3)
    w1l = MathTex("w_1, b_1", font_size=24, color=AMBER).next_to(e1, UP, buff=0.12)
    w2l = MathTex("w_2, b_2", font_size=24, color=AMBER).next_to(e2, UP, buff=0.12)
    tags = VGroup(
        Text("input", font_size=18, color=MUTED).next_to(x, DOWN, buff=0.55),
        Text("hidden", font_size=18, color=MUTED).next_to(a1, DOWN, buff=0.55),
        Text("output", font_size=18, color=MUTED).next_to(a2, DOWN, buff=0.55),
    )
    group = VGroup(x, a1, a2, xl, a1l, a2l, e1, e2, w1l, w2l, tags)
    parts = dict(x=x, a1=a1, a2=a2)
    return group, parts


def mkNetwork(sizes, x_gap=2.4, y_gap=1.1, radius=0.18, color=BLUE):
    layers = []
    nodes = VGroup()
    for li, n in enumerate(sizes):
        col = VGroup()
        for i in range(n):
            yy = (i - (n - 1) / 2) * y_gap
            col.add(Dot([li * x_gap, yy, 0], radius=radius, color=color))
        layers.append(col)
        nodes.add(col)
    edges = VGroup()
    for a, b in zip(layers, layers[1:]):
        for da in a:
            for db in b:
                edges.add(Line(da.get_center(), db.get_center(),
                               stroke_width=1.2, color=MUTED))
    group = VGroup(edges, nodes)
    group.move_to(ORIGIN)
    return group, layers, edges


def mkNotebookHeader():
    badge = RoundedRectangle(width=4.4, height=1.0, corner_radius=0.2, color=GREEN, stroke_width=3)
    txt = Text("NOTEBOOK DEMO", font_size=28, color=GREEN, weight=BOLD).move_to(badge)
    return VGroup(badge, txt)


def stepLabel(z):
    return "T" if z >= 0 else "F"


class Day1Base(Slide):
    """shared slide behavior."""

    # skip reverse-animation generation: faster, and avoids an arrow-tip bug path
    skip_reversing = True

    def clearSlide(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)


# scene 1: intro (slides 1-9)

class Day1Intro(Day1Base):
    def construct(self):
        # s1 title
        title = Text("Neural Networks", font_size=64, color=INK, weight=BOLD)
        subtitle = Text("Day 1 of 4", font_size=28, color=MUTED).next_to(title, DOWN, buff=0.4)
        credit = mkParagraph(
            "Materials based on Jurafsky's SLP and Berkeley's Applied Machine "
            "Learning (INFO 251, Joshua Blumenstock).",
            font_size=18, color=MUTED, width_chars=64).next_to(subtitle, DOWN, buff=0.7)
        self.play(FadeIn(title, shift=UP * 0.3), FadeIn(subtitle))
        self.play(FadeIn(credit))
        self.next_slide(notes="Land the title, set the tone, then move quickly to who you are. Credit the source decks (Jurafsky SLP, Blumenstock INFO 251).")
        self.clearSlide()

        # s2 about me
        t = mkTitle("About me")
        bullets = mkBullets([
            "My name is Carl",
            'I am part of the "Responsibility Lab"',
            "We work on computational social science, mainly at the intersection of CS and Psychology; currently on CS and legal problems",
        ], font_size=30, width_chars=52)
        fitBox(bullets, 11.0, 4.5).next_to(t, DOWN, buff=0.9)
        self.play(FadeIn(t), FadeIn(bullets))
        self.next_slide(notes="Quick intro: Carl, the Responsibility Lab - computational social science at the CS/Psychology intersection, currently CS + legal problems.")
        self.clearSlide()

        # s3 where the week is going
        t = mkTitle("Where this week is going")
        bullets = mkBullets([
            "Day 1  -  Neural Networks",
            "Day 2  -  NLP",
            "Day 3  -  Computer Vision",
            "Day 4  -  Transformers",
        ], font_size=34, buff=0.5).next_to(t, DOWN, buff=0.8)
        self.play(FadeIn(t), LaggedStartMap(FadeIn, bullets, shift=RIGHT * 0.3, lag_ratio=0.2))
        self.next_slide(notes="Each day stands alone but compounds. Today is the foundation. Plant: attention (Day 4) fixes an NLP problem.")
        self.clearSlide()

        # s4 agenda
        t = mkTitle("Today")
        bullets = mkBullets([
            "Morning: from the neuron to the perceptron; the XOR problem",
            "Afternoon: multilayer networks, training, backprop",
            "Late afternoon: PyTorch + hands-on notebooks",
        ], font_size=28).next_to(t, DOWN, buff=0.8)
        self.play(FadeIn(t), FadeIn(bullets))
        self.next_slide(notes="Set the rhythm: slides motivate, notebooks apply. Notebooks are follow-along - they code with you.")
        self.clearSlide()

        # s6 where neural nets show up
        t = mkTitle("Where neural nets show up")
        p1 = mkImage("alphafold.png", max_w=4.4, max_h=2.7)
        c1 = mkParagraph("AlphaFold: a protein's 3D structure from its amino-acid "
                         "sequence; near-experimental accuracy at CASP14 (2020).",
                         font_size=17, width_chars=34).next_to(p1, DOWN, buff=0.25)
        col1 = Group(p1, c1)
        p2 = mkImage("poverty.png", max_w=4.4, max_h=2.7)
        c2 = mkParagraph("Poverty mapping (a colleague's work): ML on satellite "
                         "imagery plus other geospatial data builds high-resolution "
                         "poverty maps, helping governments target cash transfers to "
                         "the poorest families where survey data is missing (Nigeria).",
                         font_size=15, width_chars=36).next_to(p2, DOWN, buff=0.25)
        col2 = Group(p2, c2)
        body = Group(col1, col2).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        fitBox(body, 12.5, 5.4).next_to(t, DOWN, buff=0.5)
        self.play(FadeIn(t), FadeIn(body))
        self.next_slide(notes="AlphaFold: decades-old biology problem (structure from sequence), point at impact. Poverty: ML on satellite + geospatial data produces high-res poverty maps used to target government cash transfers to the poorest families - matters where disaggregated survey data is missing (Nigeria study).")
        self.clearSlide()

        # s7 why now
        t = mkTitle("Why now?")
        cap = mkParagraph(
            "The perceptron dates to 1958, gradient descent even earlier to 1847, "
            "and backprop was popularized in 1986 - the core ideas are decades old. "
            "So what changed?",
            font_size=24, width_chars=52).next_to(t, DOWN, buff=0.4).to_edge(LEFT, buff=0.9)
        bullets = mkBullets([
            "Compute: GPUs make big matrix math cheap",
            "Data: large labeled datasets (e.g. ImageNet)",
            "Software: autodiff frameworks like PyTorch",
        ], font_size=24, width_chars=40).next_to(cap, DOWN, buff=0.45).to_edge(LEFT, buff=1.1)
        infl = mkParagraph(
            "Sidelined for years - then AlexNet won ImageNet in 2012 (~15% vs ~26% "
            "top-5 error). That kicked off the modern boom.",
            font_size=20, color=MUTED, width_chars=34)
        img = mkImage("neural_net_game_change.png", max_w=4.2, max_h=3.0)
        right = Group(img, infl).arrange(DOWN, buff=0.3).to_edge(RIGHT, buff=0.6).shift(UP * 0.1)
        self.play(FadeIn(t), FadeIn(cap))
        self.play(LaggedStartMap(FadeIn, bullets, shift=RIGHT * 0.3, lag_ratio=0.3), FadeIn(right))
        self.next_slide(notes="Say the three dates out loud. The inflection: AlexNet 2012, GPUs + ImageNet. ~15% vs ~26% top-5 (verify figure).")
        self.clearSlide()

        # s9 the biological neuron (image)
        t = mkTitle("The biological neuron")
        img = mkImage("bioneuron.jpg", max_w=6.5, max_h=3.0).next_to(t, DOWN, buff=0.4)
        pts = mkBullets([
            "dendrites = inputs, soma = sums, axon = fires past a threshold",
            "the McCulloch-Pitts neuron (1943) modeled this with propositional logic",
        ], font_size=21, width_chars=72).next_to(img, DOWN, buff=0.35)
        cap = Text("Perceptrons are analogies to neurons, not one-to-one models.",
                   font_size=21, color=MUTED).next_to(pts, DOWN, buff=0.25)
        self.play(FadeIn(t), FadeIn(img))
        self.play(FadeIn(pts), FadeIn(cap))
        self.next_slide(notes="Real neuron: collects signals, fires past threshold. The McCulloch-Pitts model (1943) cast this as propositional logic (AND/OR/NOT from neurons). We borrow only the shape. Insert the labeled neuron image here.")
        self.clearSlide()


# scene 2: perceptron (slides 10-12)

class Day1Perceptron(Day1Base):
    def construct(self):
        # s10 perceptron math
        t = mkTitle("The perceptron")
        perc, parts = mkPerceptron("w_1", "w_2", "b")
        perc.scale(0.62)
        eq = MathTex(r"z = \mathbf{w}\cdot\mathbf{x} + b = \sum_i w_i x_i + b", font_size=32)
        eq2 = MathTex(r"y = f(z)\quad(\text{perceptron: } f=\text{step})", font_size=26, color=MUTED)
        pts = mkBullets([
            "the dot product w.x measures alignment (similarity) of input x with weights w",
            "the bias term b shifts the decision boundary left or right",
        ], font_size=22, width_chars=58)
        content = VGroup(perc, eq, eq2, pts).arrange(DOWN, buff=0.32)
        fitBox(content, 12.5, 5.7).next_to(t, DOWN, buff=0.4)
        self.play(FadeIn(t), FadeIn(perc))
        self.play(Write(eq), FadeIn(eq2))
        self.play(FadeIn(pts))
        self.next_slide(notes="Walk left to right: multiply, sum, add bias, threshold. The dot product w.x is a measure of alignment / similarity between the input vector x and the perceptron's weights w. The bias shifts the decision boundary left or right.")
        self.play(Indicate(parts["w1"], color=AMBER), Indicate(parts["w2"], color=AMBER))
        self.play(Indicate(parts["b"], color=GREEN))
        self.next_slide()
        self.clearSlide()


# scene 3: perceptron -> MLP narrative (Blumenstock-style)

class Day1LogicGates(Day1Base):
    def plotPoint(self, a, b, center, s=1.2):
        return center + RIGHT * (a - 0.5) * 2 * s + UP * (b - 0.5) * 2 * s

    def cornerPlot(self, outs, center, line_pts=None, fail=False):
        box = Square(side_length=1.8, color="#33343a", stroke_width=1.2).move_to(center)
        g = VGroup(box)
        for (a, b), lab in outs.items():
            col = AMBER if lab == "T" else BLUE
            g.add(Dot(center + RIGHT * (a - 0.5) * 1.2 + UP * (b - 0.5) * 1.2, radius=0.08, color=col))
        if line_pts is not None:
            p, q = line_pts
            g.add(Line(center + p, center + q, color=GREEN, stroke_width=3))
        if fail:
            g.add(Text("no line works", font_size=14, color=RED).move_to(center + DOWN * 1.25))
        return g

    def construct(self):
        # L1 linearly separable data
        t = mkTitle("Linearly separable data")
        pts = mkBullets([
            "the perceptron is a linear classifier",
            "its boundary is a hyperplane",
            "predict True on one side, False on the other",
        ], font_size=24, width_chars=40).next_to(t, DOWN, buff=0.5).to_edge(LEFT, buff=0.7)
        eq = MathTex(r"w_0 + w_1 x_1 + \cdots + w_k x_k = 0", font_size=26, color=AMBER).next_to(pts, DOWN, buff=0.5).to_edge(LEFT, buff=0.9)
        plane = NumberPlane(x_range=[0, 4, 1], y_range=[0, 4, 1], x_length=4.2, y_length=4.2,
                            background_line_style={"stroke_color": "#33343a", "stroke_width": 1},
                            axis_config={"include_tip": False, "stroke_color": MUTED, "stroke_width": 1.3}).to_edge(RIGHT, buff=0.7).shift(DOWN * 0.3)
        np.random.seed(3)
        cA = VGroup(*[Dot(plane.c2p(np.random.uniform(0.3, 1.6), np.random.uniform(0.3, 2.2)), color=BLUE, radius=0.07) for _ in range(10)])
        cB = VGroup(*[Dot(plane.c2p(np.random.uniform(2.1, 3.6), np.random.uniform(1.5, 3.6)), color=RED, radius=0.07) for _ in range(10)])
        line = Line(plane.c2p(1.1, 0), plane.c2p(2.7, 4), color=AMBER, stroke_width=4)
        self.play(FadeIn(t), FadeIn(pts), Create(plane), FadeIn(cA), FadeIn(cB))
        self.play(FadeIn(eq), Create(line))
        self.next_slide(notes="The perceptron only ever draws a straight boundary. Ask: what weights separate these two clouds?")
        self.clearSlide()

        # L2 perceptron for AND
        t = mkTitle("Perceptron: AND")
        h = MathTex(r"h(x) = \text{Sign}(b + w_1 x_1 + w_2 x_2)", font_size=26).next_to(t, DOWN, buff=0.5).to_edge(LEFT, buff=0.7)
        rows = [("1", "1", "T"), ("1", "0", "F"), ("0", "1", "F"), ("0", "0", "F")]
        grid, _ = mkTruthTable(["x1", "x2", "y"], rows)
        grid.scale(0.78).to_corner(UR, buff=0.8).shift(DOWN * 0.5)
        sol = mkBullets([
            "one solution:  w1 = 0.5,  w2 = 0.5,  b = -0.8",
            "can you find another?",
        ], font_size=23, width_chars=40).next_to(h, DOWN, buff=0.5).to_edge(LEFT, buff=0.7)
        perc, _ = mkPerceptron("0.5", "0.5", "-0.8")
        perc.scale(0.58).to_corner(DR, buff=0.7)
        tnote = Text("threshold T = -b  (here T = 0.8)", font_size=17, color=MUTED).next_to(perc, UP, buff=0.2)
        self.play(FadeIn(t), FadeIn(h), FadeIn(grid))
        self.play(FadeIn(perc), FadeIn(tnote))
        self.next_slide(notes="A single perceptron computes AND. h = Sign(b + w1x1 + w2x2). One solution w1=w2=0.5, b=-0.8. Bias b is sometimes written w0; threshold T=-b.")
        self.play(FadeIn(sol))
        self.next_slide(notes="Ask if they can find another solution - many weight sets work.")
        self.clearSlide()

        # L3 perceptron: your turn (OR)
        t = mkTitle("Perceptron: your turn (OR)")
        h = MathTex(r"h(x) = \text{Sign}(b + w_1 x_1 + w_2 x_2)", font_size=26).next_to(t, DOWN, buff=0.5).to_edge(LEFT, buff=0.7)
        ask = Text("find weights b, w1, w2", font_size=24, color=AMBER).next_to(h, DOWN, buff=0.4).to_edge(LEFT, buff=0.7)
        rows = [("1", "1", "T"), ("1", "0", "T"), ("0", "1", "T"), ("0", "0", "F")]
        grid, _ = mkTruthTable(["x1", "x2", "y"], rows)
        grid.scale(0.78).next_to(ask, DOWN, buff=0.5).to_edge(LEFT, buff=1.2)
        perc, parts = mkPerceptron("?", "?", "?")
        perc.scale(0.6).to_edge(RIGHT, buff=0.9).shift(DOWN * 0.2)
        self.play(FadeIn(t), FadeIn(h), FadeIn(ask), FadeIn(grid), FadeIn(perc))
        self.next_slide(notes="Interactive: have them find OR weights. A solution: w1=0.5, w2=0.5, b=-0.3 (one 'on' is enough).")
        self.play(
            Transform(parts["w1"], MathTex("w_1=0.5", font_size=24, color=AMBER).move_to(parts["w1"])),
            Transform(parts["w2"], MathTex("w_2=0.5", font_size=24, color=AMBER).move_to(parts["w2"])),
            Transform(parts["b"], MathTex("b=-0.3", font_size=24, color=GREEN).move_to(parts["b"])),
        )
        self.next_slide(notes="Reveal one solution. The bias is less negative than AND, so a single 'on' clears the threshold.")
        self.clearSlide()

        # L4 perceptron: XOR?
        t = mkTitle("Perceptron: XOR?")
        pts = mkBullets([
            "you've seen AND and OR",
            "a perceptron for XOR?  Impossible (Minsky & Papert, 1969)",
            "XOR is not linearly separable",
        ], font_size=22, width_chars=44).next_to(t, DOWN, buff=0.4).to_edge(LEFT, buff=0.7)
        rows = [("1", "1", "F"), ("1", "0", "T"), ("0", "1", "T"), ("0", "0", "F")]
        grid, _ = mkTruthTable(["x1", "x2", "y"], rows)
        grid.scale(0.66).to_corner(UR, buff=0.8).shift(DOWN * 0.3)
        and_p = self.cornerPlot({(0, 0): "F", (0, 1): "F", (1, 0): "F", (1, 1): "T"},
                                LEFT * 4.2 + DOWN * 1.9, line_pts=(UP * 0.95 + LEFT * 0.25, DOWN * 0.25 + RIGHT * 0.95))
        or_p = self.cornerPlot({(0, 0): "F", (0, 1): "T", (1, 0): "T", (1, 1): "T"},
                               DOWN * 1.9, line_pts=(UP * 0.25 + LEFT * 0.95, DOWN * 0.95 + RIGHT * 0.25))
        xor_p = self.cornerPlot({(0, 0): "F", (0, 1): "T", (1, 0): "T", (1, 1): "F"},
                                RIGHT * 4.2 + DOWN * 1.9, fail=True)
        labs = VGroup(
            Text("AND", font_size=16, color=MUTED).next_to(and_p, UP, buff=0.05),
            Text("OR", font_size=16, color=MUTED).next_to(or_p, UP, buff=0.05),
            Text("XOR", font_size=16, color=MUTED).next_to(xor_p, UP, buff=0.05),
        )
        self.play(FadeIn(t), FadeIn(pts), FadeIn(grid))
        self.play(FadeIn(and_p), FadeIn(or_p), FadeIn(xor_p), FadeIn(labs))
        self.next_slide(notes="AND and OR are separable (a line works); XOR is not - no straight line splits the two True corners from the False ones. Minsky & Papert 1969.")
        self.clearSlide()

        # L9 perceptron summary (self-contained)
        t = mkTitle("Perceptron: summary")
        pts = mkBullets([
            "a perceptron is a linear classifier",
            "it can represent AND and OR",
            "it cannot represent XOR, which is not linearly separable",
            "its decision boundary is always a straight line (a hyperplane)",
        ], font_size=26, width_chars=52).next_to(t, DOWN, buff=0.8)
        self.play(FadeIn(t), LaggedStartMap(FadeIn, pts, shift=RIGHT * 0.2, lag_ratio=0.25))
        self.next_slide(notes="A perceptron is a linear classifier: great for AND/OR, helpless on XOR, because its boundary is always linear. That limitation motivates multilayer networks.")
        self.clearSlide()

        # L10 limitations of the perceptron (big centered no-line plot, no table)
        t = mkTitle("Limitations of the perceptron")
        pts = mkBullets([
            "Only works with linearly separable data",
            "These problems led to a long 'winter' (1980s)",
        ], font_size=24, width_chars=52).next_to(t, DOWN, buff=0.5)
        xor_p = self.cornerPlot({(0, 0): "F", (0, 1): "T", (1, 0): "T", (1, 1): "F"},
                                ORIGIN, fail=True)
        xor_p.scale(1.9).move_to(DOWN * 0.8)
        lab = Text("XOR is not linearly separable", font_size=22, color=MUTED).next_to(xor_p, DOWN, buff=0.3)
        self.play(FadeIn(t), FadeIn(pts))
        self.play(FadeIn(xor_p), FadeIn(lab))
        self.next_slide(notes="XOR is the poster child: not linearly separable, so a perceptron can never get it. This stalled the field for years.")
        self.clearSlide()

        # L11 multilayer networks
        t = mkTitle("Multilayer networks")
        pts = mkBullets([
            "single-layer networks only learn hyperplanes; real problems are more complex",
            "what if we layer neurons?",
            "Multi-Layer Perceptron (MLP): an input layer, one or more hidden layers, an output layer",
            "two layers of learnable weights gives very powerful computation",
        ], font_size=22, width_chars=46).next_to(t, DOWN, buff=0.4).to_edge(LEFT, buff=0.7)
        net, layers, edges = mkNetwork([2, 3, 1])
        net.scale(0.62).to_edge(RIGHT, buff=0.7).shift(DOWN * 0.3)
        ntags = VGroup(
            Text("input", font_size=16, color=MUTED).next_to(layers[0], DOWN, buff=0.25),
            Text("hidden", font_size=16, color=MUTED).next_to(layers[1], DOWN, buff=0.25),
            Text("output", font_size=16, color=MUTED).next_to(layers[2], DOWN, buff=0.25),
        )
        self.play(FadeIn(t), FadeIn(pts))
        self.play(FadeIn(edges), FadeIn(VGroup(*layers)), FadeIn(ntags))
        self.next_slide(notes="Stack neurons into layers -> MLP. Input, hidden(s), output. Two weight layers already buy huge expressive power.")
        self.clearSlide()

        # UAT (moved here, right after multilayer networks)
        t = mkTitle("Universal Approximation Theorem")
        stmt = mkParagraph("Any sufficiently wide two-layer neural network is enough "
                           "to approximate any given function.",
                           font_size=28, width_chars=44)
        box = SurroundingRectangle(stmt, color=AMBER, buff=0.4)
        stmt_g = VGroup(stmt, box).next_to(t, DOWN, buff=0.8)
        pts = mkBullets([
            "caveat: it says such a network exists, not that training finds it, nor how many nodes",
        ], font_size=22, width_chars=64).next_to(stmt_g, DOWN, buff=0.6)
        credit = Text("Cybenko (1989); Hornik, Stinchcombe & White (1989)",
                      font_size=18, color=MUTED).next_to(pts, DOWN, buff=0.5)
        self.play(FadeIn(t), Write(stmt), Create(box))
        self.play(FadeIn(pts), FadeIn(credit))
        self.next_slide(notes="Now that we have multilayer nets: how powerful are they? UAT - a wide enough two-layer net can approximate any function. Caveat: existence, not findability. Proven by Cybenko (1989) for sigmoids and Hornik, Stinchcombe & White (1989) more generally.")
        self.clearSlide()

        # L12 XOR with an MLP (worked example, horizontal / sideways, centered)
        t = mkTitle("XOR with an MLP")
        pts = mkBullets([
            "XOR can't be solved with a single perceptron",
            "with an MLP it can - here's one worked example",
        ], font_size=22, width_chars=64).next_to(t, DOWN, buff=0.35)
        rows = [("1", "1", "-1"), ("1", "0", "1"), ("0", "1", "1"), ("0", "0", "-1")]
        grid, _ = mkTruthTable(["x1", "x2", "z"], rows)
        grid.scale(0.6).to_corner(DL, buff=0.7)
        x1 = Circle(radius=0.3, color=BLUE).move_to([-4.2, 0.9, 0])
        x2 = Circle(radius=0.3, color=BLUE).move_to([-4.2, -0.9, 0])
        x1l = MathTex("x_1", font_size=22).move_to(x1); x2l = MathTex("x_2", font_size=22).move_to(x2)
        hpos = [[-0.6, 1.4, 0], [-0.6, 0.0, 0], [-0.6, -1.4, 0]]
        hn = VGroup(*[Circle(radius=0.34, color=GREEN).move_to(p) for p in hpos])
        hv = VGroup(*[MathTex(v, font_size=20).move_to(n) for n, v in zip(hn, ["0.6", "1.5", "0.6"])])
        o = Circle(radius=0.32, color=AMBER).move_to([3.0, 0.0, 0])
        ov = MathTex("0.5", font_size=20).move_to(o)
        oy = MathTex(r"\hat{y}", font_size=20, color=MUTED).next_to(o, RIGHT, buff=0.3)
        e_in = [(x1, hn[0]), (x1, hn[1]), (x2, hn[1]), (x2, hn[2])]
        edges = VGroup(*[Line(s.get_right(), d.get_left(), buff=0.06, color=MUTED, stroke_width=1.6) for s, d in e_in])
        for hh in hn:
            edges.add(Line(hh.get_right(), o.get_left(), buff=0.06, color=MUTED, stroke_width=1.6))
        wlab = VGroup(
            *[MathTex("1", font_size=15, color=AMBER).move_to(s.get_right() + 0.35 * (d.get_left() - s.get_right()) + UP * 0.12) for s, d in e_in],
            *[MathTex(w, font_size=15, color=AMBER).move_to(hh.get_right() + 0.4 * (o.get_left() - hh.get_right()) + UP * 0.12) for hh, w in zip(hn, ["1", "-2", "1"])],
        )
        tree = VGroup(edges, wlab, x1, x2, x1l, x2l, hn, hv, o, ov, oy).move_to(RIGHT * 0.6 + DOWN * 0.4)
        self.play(FadeIn(t), FadeIn(pts), FadeIn(grid))
        self.play(FadeIn(tree))
        self.next_slide(notes="One worked MLP for XOR (drawn left to right): two inputs feed three hidden nodes (0.6, 1.5, 0.6), combined with weights 1, -2, 1 into output 0.5. The hidden layer makes XOR separable.")
        self.clearSlide()


# scene 4: training + backprop (slides 18-28)

class Day1Training(Day1Base):
    def construct(self):
        # s18 feed-forward worked example (matrix form, no bias)
        t = mkTitle("Feed-forward neural network")
        sizes = VGroup(
            Text("Input size = 3", font_size=20, color=MUTED),
            Text("Hidden layer size = 2", font_size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(t, DOWN, buff=0.1).to_edge(RIGHT, buff=0.6)
        # diagram (right) - simple numbers, weights on the lines
        in_nodes = VGroup(*[Circle(radius=0.3, color=BLUE).move_to([0, y, 0]) for y in (1.8, 0.0, -1.8)])
        in_lbls = VGroup(*[MathTex(v, font_size=24).move_to(n) for n, v in zip(in_nodes, ["1", "1", "0"])])
        h_nodes = VGroup(Circle(radius=0.34, color=GREEN).move_to([2.8, 0.95, 0]),
                         Circle(radius=0.34, color=GREEN).move_to([2.8, -0.95, 0]))
        o_node = Circle(radius=0.3, color=AMBER).move_to([5.4, 0, 0])
        wxh_to = [["0.1", "0.3", "0.5"], ["0.2", "0.4", "0.6"]]
        edges = VGroup(); elabs = VGroup()
        for j, hn in enumerate(h_nodes):
            for i, xn in enumerate(in_nodes):
                edges.add(Line(xn.get_right(), hn.get_left(), buff=0.08, color=MUTED, stroke_width=1.5))
                elabs.add(MathTex(wxh_to[j][i], font_size=16, color=AMBER).move_to(
                    xn.get_right() + 0.32 * (hn.get_left() - xn.get_right()) + UP * 0.16))
        for hn, w in zip(h_nodes, ["1", "1"]):
            edges.add(Line(hn.get_right(), o_node.get_left(), buff=0.08, color=MUTED, stroke_width=1.5))
            elabs.add(MathTex(w, font_size=16, color=AMBER).move_to(
                hn.get_right() + 0.45 * (o_node.get_left() - hn.get_right()) + UP * 0.16))
        tags = VGroup(
            Text("X", font_size=22, weight=BOLD).move_to([0, -2.7, 0]),
            Text("h", font_size=22, weight=BOLD).move_to([2.8, -2.7, 0]),
            Text("O", font_size=22, weight=BOLD).move_to([5.4, -2.7, 0]),
        )
        diagram = VGroup(edges, elabs, in_nodes, in_lbls, h_nodes, o_node, tags)
        diagram.scale(0.8).to_edge(RIGHT, buff=0.4).shift(DOWN * 0.2)
        h_vals = VGroup(
            MathTex("0.4", font_size=18, color=GREEN).next_to(h_nodes[0], DOWN, buff=0.1),
            MathTex("0.6", font_size=18, color=GREEN).next_to(h_nodes[1], DOWN, buff=0.1),
        )
        o_val = MathTex("1.0", font_size=18, color=AMBER).next_to(o_node, UP, buff=0.15)
        o_sig = MathTex(r"\sigma=0.731", font_size=18, color=AMBER).next_to(o_node, DOWN, buff=0.2)
        # matrices (left)
        xrow = VGroup(MathTex("X =", font_size=26), Matrix([["1", "1", "0"]]).scale(0.6)).arrange(RIGHT, buff=0.25)
        wxh_m = Matrix([["0.1", "0.2"], ["0.3", "0.4"], ["0.5", "0.6"]]).scale(0.6)
        wxh_row = VGroup(MathTex("W_{xh} =", font_size=26), wxh_m).arrange(RIGHT, buff=0.25)
        hrow = VGroup(MathTex(r"h = X\,W_{xh} =", font_size=26),
                      Matrix([["0.4", "0.6"]]).scale(0.6)).arrange(RIGHT, buff=0.25)
        who_row = VGroup(MathTex("W_{ho} =", font_size=26),
                         Matrix([["1"], ["1"]]).scale(0.6)).arrange(RIGHT, buff=0.25)
        oeq = MathTex(r"O = h\,W_{ho} = 1.0", font_size=26)
        sigeq = MathTex(r"\sigma(O) = 0.731", font_size=26)
        left = VGroup(xrow, wxh_row, hrow, who_row, oeq, sigeq).arrange(DOWN, aligned_edge=LEFT, buff=0.34)
        fitBox(left, 6.2, 6.2).to_edge(LEFT, buff=0.6).shift(DOWN * 0.1)
        nobias = Text("No bias", font_size=18, color=MUTED).next_to(sigeq, RIGHT, buff=0.5)
        self.play(FadeIn(t), FadeIn(sizes), FadeIn(diagram), FadeIn(xrow))
        self.next_slide(notes="Input X is a row of 3, weights W_xh are 3x2. Each line in the diagram is a weight.")
        self.play(FadeIn(wxh_row))
        self.next_slide(notes="The weight matrix matches the labeled edges in the diagram.")
        self.play(FadeIn(hrow), FadeIn(h_vals),
                  Indicate(VGroup(wxh_m.get_rows()[0], wxh_m.get_rows()[1]), color=AMBER))
        self.next_slide(notes="h = X.W_xh: h0 = 1(0.1)+1(0.3)+0(0.5) = 0.4, h1 = 1(0.2)+1(0.4)+0(0.6) = 0.6.")
        self.play(FadeIn(who_row))
        self.next_slide(notes="Hidden-to-output weights, a 2x1 column.")
        self.play(FadeIn(oeq), FadeIn(o_val), FadeIn(nobias))
        self.next_slide(notes="O = h.W_ho = 0.4(1) + 0.6(1) = 1.0. No bias here.")
        self.play(FadeIn(sigeq), FadeIn(o_sig))
        self.next_slide(notes="Apply sigmoid at the output: sigma(1.0) = 0.731. Activation applied only at the output here.")
        self.clearSlide()

        # s18b same network, now with bias (simple numbers, solved)
        t = mkTitle("Feed-forward neural network (with bias)")
        sizes = VGroup(
            Text("Input size = 3", font_size=20, color=MUTED),
            Text("Hidden layer size = 2", font_size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(t, DOWN, buff=0.1).to_edge(RIGHT, buff=0.6)
        in_nodes = VGroup(*[Circle(radius=0.3, color=BLUE).move_to([0, y, 0]) for y in (1.8, 0.2, -1.4)])
        in_lbls = VGroup(*[MathTex(v, font_size=24).move_to(n) for n, v in zip(in_nodes, ["1", "1", "0"])])
        bias_in = Circle(radius=0.26, color=MUTED).move_to([0, -2.8, 0])
        bias_in_l = MathTex("1", font_size=22).move_to(bias_in)
        h_nodes = VGroup(Circle(radius=0.34, color=GREEN).move_to([2.8, 1.0, 0]),
                         Circle(radius=0.34, color=GREEN).move_to([2.8, -0.8, 0]))
        bias_h = Circle(radius=0.26, color=MUTED).move_to([2.8, -2.4, 0])
        bias_h_l = MathTex("1", font_size=22).move_to(bias_h)
        o_node = Circle(radius=0.3, color=AMBER).move_to([5.4, 0.1, 0])
        wxh_to = [["0.1", "0.3", "0.5"], ["0.2", "0.4", "0.6"]]
        edges = VGroup(); elabs = VGroup()
        for j, hn in enumerate(h_nodes):
            for i, xn in enumerate(in_nodes):
                edges.add(Line(xn.get_right(), hn.get_left(), buff=0.08, color=MUTED, stroke_width=1.4))
                elabs.add(MathTex(wxh_to[j][i], font_size=15, color=AMBER).move_to(
                    xn.get_right() + 0.3 * (hn.get_left() - xn.get_right()) + UP * 0.14))
            edges.add(DashedLine(bias_in.get_top(), hn.get_corner(DL), buff=0.08, color=MUTED, stroke_width=1.4))
        for hn, w in zip(h_nodes, ["1", "1"]):
            edges.add(Line(hn.get_right(), o_node.get_left(), buff=0.08, color=MUTED, stroke_width=1.4))
            elabs.add(MathTex(w, font_size=15, color=AMBER).move_to(
                hn.get_right() + 0.45 * (o_node.get_left() - hn.get_right()) + UP * 0.14))
        edges.add(DashedLine(bias_h.get_right(), o_node.get_corner(DL), buff=0.08, color=MUTED, stroke_width=1.4))
        tags = VGroup(
            Text("X", font_size=22, weight=BOLD).move_to([0, -3.6, 0]),
            Text("h", font_size=22, weight=BOLD).move_to([2.8, -3.2, 0]),
            Text("O", font_size=22, weight=BOLD).move_to([5.4, -3.2, 0]),
        )
        diagram = VGroup(edges, elabs, in_nodes, in_lbls, bias_in, bias_in_l, h_nodes, bias_h, bias_h_l, o_node, tags)
        diagram.scale(0.66).to_edge(RIGHT, buff=0.4).shift(UP * 0.1)
        xrow = VGroup(MathTex("X =", font_size=24), Matrix([["1", "1", "0"]]).scale(0.55)).arrange(RIGHT, buff=0.2)
        wxh_row = VGroup(MathTex("W_{xh} =", font_size=24),
                         Matrix([["0.1", "0.2"], ["0.3", "0.4"], ["0.5", "0.6"]]).scale(0.5)).arrange(RIGHT, buff=0.2)
        hrow = MathTex(r"h = X\,W_{xh} + h_{bias} = [\,0.5,\ \ 0.7\,]\quad(h_{bias}=0.1)", font_size=22)
        who_row = VGroup(MathTex("W_{ho} =", font_size=24),
                         Matrix([["1"], ["1"]]).scale(0.55)).arrange(RIGHT, buff=0.2)
        oeq = MathTex(r"O = h\,W_{ho} + O_{bias} = 1.5\quad(O_{bias}=0.3)", font_size=22)
        sigeq = MathTex(r"\sigma(O) = \sigma(1.5) = 0.818", font_size=24, color=GREEN)
        left = VGroup(xrow, wxh_row, hrow, who_row, oeq, sigeq).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        fitBox(left, 6.8, 6.6).to_edge(LEFT, buff=0.5).shift(DOWN * 0.1)
        self.play(FadeIn(t), FadeIn(sizes), FadeIn(diagram), FadeIn(xrow), FadeIn(wxh_row))
        self.next_slide(notes="Same network plus bias nodes (the '1's). Each bias node feeds every node in the next layer.")
        self.play(FadeIn(hrow))
        self.next_slide(notes="Add the bias to each hidden value: h = [0.4+0.1, 0.6+0.1] = [0.5, 0.7].")
        self.play(FadeIn(who_row), FadeIn(oeq))
        self.next_slide(notes="Output bias too: O = 0.5(1) + 0.7(1) + 0.3 = 1.5.")
        self.play(FadeIn(sigeq))
        self.next_slide(notes="Solve it: sigma(1.5) = 0.818. The '1' nodes just add a constant to each pre-activation.")
        self.clearSlide()

        # s19 activation functions
        t = mkTitle("Activation functions")
        why = mkParagraph(
            "Activation functions introduce non-linearity, which lets us generate "
            "non-linear decision boundaries. We go from linear transformations that "
            "shift the points around, to being able to 'fold' and 'crumple' the "
            "space our points live in. So a non-linear function is inserted after "
            "each layer. It must be differentiable so we can train by gradients.",
            font_size=19, width_chars=48)
        lin = MathTex(r"W_2(W_1\mathbf{x}) = (W_2 W_1)\mathbf{x}", font_size=22, color=RED)
        lin_l = Text("stack linear layers -> still linear", font_size=15, color=MUTED).next_to(lin, DOWN, buff=0.1)
        non = MathTex(r"W_2\,\sigma(W_1\mathbf{x})", font_size=22, color=GREEN)
        non_l = Text("insert sigma -> non-linear", font_size=15, color=MUTED).next_to(non, DOWN, buff=0.1)
        contrast = VGroup(VGroup(lin, lin_l), VGroup(non, non_l)).arrange(RIGHT, buff=0.7)
        img1 = mkImage("linear_boundary.png", max_w=2.7, max_h=1.9)
        img2 = mkImage("folded_boundary.png", max_w=2.7, max_h=1.9)
        imgs = Group(img1, img2).arrange(RIGHT, buff=0.3)
        imgcred = Text("images: MIT Intro to Deep Learning", font_size=13, color=MUTED).next_to(imgs, DOWN, buff=0.12)
        imgblock = Group(imgs, imgcred)
        leftcol = Group(why, contrast, imgblock).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        leftcol.to_edge(LEFT, buff=0.55).shift(UP * 0.1)
        fitBox(leftcol, 7.4, 6.3)
        ax = Axes(x_range=[-4, 4, 2], y_range=[-1.2, 1.6, 1], x_length=4.2, y_length=2.7,
                  tips=False, axis_config={"stroke_color": MUTED, "stroke_width": 1.5}).to_edge(RIGHT, buff=0.5).shift(UP * 0.7)
        sig = ax.plot(lambda x: 1 / (1 + np.exp(-x)), color=BLUE)
        tanh = ax.plot(lambda x: np.tanh(x), color=AMBER)
        relu = ax.plot(lambda x: max(0, x), x_range=[-4, 1.6], color=GREEN)
        legend = VGroup(
            MathTex(r"\sigma(z)=\frac{1}{1+e^{-z}}", font_size=20, color=BLUE),
            MathTex(r"\tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}", font_size=20, color=AMBER),
            MathTex(r"\text{ReLU}(z)=\max(0,z)", font_size=20, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).next_to(ax, DOWN, buff=0.3)
        self.play(FadeIn(t), FadeIn(why))
        self.play(FadeIn(contrast))
        self.play(Create(ax), Create(sig), Create(tanh), Create(relu), FadeIn(legend))
        self.play(FadeIn(imgblock))
        self.next_slide(notes="Non-linearity lets us bend boundaries: linear = shift the space, non-linear = fold/crumple it. W2(W1x)=(W2W1)x is STILL linear (no activation); W2.sigma(W1x) is non-linear. Bottom-left: linear vs folded boundary (MIT Intro to Deep Learning). Must be differentiable for gradient training.")
        self.clearSlide()

        # s20 saturation + vanishing gradients (abridged Jurafsky 6.1)
        t = mkTitle("Saturation and vanishing gradients")
        why = mkParagraph(
            "In the sigmoid or tanh functions, very high values of z give outputs "
            "saturated extremely close to 1, with derivatives very close to 0. "
            "Because we train by propagating an error signal backwards, multiplying "
            "the gradients from each layer, gradients that are almost 0 make the "
            "error signal shrink until it is too small to train with - the vanishing "
            "gradient problem. Rectifiers (ReLU) don't have this problem, since the "
            "derivative of ReLU for high values of z is 1 rather than very close to 0.",
            font_size=24, width_chars=60).next_to(t, DOWN, buff=0.7)
        fitBox(why, 12.0, 4.6)
        cite = Text("abridged from Jurafsky & Martin, SLP3, section 6.1",
                    font_size=18, color=MUTED).next_to(why, DOWN, buff=0.5)
        self.play(FadeIn(t), FadeIn(why))
        self.play(FadeIn(cite))
        self.next_slide(notes="This is WHY ReLU won. Saturated derivatives (~0) multiply across layers and the error signal vanishes. ReLU's derivative is 1 for high z, so the signal survives depth. Source: SLP3 6.1.")
        self.clearSlide()

        # s21 the loss
        t = mkTitle("Loss Functions for Neural Networks")
        soft = MathTex(r"p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}", font_size=34)
        ce = MathTex(r"L = -\sum_i y_i \log p_i = -\log p_{\text{correct}}", font_size=34)
        mse = MathTex(r"\text{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2", font_size=30, color=MUTED)
        eqs = VGroup(soft, ce, mse).arrange(DOWN, buff=0.45).next_to(t, DOWN, buff=0.45)
        pts = mkBullets([
            "scores -> softmax -> probabilities that sum to 1",
            "cross-entropy = negative log prob of the correct class (punishes confident wrongness)",
            "regression uses MSE instead",
        ], font_size=22, width_chars=68).next_to(eqs, DOWN, buff=0.4)
        self.play(FadeIn(t), Write(soft), Write(ce), FadeIn(mse))
        self.play(FadeIn(pts))
        self.next_slide(notes="Softmax -> probs. CE = -log p_correct. Note to self: language modeling = classification over the vocabulary -> CE carries into NLP.")
        self.clearSlide()

        # s22 gradient descent
        t = mkTitle("Gradient descent")
        upd = MathTex(r"\theta \leftarrow \theta - \eta\, \nabla_\theta L", font_size=36).next_to(t, DOWN, buff=0.45).to_edge(LEFT, buff=1.0)
        pts = mkBullets([
            "gradient points uphill; step the opposite way",
            "eta = learning rate (step size)",
            "batch GD: use all n rows for each step",
            "mini-batch: use k of the n rows per step",
            "stochastic (SGD): one random row per step",
            "epoch = one full pass over the data",
        ], font_size=19, width_chars=34).next_to(upd, DOWN, buff=0.45).to_edge(LEFT, buff=0.9)
        fitBox(pts, 6.4, 3.9)
        ax = Axes(x_range=[-3, 3, 1], y_range=[0, 5, 1], x_length=4.6, y_length=3.0,
                  tips=False, axis_config={"stroke_color": MUTED, "stroke_width": 1.5}).to_edge(RIGHT, buff=0.7).shift(DOWN * 0.3)
        curve = ax.plot(lambda x: 0.5 * x ** 2, color=BLUE)
        ball = Dot(ax.c2p(2.6, 0.5 * 2.6 ** 2), color=AMBER, radius=0.12)
        self.play(FadeIn(t), Write(upd), FadeIn(pts), Create(ax), Create(curve), FadeIn(ball))
        self.next_slide(notes="Gradient uphill; step opposite. In practice GD is implemented as mini-batch (k of n rows) or stochastic (one random row per update); full-batch uses all n. Vocab: epoch = one full pass; batch = the rows in a step.")
        for xv in (1.7, 1.0, 0.5, 0.15):
            self.play(ball.animate.move_to(ax.c2p(xv, 0.5 * xv ** 2)), run_time=0.45)
        self.next_slide()
        self.clearSlide()

        # === Two-layer backpropagation: objective (Blumenstock example_2) ===
        t = mkTitle("Two-layer backpropagation")
        pts = mkBullets([
            "backprop = gradient descent + the chain rule",
            "for two layers the solution is not too complex",
        ], font_size=23, width_chars=42).next_to(t, DOWN, buff=0.35).to_edge(LEFT, buff=0.7)
        obj = MathTex(r"\min_{W,v}\ \sum_n \tfrac12\Big(y_n - \sum_i v_i\,f(\mathbf{w}_i\cdot\mathbf{x}_n)\Big)^2",
                      font_size=30).next_to(pts, DOWN, buff=0.5).to_edge(LEFT, buff=0.9)
        pred = Text("the inner sum is the prediction y-hat", font_size=18, color=MUTED).next_to(obj, DOWN, buff=0.25).to_edge(LEFT, buff=0.9)
        defs = mkBullets([
            "n indexes observations",
            "i indexes hidden units",
            "v are the second-layer weights",
            "f is the sigmoid function",
            "w_i are the weights feeding node i",
        ], font_size=18, width_chars=30).next_to(pred, DOWN, buff=0.4).to_edge(LEFT, buff=0.9)
        in1 = Circle(radius=0.26, color=BLUE).move_to([1.4, 1.0, 0])
        in2 = Circle(radius=0.26, color=BLUE).move_to([1.4, -0.6, 0])
        z1 = Circle(radius=0.28, color=GREEN).move_to([3.6, 0.9, 0])
        z2 = Circle(radius=0.28, color=GREEN).move_to([3.6, -0.7, 0])
        oo = Circle(radius=0.26, color=AMBER).move_to([5.6, 0.1, 0])
        z1l = MathTex("z_1", font_size=18).move_to(z1); z2l = MathTex("z_2", font_size=18).move_to(z2)
        oyl = MathTex(r"\hat{y}", font_size=18).next_to(oo, RIGHT, buff=0.2)
        nedges = VGroup(*[Line(s.get_right(), d.get_left(), buff=0.05, color=MUTED, stroke_width=1.4)
                          for s in (in1, in2) for d in (z1, z2)])
        nedges.add(Line(z1.get_right(), oo.get_left(), buff=0.05, color=MUTED, stroke_width=1.4))
        nedges.add(Line(z2.get_right(), oo.get_left(), buff=0.05, color=MUTED, stroke_width=1.4))
        zdef = MathTex(r"z_i = f(\mathbf{w}_i\cdot\mathbf{x})", font_size=20, color=MUTED).next_to(VGroup(in1, in2, oo), DOWN, buff=0.4)
        diagram = VGroup(nedges, in1, in2, z1, z2, oo, z1l, z2l, oyl, zdef).to_edge(RIGHT, buff=0.8).shift(UP * 0.6)
        self.play(FadeIn(t), FadeIn(pts), FadeIn(diagram))
        self.play(Write(obj), FadeIn(pred))
        self.play(FadeIn(defs))
        self.next_slide(notes="The objective: minimize squared error over all observations. The inner sum (second-layer weights v times sigmoid of first-layer dot products) IS the prediction. n = observations, i = hidden units, v = output weights, f = sigmoid, w_i = weights into hidden node i.")
        self.clearSlide()

        # === Two-layer backprop: the gradient ===
        t = mkTitle("Two-layer backpropagation: the gradient")
        loss = MathTex(r"\mathcal{L}(W) = \tfrac12\Big(y - \underbrace{\textstyle\sum_i v_i f(\mathbf{w}_i\cdot\mathbf{x})}_{\hat{y}}\Big)^2",
                       font_size=26).next_to(t, DOWN, buff=0.3)
        edef = MathTex(r"\frac{\partial \mathcal{L}}{\partial \hat{y}} = -(y-\hat{y}) = -e \qquad (e \equiv y-\hat{y})",
                       font_size=24, color=RED).next_to(loss, DOWN, buff=0.3)
        chain = VGroup(
            MathTex(r"\frac{\partial \mathcal{L}}{\partial \mathbf{w}_i} = \frac{\partial \mathcal{L}}{\partial f_i}\cdot\frac{\partial f_i}{\partial \mathbf{w}_i}", font_size=25),
            MathTex(r"\frac{\partial \mathcal{L}}{\partial f_i} = -(y-\hat{y})\, v_i = -e\,v_i", font_size=24),
            MathTex(r"\frac{\partial f_i}{\partial \mathbf{w}_i} = f'(\mathbf{w}_i\cdot\mathbf{x})\,\mathbf{x}", font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(edef, DOWN, buff=0.4)
        result = MathTex(r"\nabla_{\mathbf{w}_i} = -\,e\,v_i\,f'(\mathbf{w}_i\cdot\mathbf{x})\,\mathbf{x}",
                         font_size=30, color=AMBER).next_to(chain, DOWN, buff=0.4)
        self.play(FadeIn(t), Write(loss))
        self.play(FadeIn(edef))
        self.play(LaggedStartMap(FadeIn, chain, lag_ratio=0.3))
        self.play(Write(result))
        self.next_slide(notes="Where e comes from: the loss is squared error 1/2 (y - yhat)^2, and yhat is the inner sum. Differentiating the loss wrt yhat gives -(y - yhat), which we name -e. So e = y - yhat is just the residual the loss hands us. The chain rule then carries that e through: dL/df_i = -e v_i, df_i/dw_i = f'(w_i.x) x, and multiplying gives the gradient.")
        self.clearSlide()

        # === Does the gradient make sense? (sanity checks) ===
        t = mkTitle("Does this gradient make sense?")
        recall = MathTex(r"e \equiv y - \hat{y}\ \ \text{(the prediction error from the loss)}",
                         font_size=22, color=RED).next_to(t, DOWN, buff=0.3)
        result = MathTex(r"\nabla_{\mathbf{w}_i} = -\,e\,v_i\,f'(\mathbf{w}_i\cdot\mathbf{x})\,\mathbf{x}",
                         font_size=32, color=AMBER).next_to(recall, DOWN, buff=0.4)
        pts = mkBullets([
            "if the prediction error e is small, take small steps",
            "if v_i is small, hidden unit i barely affects the output, so the gradient is small",
            "if e or v_i flips sign, the gradient flips sign too",
        ], font_size=24, width_chars=56).next_to(result, DOWN, buff=0.6)
        self.play(FadeIn(t), FadeIn(recall), Write(result))
        self.play(LaggedStartMap(FadeIn, pts, shift=RIGHT * 0.2, lag_ratio=0.3))
        self.next_slide(notes="Recall e = y - yhat (the residual from the loss). Sanity-check the formula: small error -> small step; a hidden unit with little influence (small v_i) gets a small gradient; sign of the update follows the sign of e and v_i. The math matches the intuition.")
        self.clearSlide()

        # === Concrete backprop (1/2): nested function + where the deltas come from ===
        t = mkTitle("A concrete example: deriving the deltas")
        snet, _ = mkScalarNet()
        snet.scale(0.5).next_to(t, DOWN, buff=0.3)
        fwd = MathTex(r"z_1=w_1x+b_1,\ a_1=\sigma(z_1),\ z_2=w_2a_1+b_2,\ a_2=\sigma(z_2),\ L=\tfrac12(a_2-y)^2",
                      font_size=20).next_to(snet, DOWN, buff=0.3)
        nest = MathTex(r"L = \tfrac12\big(\sigma(w_2\,\sigma(w_1 x + b_1) + b_2) - y\big)^2",
                       font_size=22).next_to(fwd, DOWN, buff=0.3)
        self.play(FadeIn(t), FadeIn(snet), Write(fwd))
        self.next_slide(notes="Tiny scalar net = one nested function from x to L. To train we need dL/dw for each weight.")
        self.play(Write(nest))
        self.next_slide(notes="The whole network is one function. Now differentiate it - and watch the deltas appear.")
        # the derivation, on the same page, right alongside the formulas
        intro = MathTex(r"\delta \equiv \frac{\partial L}{\partial z}\quad\text{(a reused chunk of the chain rule)}",
                        font_size=20, color=MUTED)
        d_w2 = MathTex(r"\frac{\partial L}{\partial w_2} = \underbrace{(a_2-y)\,\sigma'(z_2)}_{\delta_2}\cdot a_1", font_size=24)
        d_w1a = MathTex(r"\frac{\partial L}{\partial w_1} = \underbrace{(a_2-y)\,\sigma'(z_2)}_{\delta_2}\cdot w_2\,\sigma'(z_1)\cdot x", font_size=24)
        d_w1b = MathTex(r"= \underbrace{\delta_2\, w_2\,\sigma'(z_1)}_{\delta_1}\cdot x", font_size=24)
        compact = VGroup(
            MathTex(r"\delta_2=(a_2-y)\sigma'(z_2),\quad \delta_1=\delta_2 w_2\sigma'(z_1)", font_size=22, color=RED),
            MathTex(r"\frac{\partial L}{\partial w_2}=\delta_2 a_1,\quad \frac{\partial L}{\partial w_1}=\delta_1 x", font_size=22, color=RED),
        ).arrange(DOWN, buff=0.2)
        deriv = VGroup(intro, d_w2, d_w1a, d_w1b, compact).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        deriv.next_to(nest, DOWN, buff=0.35)
        fitBox(deriv, 12.5, 3.7)
        self.play(FadeOut(nest), FadeIn(intro))
        self.next_slide(notes="A delta is just a name for dL/dz - a chunk of the chain rule we are about to watch appear.")
        self.play(Write(d_w2))
        self.next_slide(notes="w2 reaches L through z2->a2->L: dL/dw2 = (a2-y)sigma'(z2) * a1. Bundle the first two factors = delta_2.")
        self.play(Write(d_w1a), Write(d_w1b))
        self.next_slide(notes="w1 is deeper; its chain reuses the SAME delta_2, then bundles one more chunk into delta_1. Deltas are reused chain-rule pieces.")
        self.play(FadeIn(compact))
        self.next_slide(notes="Rules: every weight gradient = (delta of the node it feeds) x (its input); delta_1 = delta_2 pushed back a step. That reuse IS backprop. Next: plug in numbers.")
        self.clearSlide()

        # === Concrete backprop (2/2): solve it on the graph ===
        t = mkTitle("A concrete example: solve it on the graph")
        setup = MathTex(r"x=1,\ y=0,\ w_1=w_2=0.5,\ b_1=b_2=0", font_size=18, color=MUTED).next_to(t, DOWN, buff=0.15)
        ref = VGroup(
            MathTex(r"\delta_2 = (a_2-y)\,\sigma'(z_2)", font_size=18, color=RED),
            MathTex(r"\delta_1 = \delta_2\, w_2\,\sigma'(z_1)", font_size=18, color=RED),
            MathTex(r"\tfrac{\partial L}{\partial w_2} = \delta_2\, a_1", font_size=18, color=RED),
            MathTex(r"\tfrac{\partial L}{\partial w_1} = \delta_1\, x", font_size=18, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(LEFT, buff=0.5).shift(DOWN * 0.3)
        reflbl = Text("the formulas", font_size=15, color=MUTED).next_to(ref, UP, buff=0.25)
        vx = Circle(radius=0.28, color=BLUE).move_to([2.2, 1.9, 0])
        va1 = Circle(radius=0.28, color=GREEN).move_to([2.2, 0.3, 0])
        va2 = Circle(radius=0.28, color=AMBER).move_to([2.2, -1.3, 0])
        vL = Circle(radius=0.28, color=RED).move_to([2.2, -2.9, 0])
        vxl = MathTex("x", font_size=20).move_to(vx); va1l = MathTex("a_1", font_size=18).move_to(va1)
        va2l = MathTex("a_2", font_size=18).move_to(va2); vLl = MathTex("L", font_size=18).move_to(vL)
        ve1 = Line(vx.get_bottom(), va1.get_top(), buff=0.05, color=MUTED)
        ve2 = Line(va1.get_bottom(), va2.get_top(), buff=0.05, color=MUTED)
        ve3 = Line(va2.get_bottom(), vL.get_top(), buff=0.05, color=MUTED)
        vtree = VGroup(vx, va1, va2, vL, vxl, va1l, va2l, vLl, ve1, ve2, ve3)
        f_x = MathTex(r"x=1", font_size=16, color=GREEN).next_to(vx, LEFT, buff=0.3)
        f_a1 = MathTex(r"z_1=0.5,\ a_1=0.6225", font_size=16, color=GREEN).next_to(va1, LEFT, buff=0.3)
        f_a2 = MathTex(r"z_2=0.3113,\ a_2=0.5772", font_size=16, color=GREEN).next_to(va2, LEFT, buff=0.3)
        f_L = MathTex(r"L=0.1666", font_size=16, color=GREEN).next_to(vL, LEFT, buff=0.3)
        b_d2 = MathTex(r"\delta_2=0.1409", font_size=16, color=RED).next_to(va2, RIGHT, buff=0.3)
        b_gw2 = MathTex(r"\partial L/\partial w_2=0.0877", font_size=16, color=RED).next_to(ve2, RIGHT, buff=0.3)
        b_d1 = MathTex(r"\delta_1=0.01655", font_size=16, color=RED).next_to(va1, RIGHT, buff=0.3)
        b_gw1 = MathTex(r"\partial L/\partial w_1=0.01655", font_size=16, color=RED).next_to(ve1, RIGHT, buff=0.3)
        self.play(FadeIn(t), FadeIn(setup), FadeIn(reflbl), FadeIn(ref), FadeIn(vtree))
        self.next_slide(notes="The formulas from the last slide sit on the left for reference; the graph is a vertical tree. Now the forward pass, top to bottom, on the LEFT (green).")
        self.play(LaggedStartMap(FadeIn, VGroup(f_x, f_a1, f_a2, f_L), lag_ratio=0.4))
        self.next_slide(notes="Now backprop, bottom to top, on the RIGHT (red). Each red number is a left-hand formula evaluated: delta_2 = (a2-y)sigma'(z2) = 0.1409, dL/dw2 = delta_2 * a1 = 0.0877, etc.")
        self.play(LaggedStartMap(FadeIn, VGroup(b_d2, b_gw2, b_d1, b_gw1), lag_ratio=0.4))
        self.next_slide(notes="Left = the formulas; tree-left = forward numbers; tree-right = the same formulas solved. The deltas (red, at nodes) attribute error; the dL/dw terms are the gradients you apply.")
        self.clearSlide()

        # === Generalizing backprop: three steps ===
        t = mkTitle("Backpropagation: the general recipe")
        q = Text("How do we generalize beyond two layers?", font_size=24, color=MUTED).next_to(t, DOWN, buff=0.5)
        steps = VGroup(
            Text("1.  Forward propagation  ->  outputs", font_size=26, color=INK),
            Text("2.  Backward propagation  ->  generate 'deltas'", font_size=26, color=INK),
            Text("3.  Weight update  ->  same as gradient descent", font_size=26, color=INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).next_to(q, DOWN, buff=0.7)
        self.play(FadeIn(t), FadeIn(q))
        self.play(LaggedStartMap(FadeIn, steps, shift=RIGHT * 0.2, lag_ratio=0.3))
        self.next_slide(notes="The recipe scales to any depth: (1) forward to get outputs, (2) backward to generate a delta at every node, (3) update each weight with its delta - exactly the gradient-descent step.")
        self.clearSlide()

        # === Intuition: forward then backward propagation ===
        t = mkTitle("Intuition: forward, then backward")
        net2, layers2, edges2 = mkNetwork([3, 4, 2])
        net2.scale(0.6).shift(DOWN * 0.3)
        fwd_lbl = Text("Forward propagation of input  ->", font_size=20, color=GREEN).next_to(net2, UP, buff=0.3)
        bwd_lbl = Text("<-  Backward propagation of error", font_size=20, color=RED).next_to(net2, DOWN, buff=0.3)
        pts = mkBullets([
            "Forward: apply the sigmoid to dot products, layer by layer, to get outputs and residuals",
            "Backward: send the error back; each node gets an 'error' term, a delta",
        ], font_size=20, width_chars=70).next_to(t, DOWN, buff=0.3)
        self.play(FadeIn(t), FadeIn(pts))
        self.play(FadeIn(edges2), FadeIn(VGroup(*layers2)), FadeIn(fwd_lbl))
        self.next_slide(notes="Forward: inputs flow right, sigmoid of dot products at each node, producing outputs and residuals (errors at the output).")
        self.play(FadeIn(bwd_lbl))
        self.next_slide(notes="Backward: the error flows left; each node accumulates a delta - its share of the blame for the loss.")
        self.clearSlide()

        # === Intuition: the deltas and the update ===
        t = mkTitle("Intuition: deltas and the update")
        cost = MathTex(r"\text{Cost}(i) = Y\log \hat{Y}_i + (1-Y_i)\log(1-\hat{Y}_i)", font_size=24).next_to(t, DOWN, buff=0.4)
        out_delta = VGroup(
            Text("Output-layer delta (how wrong this node is):", font_size=20, color=MUTED),
            MathTex(r"\delta_{jK} = Y_{jK}(1 - Y_{jK})(\hat{Y}_{jK} - Y_{jK})", font_size=26),
        ).arrange(DOWN, buff=0.2).next_to(cost, DOWN, buff=0.45)
        hid_delta = VGroup(
            Text("Hidden delta = downstream delta times the connecting weight:", font_size=20, color=MUTED),
            MathTex(r"\delta_5 = w_{56}\,\delta", font_size=26, color=RED),
        ).arrange(DOWN, buff=0.2).next_to(out_delta, DOWN, buff=0.45)
        upd = MathTex(r"\text{update each weight: } \ w \mathrel{+}= \eta\,\delta_{jk}\,x_i",
                      font_size=26, color=AMBER).next_to(hid_delta, DOWN, buff=0.45)
        self.play(FadeIn(t), Write(cost))
        self.play(FadeIn(out_delta))
        self.play(FadeIn(hid_delta))
        self.play(Write(upd))
        self.next_slide(notes="Cost measures how close the output is to the truth. The output delta is Y(1-Y)(Yhat-Y). A hidden node's delta is the downstream delta times the connecting weight (delta_5 = w56 * delta). Then update every weight by eta * delta * input - gradient descent. Deriving these is mostly chain rule + gradient descent (ESL ch.11).")
        self.clearSlide()


# scene 5: pytorch (slides 28-34)

class Day1PyTorch(Day1Base):
    def codeBlock(self, lines, font_size=26):
        rows = VGroup(*[Text(ln, font_size=font_size, color=BLUE) for ln in lines])
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        return rows

    def construct(self):
        # s28 why a framework
        t = mkTitle("Why a framework?")
        bullets = mkBullets([
            "composability: stack layers like Lego, without re-deriving anything",
            "build on others' work: load and fine-tune pretrained networks (preview of Days 3-4)",
            "speed: highly optimized GPU implementations, far faster than our NumPy",
            "autodiff: gradients computed automatically, for any architecture",
        ], font_size=24, width_chars=64).next_to(t, DOWN, buff=0.6)
        fitBox(bullets, 12.5, 5.2)
        self.play(FadeIn(t), LaggedStartMap(FadeIn, bullets, shift=RIGHT * 0.25, lag_ratio=0.25))
        self.next_slide(notes="Callback to NB2's hand-written backward. Composability + pretrained + speed + autodiff. Our NumPy version was for understanding.")
        self.clearSlide()

        # s29 pytorch building blocks
        t = mkTitle("PyTorch building blocks")
        bullets = mkBullets([
            "Tensor: an n-d array (like NumPy) that can live on a GPU",
            "autograd: records operations; .backward() fills each tensor's .grad",
            "nn.Module: define layers in __init__, the data flow in forward(); parameters tracked automatically",
            "torch.optim: reads .grad and updates weights (SGD, Adam)",
        ], font_size=24, width_chars=64).next_to(t, DOWN, buff=0.6)
        fitBox(bullets, 12.5, 5.2)
        self.play(FadeIn(t), FadeIn(bullets))
        self.next_slide(notes="Map each to NB2: tensor=arrays, autograd=our backward, Module=our layer fns, optim=our update. requires_grad is the tracking switch.")
        self.clearSlide()

        # s30 training loop
        t = mkTitle("The training loop")
        code = self.codeBlock([
            "y_hat = model(x)",
            "loss = criterion(y_hat, y)",
            "optimizer.zero_grad()",
            "loss.backward()",
            "optimizer.step()",
        ]).next_to(t, DOWN, buff=0.6)
        pts = mkBullets([
            "zero_grad: gradients accumulate by default - clear them each step (common bug)",
            "this loop returns on Day 3 (CNN) and Day 4 (fine-tuning)",
        ], font_size=22, width_chars=62).next_to(code, DOWN, buff=0.5)
        self.play(FadeIn(t), LaggedStartMap(FadeIn, code, shift=RIGHT * 0.2, lag_ratio=0.25))
        self.play(FadeIn(pts))
        self.next_slide(notes="Walk the five lines slowly - the heart of every model this week. zero_grad gotcha. Memorize the shape of this loop.")
        self.clearSlide()

        # s31 NOTEBOOK DEMO: PyTorch MLP (the only demo notebook)
        badge = mkNotebookHeader().scale(1.4).move_to(ORIGIN)
        sub = Text("PyTorch MLP on Fashion-MNIST", font_size=26, color=MUTED).next_to(badge, DOWN, buff=0.45)
        self.play(FadeIn(badge), FadeIn(sub))
        self.next_slide(notes="Our hands-on demo: build and train an MLP in PyTorch on real images - the 5-line training loop in action. notebooks/01_pytorch_mlp_fashionmnist.ipynb. (Backprop-from-scratch is the lab assignment, so we don't spoil it here.)")
        self.clearSlide()

        # === Neural networks: issues (1) - init + overfitting ===
        t = mkTitle("Neural networks: issues")
        pts = mkBullets([
            "Non-convex and sensitive to initialization",
            "Fix: small random / uniform initial weights; train several networks",
            "Avoiding overfitting:",
        ], font_size=23, width_chars=44).next_to(t, DOWN, buff=0.4).to_edge(LEFT, buff=0.7)
        sub = mkBullets([
            "Early stopping",
            "Fewer layers / weights per layer",
            "Penalize large weights (L1 / L2 regularization)",
            "Dropout: randomly drop neurons during training",
        ], font_size=21, width_chars=40).next_to(pts, DOWN, buff=0.3).to_edge(LEFT, buff=1.2)
        ax = Axes(x_range=[0, 10, 5], y_range=[0, 1, 0.5], x_length=4.0, y_length=2.6, tips=False,
                  axis_config={"stroke_color": MUTED, "stroke_width": 1.3}).to_edge(RIGHT, buff=0.7).shift(UP * 0.5)
        train = ax.plot(lambda x: 1 - np.exp(-x / 2.5), x_range=[0, 10], color=RED)
        test = ax.plot(lambda x: 0.85 * np.exp(-((x - 4) ** 2) / 22), x_range=[0, 10], color=BLUE)
        leg = VGroup(Text("train accuracy", font_size=15, color=RED),
                     Text("test accuracy (overfits)", font_size=15, color=BLUE)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(ax, DOWN, buff=0.25)
        self.play(FadeIn(t), FadeIn(pts), FadeIn(sub))
        self.play(Create(ax), Create(train), Create(test), FadeIn(leg))
        self.next_slide(notes="Training is non-convex, so initialization matters (random small weights; train several nets). Overfitting fixes: early stopping, smaller nets, L1/L2, dropout. The curve: train accuracy keeps rising while test accuracy peaks then falls - stop at the peak.")
        self.clearSlide()

        # === Neural networks: issues (2) - gradients, LR, compute ===
        t = mkTitle("Neural networks: issues (continued)")
        pts = mkBullets([
            "Vanishing and exploding gradients: gradients get too small or too large in backprop; fix with ReLU, batch normalization",
            "Choosing the learning rate: too high and it won't converge, too low and training is slow; use schedules or adaptive optimizers (Adam, RMSProp)",
            "Computational cost: large networks need lots of compute; use GPU / TPU, parallelization, gradient checkpointing, SGD / mini-batches",
        ], font_size=22, width_chars=66).next_to(t, DOWN, buff=0.6)
        fitBox(pts, 12.5, 5.2)
        self.play(FadeIn(t), LaggedStartMap(FadeIn, pts, shift=RIGHT * 0.2, lag_ratio=0.25))
        self.next_slide(notes="Three big practical issues: gradients (vanish/explode -> ReLU, batchnorm), learning rate (schedules, adaptive optimizers), and compute (GPU/TPU, mini-batches, checkpointing).")
        self.clearSlide()

        # === Tuning networks: not easy ===
        t = mkTitle("Tuning networks: not easy to get right")
        pts = mkBullets([
            "Learning rate - tune it, or use adaptive optimizers",
            "How to initialize - randomize weights, train several networks",
            "Regularization - weight decay, dropout",
            "Batch size - small batches are noisy, large batches are slow",
            "How many layers - generalization vs. overfitting",
            "How many units per layer - generalization vs. overfitting",
            "When to stop?",
        ], font_size=22, width_chars=58).next_to(t, DOWN, buff=0.5)
        fitBox(pts, 12.5, 5.4)
        self.play(FadeIn(t), LaggedStartMap(FadeIn, pts, shift=RIGHT * 0.2, lag_ratio=0.18))
        self.next_slide(notes="Lots of knobs, and they interact. There is no single recipe - this motivates automated search.")
        self.clearSlide()

        # === Tuning networks: semi-automated methods ===
        t = mkTitle("Tuning networks: semi-automated methods")
        pts = mkBullets([
            "Grid search: finds the best within a grid, but very expensive",
            "Random search (a subset of the grid): often nearly as good and much faster, but can miss the best",
            "Bayesian optimization: uses a probabilistic model to predict which hyperparameters will help, and focuses there - best when each model is costly to train",
        ], font_size=22, width_chars=64).next_to(t, DOWN, buff=0.6)
        fitBox(pts, 12.5, 5.2)
        self.play(FadeIn(t), LaggedStartMap(FadeIn, pts, shift=RIGHT * 0.2, lag_ratio=0.25))
        self.next_slide(notes="Grid search = exhaustive but expensive; random search = cheaper, usually nearly as good; Bayesian optimization = model which settings help and explore smartly, great when each run is costly.")
        self.clearSlide()

        # === Tuning networks: other methods (big table = screenshot) ===
        t = mkTitle("Tuning networks: other methods")
        sub = Text("There are other (less common) methods too!", font_size=22, color=MUTED).next_to(t, DOWN, buff=0.3)
        img = mkImage("tuning_table.png", max_w=11.8, max_h=4.6).next_to(sub, DOWN, buff=0.35)
        credit = Text("From Blumenstock's Applied Machine Learning at Berkeley",
                      font_size=16, color=MUTED).next_to(img, DOWN, buff=0.2)
        self.play(FadeIn(t), FadeIn(sub), FadeIn(img), FadeIn(credit))
        self.next_slide(notes="The methods table (Grid, Random, Bayesian, Genetic Algorithms, Hyperband, Population-Based Training, RL-based, Neural Architecture Search). Credit Blumenstock.")
        self.clearSlide()

        # === Tuning networks: modern tools ===
        t = mkTitle("Tuning networks: modern tools")
        pts = mkBullets([
            "Bayesian optimization: Scikit-Optimize, Ax (Meta), Spearmint, GPyOpt",
            "Hyperparameter optimization (friendlier): Optuna, Hyperopt, Ray Tune, Keras Tuner, GridSearchCV",
            "AutoML: Auto-sklearn, TPOT, FLAML, AutoKeras, Google AutoML, H2O AutoML",
            "Experiment tracking: Weights & Biases, TensorBoard, MLflow, Neptune.ai, Comet",
        ], font_size=21, width_chars=66).next_to(t, DOWN, buff=0.6)
        fitBox(pts, 12.5, 5.2)
        self.play(FadeIn(t), LaggedStartMap(FadeIn, pts, shift=RIGHT * 0.2, lag_ratio=0.22))
        self.next_slide(notes="You rarely tune by hand today - name-drop the tool families so they know what to reach for: Bayesian-opt libraries, friendlier HPO tools, AutoML, and experiment trackers.")
        self.clearSlide()

        # === Key takeaways ===
        t = mkTitle("Key takeaways")
        pts = mkBullets([
            "Perceptrons cannot represent nonlinear decision boundaries",
            "Multilayer networks can represent complex functions",
            "Learning requires differentiable, nonlinear activations",
            "Backpropagation computes gradients using the chain rule",
            "Training networks is subtle",
            "Final networks can be very difficult to interpret",
        ], font_size=24, width_chars=58).next_to(t, DOWN, buff=0.6)
        fitBox(pts, 12.5, 5.4)
        self.play(FadeIn(t), LaggedStartMap(FadeIn, pts, shift=RIGHT * 0.2, lag_ratio=0.2))
        self.next_slide(notes="The six things to remember from Day 1. This mirrors the Blumenstock summary; use it to recap before the bridge to NLP.")
        self.clearSlide()

        # s33 wrap + bridge
        t = mkTitle("Wrap-up")
        chain = mkParagraph("node -> layer -> loss -> gradient descent -> backprop -> PyTorch loop",
                            font_size=24, width_chars=52).next_to(t, DOWN, buff=0.7)
        bridge = mkParagraph(
            "Tomorrow, we will focus on some ways text has been dealt with "
            "computationally, and how text is dealt with in the context of neural "
            "nets.", font_size=26, width_chars=50).next_to(chain, DOWN, buff=0.7)
        self.play(FadeIn(t), FadeIn(chain))
        self.play(FadeIn(bridge))
        self.next_slide(notes="Retrace the day in one breath. Pose the open question: text isn't numbers - how do we feed it to a net? That's tomorrow.")
        self.clearSlide()

        # s34 lab time - setup commands
        t = mkTitle("Lab time")
        repo = Text("github.com/cillustrisimo/ydl_2026   (setup/ and notebooks/)",
                    font_size=22, color=MUTED).next_to(t, DOWN, buff=0.4)
        mac_h = Text("macOS / Linux  (Terminal)", font_size=24, color=AMBER, weight=BOLD)
        mac_cmd = Text("curl -fsSL https://raw.githubusercontent.com/cillustrisimo/ydl_2026/main/setup/setup_lab.sh | bash",
                       font_size=18, color=BLUE)
        win_h = Text("Windows  (Anaconda PowerShell Prompt)", font_size=24, color=AMBER, weight=BOLD)
        win_cmd = Text("iwr -useb https://raw.githubusercontent.com/cillustrisimo/ydl_2026/main/setup/setup_lab.ps1 | iex",
                       font_size=18, color=BLUE)
        for c in (mac_cmd, win_cmd):
            fitWidth(c, 12.5)
        block = VGroup(mac_h, mac_cmd, win_h, win_cmd).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        block.next_to(repo, DOWN, buff=0.7)
        foot = Text("then: conda activate nn-lab  ->  jupyter lab",
                    font_size=20, color=MUTED).next_to(block, DOWN, buff=0.6)
        self.play(FadeIn(t), FadeIn(repo))
        self.play(FadeIn(block))
        self.play(FadeIn(foot))
        self.next_slide(notes="Run the one command for your OS - it installs everything (incl. Graphviz/dot) into a fresh nn-lab conda env and downloads the lab notebook. Then activate the env and launch Jupyter.")
        self.clearSlide()
