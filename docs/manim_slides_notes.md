# Manim Slides — working notes

Reference distilled from the official docs (manim-slides.eertmans.be, v latest,
© 2024–2026). Manim Slides sits on top of **manim-ce** (Manim Community). You
write normal Manim scenes, mark slide breaks, render, then present.

## Mental model
Manim Slides' API is tiny: two classes — `Slide` and `ThreeDSlide` — that
subclass Manim's `Scene` / `ThreeDScene`. Everything else is just Manim. A
"slide" is the chunk of animation between two `self.next_slide()` calls; pressing
the arrow key advances to the next chunk. So a presentation is one `construct()`
with pauses, not many files.

## Two-step workflow
1. Render: `manim-slides render scene.py SceneName`
   (wrapper around `manim render`; uses the manim in the same env).
2. Present: `manim-slides present SceneName` (live, opens a player),
   or convert: `manim-slides convert SceneName out.html` (self-contained
   RevealJS HTML — best for sharing/portability).
   - `manim-slides convert SceneName out.html -ccontrols=true` shows nav arrows.
   - `--to=pptx` and `--to=html` also supported. `--open` opens it.

## Core API (the only methods we need)
- `self.next_slide(...)` — end current slide, start the next. Key kwargs:
  - `loop=True` — the slide's animation loops until advanced (great for a
    repeating gradient-descent or sliding-kernel animation). NOTE: a loop cannot
    be the first or last slide in RevealJS/HTML output.
  - `auto_next=True` — auto-advance to next slide when this one finishes (HTML /
    present only). Good for a build that should flow without a keypress.
  - `notes="..."` — presenter notes (Markdown). View with `S` in the browser
    speaker view. Supported in present / convert html / convert pptx.
  - `playback_rate`, `reversed_playback_rate` — present only.
  - `src=Path(...)` — insert an EXTERNAL media file (image/GIF/video) as the next
    slide. This is our hook for "insert images later" (see below).
- `self.wipe(current, future, direction=...)` — shift current mobjects out and
  next ones in. Clean section transitions. `direction=UP/DOWN/LEFT/RIGHT` (and
  combos). `return_animation=True` to compose with other anims.
- `self.zoom(current, future, ...)` — fade out current, fade in next toward
  camera. `out=True`, `scale=...` available.
- Canvas (persistent objects across slides): `self.add_to_canvas(name=mobj)`,
  `self.canvas["name"]`, `self.mobjects_without_canvas`,
  `self.remove_from_canvas("name")`. Use for a persistent title + slide number.
- `self.wait_time_between_slides = 0.1` — adds a tiny wait at end of each slide
  so the last frame of an animation actually completes (avoids "circle not quite
  closed" artifacts). Set a small value > 1/FPS.
- `start_skip_animations()` / `stop_skip_animations()` or `skip_animations=True`
  on `next_slide` — skip rendering some slides while iterating (faster dev).
- Calls to `next_slide()` at the very start/end are auto-added; don't bother.

## Minimal pattern
```python
from manim import *
from manim_slides import Slide

class Example(Slide):
    def construct(self):
        text = Text("Hello")
        self.play(FadeIn(text))
        self.next_slide()              # pause / new slide
        self.play(FadeOut(text))
```

## 3D
Use `ThreeDSlide` (subclass of `ThreeDScene`). With manim-ce use
`self.set_camera_orientation(phi=..., theta=...)` and
`self.begin_ambient_camera_rotation(rate=...)`. (ManimGL handles 3D via
`self.camera.frame` instead — we are using manim-ce, so ignore ManimGL paths.)

## Other scene types (e.g. moving camera)
Manim Slides only ships `Slide` / `ThreeDSlide`, but you can mix in any Manim
scene by multiple inheritance:
```python
class MovingCameraSlide(Slide, MovingCameraScene):
    pass
```
Then subclass `MovingCameraSlide`. Useful for pan/zoom over a big diagram.

## Inserting images / media (our "add images later" requirement)
Two ways:
1. Inline in a slide as a Manim mobject:
   `img = ImageMobject("path.png"); self.play(FadeIn(img))`. Position with
   `.to_edge()`, `.next_to()`, `.scale()`. (ImageMobject is plain manim-ce.)
2. As a whole slide via `self.next_slide(src=Path(...)/"figure.png")` — supports
   images, GIFs, and videos (`SlideTypesExample` in docs).
DESIGN RULE for us: reserve a region (e.g. right third) and drop in a
placeholder `Rectangle` with a `Text("image: ...")` label now; swap for
`ImageMobject` later. Keep a `# TODO image:` comment so they're easy to find.

## Theming for our minimalist look (charcoal bg, white text)
- Background: set per-scene in `construct` with
  `self.camera.background_color = "#1e1e22"` (a pleasant dark charcoal), or
  globally via a manim config / `config.background_color`.
- Default text/stroke: pass `color=WHITE` (manim's default text is already
  near-white, but be explicit). Consider a small palette of 1–2 accent colors
  (e.g. a muted blue `#5aa9e6` / amber `#e6b35a`) used sparingly for emphasis.
- Keep one `Text`/`MathTex` font size scheme; avoid clutter. Lots of negative
  space. Use `MathTex` for all equations (LaTeX) so notation looks clean.
- Define a helper (e.g. `slideTitle(str)`) so every slide title is consistent.

## Render quality / housekeeping
- `manim-slides render -qh scene.py SceneName` for high quality (`-ql` low while
  drafting). Output media land in manim's media dir; convert to HTML into
  `manim_slides/outputs/`.
- Caching: `Slide` supports `disable_caching`, `flush_cache`. Reversed-animation
  generation is memory-heavy; `max_duration_before_split_reverse` and
  `skip_reversing` control it. Leave defaults unless renders get slow.

## Gotchas
- A `loop=True` slide can't be first or last (RevealJS limit).
- `auto_next`, `playback_rate`, `notes` markdown only matter for present / HTML.
- Manim Slides wraps manim render — make sure manim-ce is installed in the same
  Python environment.

## Install (record once confirmed in our env)
`pip install manim-slides[manim]` (pulls manim-ce). System deps for manim:
ffmpeg, and a LaTeX install for `MathTex`/`Tex`.
