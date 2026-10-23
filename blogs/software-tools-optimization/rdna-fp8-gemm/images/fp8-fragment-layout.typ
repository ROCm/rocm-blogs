// Operand fragment layout per lane: RDNA 4 FP16 vs RDNA 4 FP8.
//
// Left panel:  8 FP16 values per lane (16 bytes).
// Right panel: 8 FP8 values per lane (8 bytes, two packed 32-bit integers).
//
// The K split is identical: lanes 0-15 hold K 0-7, lanes 16-31 hold K 8-15.
// Lanes 4-11 and slots 3-4 are elided.
//
// Sized for phones: the panels are stacked so the figure is about 540 units
// wide, and no text is below 13 units.

#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)
#let zoom = 210%

#let rose   = rgb("#f5cccd")
#let ink    = rgb("#1e1e1e")
#let dimmed = rgb("#555555")
#let line   = rgb("#9a9a9a")
#let white  = rgb("#ffffff")

#let cell   = 24pt
#let cell-h = 17pt
#let brk    = 12pt
#let lead-w = 40pt

#let lanes = (0, 1, 2, 3, -1, 12, 13, 14, 15)
#let slots = (0, 1, 2, -1, 5, 6, 7)
#let col-widths = lanes.map(l => if l < 0 { brk } else { cell })

#let grid-for(k-of) = grid(
  columns: (lead-w, 3pt) + col-widths,
  rows: (10pt, 9pt, 3pt) + slots.map(s => if s < 0 { brk } else { cell-h }),

  align(right + horizon, text(size: 6.5pt, fill: dimmed)[lane]),
  [],
  ..lanes.map(l => align(center + horizon,
    if l < 0 { text(size: 7pt, fill: dimmed)[#sym.dots.h] }
    else      { text(size: 6.5pt, fill: ink)[#l] })),

  align(right + horizon, text(size: 6.5pt, fill: dimmed)[reads row]),
  [],
  ..lanes.map(l => align(center + horizon,
    if l < 0 { [] }
    else      { text(size: 6.5pt, fill: dimmed)[#calc.rem(l, 16)] })),

  [], [], ..lanes.map(_ => []),

  ..slots.map(s => (
    align(right + horizon,
      if s < 0 { text(size: 7pt, fill: dimmed)[#sym.dots.v] }
      else      { text(size: 6.5pt, fill: dimmed)[slot #s] }),
    [],
    ..lanes.map(l =>
      if s < 0 {
        align(center + horizon,
          if l >= 0 { text(size: 7pt, fill: dimmed)[#sym.dots.v] }
          else      { [] })
      } else if l < 0 {
        align(center + horizon, text(size: 7pt, fill: dimmed)[#sym.dots.h])
      } else {
        rect(width: cell, height: cell-h,
          fill: rose, stroke: 0.25pt + white,
          align(center + horizon,
            text(size: 6.5pt, fill: ink)[K#k-of(l, s)]))
      }
    ),
  )).flatten(),
)

#let fig-w = lead-w + 3pt + col-widths.sum()

#let panel(title, note, body) = stack(
  dir: ttb, spacing: 4pt,
  text(size: 7.5pt, weight: "bold")[#title],
  text(size: 6.5pt, fill: dimmed)[#note],
  v(2pt),
  body,
)

#scale(zoom, reflow: true, pad(
  top: 10pt, bottom: 10pt, left: 10pt, right: 10pt,
  box(width: fig-w, stack(
    dir: ttb, spacing: 12pt,

    text(size: 8pt, weight: "bold")[Operand fragment layout per lane — RDNA 4],

    panel("FP16 — 8 FP16 values per lane",
      "16 bytes per lane. Lanes 0–15 hold K 0–7; lanes 16–31 hold K 8–15.",
      grid-for((l, s) => s + 8 * calc.div-euclid(l, 16))),

    panel("FP8 — 8 FP8 values per lane",
      "8 bytes per lane (two packed 32-bit integers). Same K split as FP16.",
      grid-for((l, s) => s + 8 * calc.div-euclid(l, 16))),

    text(size: 6.5pt, fill: dimmed)[
      The FP8 fragment is half the width of FP16 but holds the same K values per lane.
      Lanes 4–11 and slots 3–4 are elided.
    ],
  )),
))
