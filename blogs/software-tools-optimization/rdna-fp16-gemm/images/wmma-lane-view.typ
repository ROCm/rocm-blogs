// What each lane of the wave holds, RDNA 3 against RDNA 4.
//
// Columns are the 32 lanes; rows are the slots of one lane's fragment. A cell
// gives the K value that slot holds. The tile row a lane reads is lane % 16,
// printed under the lane number.
//
//   RDNA 3   lane L and lane L+16 read the same row and the same 16 K values
//   RDNA 4   lanes 0-15 take K 0-7 of their row, lanes 16-31 take K 8-15
//
// Lanes 4-14 and 20-30 are elided, as are the middle slots.

#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)
#let zoom = 210%

#let rose   = rgb("#f5cccd")
#let grey   = rgb("#bfbfbf")
#let both   = rgb("#dcc6cc")
#let ink    = rgb("#1e1e1e")
#let dimmed = rgb("#777777")
#let line   = rgb("#9a9a9a")

#let cell   = 24pt
#let brk    = 12pt
#let lead-w = 34pt

// -1 marks an elided column or row.
#let lanes = (0, 1, 2, 3, -1, 15, 16, 17, 18, 19, -1, 31)
#let col-widths = lanes.map(l => if l < 0 { brk } else { cell })

#let grid-for(slots, k-of, fill-of) = grid(
  columns: (lead-w, 3pt) + col-widths,
  rows: (10pt, 9pt, 3pt) + slots.map(s => if s < 0 { brk } else { cell }),

  // lane number
  align(right + horizon, text(size: 5.5pt, fill: dimmed)[lane]),
  [],
  ..lanes.map(l => align(center + horizon,
    if l < 0 { text(size: 7pt, fill: dimmed)[#sym.dots.h] }
    else { text(size: 5.5pt, fill: ink)[#l] })),

  // tile row that lane reads
  align(right + horizon, text(size: 5pt, fill: dimmed)[reads row]),
  [],
  ..lanes.map(l => align(center + horizon,
    if l < 0 { [] } else { text(size: 5pt, fill: dimmed)[#calc.rem(l, 16)] })),

  [], [], ..lanes.map(_ => []),

  ..slots.map(s => (
    align(right + horizon,
      if s < 0 { text(size: 7pt, fill: dimmed)[#sym.dots.v] }
      else { text(size: 5pt, fill: dimmed)[slot #s] }),
    [],
    ..lanes.map(l => {
      if s < 0 or l < 0 {
        align(center + horizon, text(size: 7pt, fill: dimmed)[
          #if s < 0 and l >= 0 { sym.dots.v } else if l < 0 and s >= 0 { sym.dots.h } else { "" }
        ])
      } else {
        rect(width: cell, height: cell, fill: fill-of(l), stroke: 0.25pt + white,
          align(center + horizon, text(size: 5.5pt, fill: ink)[K#(k-of(l, s))]))
      }
    }),
  )).flatten(),
)

#let rdna3 = grid-for((0, 1, 2, -1, 15), (l, s) => s, l => both)
#let rdna4 = grid-for((0, 1, 2, -1, 7), (l, s) => s + 8 * calc.div-euclid(l, 16), l => if l < 16 { rose } else { grey })

#let fig-w = lead-w + 3pt + col-widths.sum()

#let panel(title, note, body) = stack(
  dir: ttb, spacing: 4pt,
  text(size: 7.5pt, weight: "bold")[#title],
  text(size: 6pt, fill: dimmed)[#note],
  v(2pt),
  body,
)

#scale(zoom, reflow: true, pad(
  top: 12pt, bottom: 12pt, left: 12pt, right: 12pt,
  box(width: fig-w, stack(
    dir: ttb, spacing: 13pt,

    text(size: 8pt, weight: "bold")[What each lane holds when the instruction issues],

    panel("RDNA 3 — 16 FP16 per lane",
      "Lanes L and L+16 hold the same row and the same K values: the right half of the wave repeats the left.",
      rdna3),

    panel("RDNA 4 — 8 FP16 per lane",
      "Lanes 0-15 hold the first half of K, lanes 16-31 the second. No duplication.",
      rdna4),

    align(center, grid(
      columns: (9pt, 4pt, auto, 16pt, 9pt, 4pt, auto, 16pt, 9pt, 4pt, auto),
      rect(width: 9pt, height: 9pt, fill: both, stroke: 0.3pt + line),
      [], text(size: 6pt, fill: dimmed)[held by both halves],
      [],
      rect(width: 9pt, height: 9pt, fill: rose, stroke: 0.3pt + line),
      [], text(size: 6pt, fill: dimmed)[lanes 0-15],
      [],
      rect(width: 9pt, height: 9pt, fill: grey, stroke: 0.3pt + line),
      [], text(size: 6pt, fill: dimmed)[lanes 16-31],
    )),

    text(size: 6pt, fill: dimmed)[
      The row a lane holds is `lane % 16`, so the same picture serves A and B. Lanes 4-14
      and 20-30 are elided, as are the middle slots.
    ],
  )),
))
