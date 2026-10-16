// The order a wave issues instructions in, before and after interleaving.
//
// Both schedules do the same work for one K tile: 14 global chunk loads of the
// *next* tile, 28 LDS fragment loads of the *current* one, and 48 matrix
// instructions. Only the order differs.
//
//   before   every global load first, then the fragments, then the matrix
//            instructions in one long run with nothing behind them
//   after    a global chunk between matrix instructions, and each fragment
//            issued just before the instruction that consumes it
//
// The sequences are abbreviated: an ellipsis stands for more of the same. The
// figure is about the shape of the schedule, not the exact indices.
//
// Sized for phones: the figure is about 540 units wide and no text is below
// 13 units, so labels stay readable when it is scaled down to a narrow screen.

#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)
#let zoom = 200%

#let rose   = rgb("#e39a9c")     // global -> VGPR
#let teal   = rgb("#8fbfbb")     // LDS -> VGPR
#let pale   = rgb("#f6eaea")     // WMMA
#let ink    = rgb("#1e1e1e")
#let dimmed = rgb("#555555")
#let line   = rgb("#bdbdbd")

// -- Geometry ---------------------------------------------------------------
#let cell   = 27pt
#let skip   = 12pt              // an abbreviated run
#let arrow  = 7pt               // space taken by the arrow between instructions
#let row-h  = 15pt

// ("G"|"L"|"M", label) is an instruction; ("E", "") is an ellipsis.
#let before-seq = (
  ("G", "glb A0"), ("E", ""), ("G", "glb B5"),
  ("L", "lds A0"), ("E", ""),
  ("M", "wmma"), ("M", "wmma"), ("E", ""),
)

#let after-seq = (
  ("L", "lds A0"), ("G", "glb A0"), ("L", "lds B1"), ("M", "wmma"),
  ("G", "glb B0"), ("L", "lds B2"), ("M", "wmma"), ("E", ""),
)

// Once the 14 chunks are issued the global stream stops; fragments and matrix
// instructions carry on to the end of the fold.
#let tail-seq = (
  ("M", "wmma"), ("M", "wmma"), ("L", "lds A3"), ("M", "wmma"), ("M", "wmma"), ("E", ""),
)

#let colour-of(k) = if k == "G" { rose } else if k == "L" { teal } else { pale }

#let insn(entry) = {
  let (kind, label) = entry
  if kind == "E" {
    box(width: skip, height: row-h,
      align(center + horizon, text(size: 7pt, fill: dimmed)[#sym.dots.h]))
  } else {
    rect(width: cell, height: row-h, inset: 1pt,
      fill: colour-of(kind), stroke: 0.3pt + line,
      align(center + horizon, text(size: 6.5pt, fill: ink)[#label]))
  }
}

// One sequence on a single row, arrows between instructions.
#let sequence(seq) = {
  let widths = seq.map(e => if e.at(0) == "E" { skip } else { cell })
  grid(
    columns: widths.zip(range(seq.len())).map(((w, i)) =>
      if i < seq.len() - 1 { (w, arrow) } else { (w,) }).flatten(),
    ..seq.enumerate().map(((i, e)) => {
      if i < seq.len() - 1 {
        (insn(e), align(center + horizon, text(size: 6.5pt, fill: dimmed)[#sym.arrow.r]))
      } else {
        (insn(e),)
      }
    }).flatten()
  )
}

#let seq-width(seq) = {
  seq.map(e => if e.at(0) == "E" { skip } else { cell }).sum() + (seq.len() - 1) * arrow
}
#let fig-w = calc.max(seq-width(before-seq), seq-width(after-seq), seq-width(tail-seq))

#let legend-row(colour, label) = grid(
  columns: (9pt, 4pt, auto),
  align: horizon,
  rect(width: 9pt, height: 9pt, fill: colour, stroke: 0.3pt + line),
  [], text(size: 6.5pt, fill: dimmed)[#label],
)

#let legend = stack(
  dir: ttb, spacing: 3pt,
  legend-row(rose, [global #sym.arrow.r VGPR, next K tile]),
  legend-row(teal, [LDS #sym.arrow.r VGPR, current tile]),
  legend-row(pale, [WMMA]),
)

#scale(zoom, reflow: true, pad(
  top: 10pt, bottom: 10pt, left: 10pt, right: 10pt,
  box(width: fig-w, stack(
    dir: ttb, spacing: 11pt,

    text(size: 7.5pt, weight: "bold")[The order a wave issues instructions in],

    stack(
      dir: ttb, spacing: 4pt,
      text(size: 7pt, weight: "bold")[Before],
      sequence(before-seq),
      text(size: 6.5pt, fill: dimmed)[
        All 14 global loads go out first, then the fragments, then 48 matrix instructions
        with no memory traffic behind any of them.
      ],
    ),

    stack(
      dir: ttb, spacing: 4pt,
      text(size: 7pt, weight: "bold")[After],
      sequence(after-seq),
      text(size: 6.5pt, fill: dimmed)[
        A global chunk sits between matrix instructions, and each fragment is issued an
        instruction ahead of the one that needs it.
      ],
    ),

    stack(
      dir: ttb, spacing: 4pt,
      text(size: 7pt, weight: "bold")[After, once the 14 chunks have run out],
      sequence(tail-seq),
      text(size: 6.5pt, fill: dimmed)[
        The global stream stops. The rest of the fold is matrix instructions and the
        occasional fragment read.
      ],
    ),

    legend,
  )),
))
