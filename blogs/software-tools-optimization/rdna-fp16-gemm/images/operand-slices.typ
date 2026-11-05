// What one workgroup holds in LDS, and how it grows with block_k.
//
// The output tile is fixed at 256x256. The operand slices behind it are
// 256 x block_k of A and block_k x 256 of B, so deepening K widens both and the
// output tile is unchanged. Three depths are shown to make the scaling visible.
//
//   block_k = 16   256x16 and 16x256    16 KB per buffer
//   block_k = 32   256x32 and 32x256    32 KB per buffer
//   block_k = 64   256x64 and 64x256    64 KB per buffer
//
// Figures exclude the row padding, which the post covers separately.

#set page(width: auto, height: auto, margin: 0pt)
#let zoom = 220%
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

#let rose   = rgb("#f5cccd")
#let teal   = rgb("#bcd9d7")
#let teal2  = rgb("#8fbfbb")
#let ink    = rgb("#1e1e1e")
#let dimmed = rgb("#777777")
#let line   = rgb("#9a9a9a")

// -- Geometry ---------------------------------------------------------------
#let side = 46pt          // the 256x256 output tile, same in every panel
#let unit = 5pt           // visual width of 16 elements of K
#let maxdepth = unit * 4  // depth reserved in every panel, so panels align
#let panel-w  = 92pt   // wide enough for the 'A -> 256 x NN  B -> NN x 256' caption
#let fig-w    = 3 * panel-w + 40pt

#let panel(k, kb) = {
  let depth = unit * (k / 16)
  stack(
    dir: ttb, spacing: 4pt,
    // header
    text(size: 6.5pt)[
      #text(weight: "bold")[`block_k` = #k] #h(4pt) #text(fill: dimmed)[#kb KB]
    ],
    grid(
      columns: (maxdepth, 3pt, side),
      rows: (maxdepth, 3pt, side),
      // B slice above the output tile
      [], [],
      align(bottom, rect(width: side, height: depth, fill: teal, stroke: 0.5pt + line)),
      [], [], [],
      // A slice to the left
      align(right, rect(width: depth, height: side, fill: teal, stroke: 0.5pt + line)),
      [],
      rect(width: side, height: side, fill: rose, stroke: 0.5pt + line),
    ),
    v(1pt),
    text(size: 5.5pt, fill: dimmed)[
      A #sym.arrow.r 256 #sym.times #k #h(3pt) B #sym.arrow.r #k #sym.times 256
    ],
  )
}

#scale(zoom, reflow: true, pad(
  top: 12pt, bottom: 12pt, left: 12pt, right: 12pt,
  stack(
    dir: ttb, spacing: 9pt,

    text(size: 8pt, weight: "bold")[What a deeper K slice costs],
    text(size: 6pt, fill: dimmed)[
      The output tile is 256x256 in all three. Only the operand slices grow.
    ],

    grid(
      columns: (panel-w, 20pt, panel-w, 20pt, panel-w),
      align(top, panel(16, 16)), [], align(top, panel(32, 32)), [], align(top, panel(64, 64)),
    ),

    v(7pt),
    // Centred on the true figure width: three panels plus two gutters.
    grid(
      columns: (fig-w,),
      align(center, grid(
        columns: (8pt, 4pt, auto, 18pt, 8pt, 4pt, auto),
        rect(width: 8pt, height: 8pt, fill: rose, stroke: 0.3pt + line),
        [],
        text(size: 6pt, fill: dimmed)[256x256 output tile in registers],
        [],
        rect(width: 8pt, height: 8pt, fill: teal, stroke: 0.3pt + line),
        [],
        text(size: 6pt, fill: dimmed)[operand slices in LDS],
      )),
    ),

    v(3pt),
    text(size: 6pt, fill: dimmed)[
      Double buffering doubles the LDS figure, so `block_k = 32` already needs 64 KB of the \
      64 KB a workgroup may use in CU mode. That is what forces the output tile down to 128x96.
    ],
  )
))
