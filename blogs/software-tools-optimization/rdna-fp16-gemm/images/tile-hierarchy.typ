// Tile hierarchy for the 256x256 block tile (step 2 configuration).
//
//   block tile      256 x 256 patch of C, computed by one workgroup
//   wave grid       4 x 4 waves, so each wave owns a 64 x 64 quadrant
//   WMMA tiles      each quadrant is a 4 x 4 grid of 16 x 16 instruction tiles
//   operand slices  256 x block_k of A and block_k x 256 of B in LDS
//
// The point of the figure is the nesting and the operand slices along the edges,
// not the individual cells.
//
// Sized for phones: annotations sit below the tile instead of beside it, so the
// figure is about 540 units wide and no text is below 13 units.

#set page(width: auto, height: auto, margin: 0pt)
#let zoom = 190%
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

#let rose    = rgb("#f5cccd")
#let rose-hi = rgb("#e39a9c")
#let teal    = rgb("#bcd9d7")
#let teal-hi = rgb("#8fbfbb")
#let ink     = rgb("#1e1e1e")
#let dimmed  = rgb("#777777")
#let line    = rgb("#9a9a9a")

// -- Geometry ---------------------------------------------------------------
#let cell   = 13pt          // one 16x16 WMMA tile
#let quad   = cell * 4      // one wave quadrant = 4x4 WMMA tiles
#let full   = quad * 4      // block tile = 4x4 quadrants
#let kdepth = 15pt          // visual depth of the block_k dimension

// -- The C tile: 4x4 quadrants, each a 4x4 grid of WMMA tiles ---------------
#let c-tile = {
  // one quadrant, with its 16 WMMA cells
  let quadrant(hl) = grid(
    columns: (cell,) * 4,
    rows: (cell,) * 4,
    ..range(16).map(i => rect(
      width: cell, height: cell,
      fill: if hl { rose-hi } else { rose },
      stroke: 0.3pt + rgb("#ffffff"),
    ))
  )
  // 4x4 grid of quadrants, wave 0 highlighted
  grid(
    columns: (quad,) * 4,
    rows: (quad,) * 4,
    ..range(16).map(w => block(
      stroke: 1.1pt + rgb("#6b6b6b"),
      quadrant(w == 0),
    ))
  )
}

// -- A slice: 256 rows x block_k, drawn to the left -------------------------
#let a-slice = grid(
  columns: (kdepth,),
  rows: (quad,) * 4,
  ..range(4).map(_ => rect(
    width: kdepth, height: quad,
    fill: teal, stroke: 0.6pt + line,
  ))
)

// -- B slice: block_k x 256, drawn above ------------------------------------
#let b-slice = grid(
  columns: (quad,) * 4,
  rows: (kdepth,),
  ..range(4).map(_ => rect(
    width: quad, height: kdepth,
    fill: teal, stroke: 0.6pt + line,
  ))
)

// -- Labels -----------------------------------------------------------------
#let label-w = 42pt
#let grid-w = label-w + kdepth + 4pt + full

#let note(swatch, body) = grid(
  columns: (9pt, 4pt, 1fr),
  align: top,
  swatch, [], body,
)

#scale(zoom, reflow: true, pad(
  top: 12pt, bottom: 10pt, left: 10pt, right: 10pt,
  box(width: grid-w, stack(
    dir: ttb, spacing: 8pt,

    text(size: 8pt, weight: "bold")[Tile hierarchy for one workgroup],

    grid(
      columns: (label-w, kdepth, 4pt, full),
      rows: (kdepth, 4pt, full),
      row-gutter: 0pt,

      // --- row 1: B slice sits above the C tile ---
      align(right + horizon, pad(right: 4pt, text(size: 7pt, fill: dimmed)[B slice])),
      [], [],
      b-slice,

      // --- spacer row ---
      [], [], [], [],

      // --- row 2: A slice, then the C tile ---
      align(right + horizon, pad(right: 4pt, text(size: 7pt, fill: dimmed)[A slice])),
      a-slice,
      [],
      c-tile,
    ),

    stack(
      dir: ttb, spacing: 4pt,
      text(size: 7pt)[
        #text(weight: "bold")[256 x 256] block tile:
        #text(fill: dimmed)[the patch of C one workgroup computes]
      ],
      text(size: 7pt)[
        #text(weight: "bold")[64 x 64] wave quadrant:
        #text(fill: dimmed)[outlined; 16 waves in a 4x4 grid]
      ],
      text(size: 7pt)[
        #text(weight: "bold")[16 x 16] WMMA tile:
        #text(fill: dimmed)[one matrix instruction, 16 per wave]
      ],
      text(size: 7pt)[
        #text(weight: "bold")[`block_k` x 256] B slice:
        #text(fill: dimmed)[one K slice of B]
      ],
      note(
        rect(width: 8pt, height: 8pt, fill: rose-hi, stroke: 0.3pt + line),
        text(size: 7pt, fill: dimmed)[wave 0's quadrant],
      ),
    ),

    text(size: 7pt, fill: dimmed)[
      Only the K dimension of the operand slices grows when `block_k` deepens; the block tile
      is unchanged. At `block_k = 16` the two slices are 256x16 and 16x256, which is 16 KB of LDS
      per buffer.
    ],
  )),
))
