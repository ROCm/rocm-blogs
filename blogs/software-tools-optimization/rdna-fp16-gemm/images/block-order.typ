// Block scheduling: row-by-row against grouped-by-rows.
//
// The grid is the output matrix divided into block tiles. Shaded cells are the
// blocks resident at one moment (16 of them, for illustration). The bars along
// the edges show which strips of A and B those blocks read between them.
//
//   row-by-row   16 blocks span a wide band: 1 row-strip of A, 16 column-strips of B
//   grouped      16 blocks form a tall block: 8 row-strips of A, 2 column-strips of B
//
// Fewer distinct strips touched at once means more of them stay in cache.

#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)
#let zoom = 220%

#let rose    = rgb("#e39a9c")
#let pale    = rgb("#f6eaea")
#let teal    = rgb("#8fbfbb")
#let teal-pl = rgb("#e4efee")
#let ink     = rgb("#1e1e1e")
#let dimmed  = rgb("#777777")
#let line    = rgb("#bdbdbd")

// -- Geometry ---------------------------------------------------------------
#let cell    = 9pt
#let cols    = 16
#let rows    = 10
#let bar     = 6pt                            // thickness of the operand strips
#let gutter  = 22pt
#let panel-w = bar + 2pt + cols * cell        // one panel, strip included
#let fig-w   = 2 * panel-w + gutter

// -- Which cells are resident, for each schedule ----------------------------
#let rowwise(r, c) = r == 0                   // one grid row
#let grouped(r, c) = r < 8 and c < 2          // 8 rows x 2 columns

// -- The output grid, with A and B strips along the edges -------------------
#let panel(active, a-hit, b-hit, accent) = {
  let b-strip = grid(
    columns: (cell,) * cols,
    ..range(cols).map(c => rect(
      width: cell, height: bar,
      fill: if b-hit(c) { teal } else { teal-pl },
      stroke: 0.2pt + white,
    ))
  )
  let a-strip = grid(
    rows: (cell,) * rows,
    ..range(rows).map(r => rect(
      width: bar, height: cell,
      fill: if a-hit(r) { teal } else { teal-pl },
      stroke: 0.2pt + white,
    ))
  )
  let body = grid(
    columns: (cell,) * cols,
    rows: (cell,) * rows,
    ..range(rows).map(r => range(cols).map(c => rect(
      width: cell, height: cell,
      fill: if active(r, c) { accent } else { pale },
      stroke: 0.2pt + white,
    ))).flatten()
  )
  grid(
    columns: (bar, 2pt, auto),
    rows: (bar, 2pt, auto),
    [], [], b-strip,
    [], [], [],
    a-strip, [], body,
  )
}

#let caption(title, note) = stack(
  dir: ttb, spacing: 2pt,
  text(size: 7.5pt, weight: "bold")[#title],
  text(size: 6pt, fill: dimmed)[#note],
)

#let legend = grid(
  columns: (8pt, 4pt, auto, 18pt, 8pt, 4pt, auto),
  rect(width: 8pt, height: 8pt, fill: rose, stroke: 0.3pt + line),
  [],
  text(size: 6pt, fill: dimmed)[blocks resident now],
  [],
  rect(width: 8pt, height: 8pt, fill: teal, stroke: 0.3pt + line),
  [],
  text(size: 6pt, fill: dimmed)[operand strips they read],
)

#scale(zoom, reflow: true, pad(
  top: 12pt, bottom: 12pt, left: 12pt, right: 12pt,
  box(width: fig-w, stack(
    dir: ttb, spacing: 9pt,

    text(size: 8pt, weight: "bold")[Which blocks run at the same time],

    grid(
      columns: (panel-w, gutter, panel-w),
      stack(
        dir: ttb, spacing: 5pt,
        caption("Row by row", "16 concurrent blocks span a whole row of the output."),
        panel(rowwise, r => r == 0, c => true, rose),
      ),
      [],
      stack(
        dir: ttb, spacing: 5pt,
        caption("Grouped by rows", "The same 16 blocks, handed out in groups of 8 rows."),
        panel(grouped, r => r < 8, c => c < 2, rose),
      ),
    ),

    // Legend between the grids and the notes. A single-cell grid of the same
    // total width as the panels above centres it on the figure, not the page.
    grid(
      columns: (fig-w,),
      align(center, legend),
    ),

    grid(
      columns: (panel-w, gutter, panel-w),
      text(size: 6pt, fill: dimmed)[
        Reads 1 row-strip of A and #text(fill: ink)[all 16] column-strips of B.
        B is walked end to end before any of it is reused.
      ],
      [],
      text(size: 6pt, fill: dimmed)[
        Reads 8 row-strips of A and #text(fill: ink)[2] column-strips of B.
        Both operands stay small enough to be reused from cache.
      ],
    ),
  )),
))
