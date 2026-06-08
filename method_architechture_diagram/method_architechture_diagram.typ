#import "@preview/fletcher:0.5.8" as fletcher
#import fletcher: diagram, edge, node
#import fletcher.shapes: *
#import "@preview/plotsy-3d:0.2.1": plot-3d-surface

#set page(height: auto, width: auto, fill: none)

#scale(40%, reflow: true)[
  #diagram(
    node(
      (-1, 0),
      [
        *Create Initial Population*
        #linebreak()
        _(random genomes)_
        #linebreak()

        #set align(left)

        - hidden layer sizes
        - hidden layer count
        - hidden layer activation functions
        - learning rate
      ],
      corner-radius: 5pt,
      fill: green.lighten(80%),
      stroke: green.darken(20%),
      name: <init>,
    ),
    node(
      (1, 0),
      [
        *Build MLP from Genome*
        #linebreak()
        #scale(40%, reflow: true)[
          #diagram(
            node((0, 0), shape: circle, stroke: black + 1pt, [$space$], name: <n1_1>),
            node((0, 1), shape: circle, stroke: black + 1pt, [$space$], name: <n1_2>),

            node((1, -0.5), shape: circle, stroke: black + 1pt, [$space$], name: <n2_1>),
            node((1, 0.5), shape: circle, stroke: black + 1pt, [$space$], name: <n2_2>),
            node((1, 1.5), shape: circle, stroke: black + 1pt, [$space$], name: <n2_3>),

            node((1.5, -0.5), [...], name: <o.1>),
            node((1.5, 0.5), [...], name: <o.1>),
            node((1.5, 1.5), [...], name: <o.1>),

            node((2, -0.5), shape: circle, stroke: black + 1pt, [$space$], name: <nx_1>),
            node((2, 0.5), shape: circle, stroke: black + 1pt, [$space$], name: <nx_2>),
            node((2, 1.5), shape: circle, stroke: black + 1pt, [$space$], name: <nx_3>),

            node((3, 0.5), shape: circle, stroke: black + 1pt, [$space$], name: <o_1>),

            edge(<n1_1>, <n2_1>),
            edge(<n1_1>, <n2_2>),
            edge(<n1_1>, <n2_3>),

            edge(<n1_2>, <n2_1>),
            edge(<n1_2>, <n2_2>),
            edge(<n1_2>, <n2_3>),

            edge(<nx_1>, <o_1>),
            edge(<nx_2>, <o_1>),
            edge(<nx_3>, <o_1>),
          )
        ]
      ],
      corner-radius: 5pt,
      fill: blue.lighten(80%),
      stroke: blue.darken(20%),
      shape: rect,
      name: <build_mlp>,
    ),
    node(
      (2, 0),
      [
        *Game Simulation*
        #linebreak()
        _(training)_
        #linebreak()
        #box(stroke: 1pt + blue.darken(20%))[
          #image("screenshot.png", width: 2cm)
        ]
      ],
      corner-radius: 5pt,
      fill: blue.lighten(80%),
      stroke: blue.darken(20%),
      shape: rect,
      name: <simu_training>,
    ),
    node(
      (3, 0),
      [
        *Reinforcement*
        #linebreak()

        #scale(40%, reflow: true)[

          #let size = 10
          #let scale-factor = 0.11
          #let (xscale, yscale, zscale) = (0.3, 0.3, 0.02)
          #let scale-dim = (xscale * scale-factor, yscale * scale-factor, zscale * scale-factor)
          #let func(x, y) = x * x + y * y
          #let color-func(x, y, z, x-lo, x-hi, y-lo, y-hi, z-lo, z-hi) = {
            gradient.linear(..color.map.viridis).sample(calc.clamp((z + 50) / 250, 0, 1) * 100%)
          }

          #plot-3d-surface(
            func,
            color-func: color-func,
            scale-dim: scale-dim,
            xdomain: (-size, size),
            ydomain: (-size, size),
            axis-labels: ([], [], []),
            pad-high: (0, 0, 0),
            pad-low: (0, 0, 5),
            dot-thickness: 0em,
            front-axis-thickness: 0em,
            front-axis-dot-scale: (0.05, 0.05),
            rear-axis-dot-scale: (0.08, 0.08),
            rear-axis-text-size: 0em,
            axis-label-size: 1.5em,
            xyz-colors: (black.transparentize(100%), black.transparentize(100%), black.transparentize(100%)),
          )
        ]
      ],
      corner-radius: 5pt,
      fill: blue.lighten(80%),
      stroke: blue.darken(20%),
      shape: rect,
      name: <reinforce>,
    ),
    node(
      (4, 0),
      [
        *Game Simulation*
        #linebreak()
        _(evaluation)_
        #linebreak()
        #box(stroke: 1pt + blue.darken(20%))[
          #image("screenshot.png", width: 2cm)
        ]
      ],
      corner-radius: 5pt,
      fill: blue.lighten(80%),
      stroke: blue.darken(20%),
      shape: rect,
      name: <simu_eval>,
    ),
    node(
      (5, 0),
      [
        *Compute Fitness*
        #linebreak()
        _(score)_
        #linebreak()
        #image("trophy.png", width: 2cm)
      ],
      corner-radius: 5pt,
      fill: blue.lighten(80%),
      stroke: blue.darken(20%),
      shape: rect,
      name: <score>,
    ),
    node(
      (7, 0),
      [
        Max Generation
        #linebreak()
        Reached ?
      ],
      inset: 15pt,
      fill: orange.lighten(80%),
      stroke: orange.darken(20%),
      shape: diamond,
      name: <continue>,
    ),
    node(
      (4, 1.3),
      [
        *Selection*
        #linebreak()
        _(pick the best individuals)_
        #let w = 0.4cm
        #let picked = 12
        #let not_picked = 32 - picked
        #grid(
          columns: 8,
          ..(image("dna_red.svg", width: w),) * picked,
          ..(image("dna.svg", width: w),) * not_picked,
        )
      ],
      inset: 15pt,
      corner-radius: 5pt,
      fill: purple.lighten(80%),
      stroke: purple.darken(20%),
      shape: rect,
      name: <select>,
    ),
    node(
      (3, 1.3),
      [
        *Crossover*
        #linebreak()
        _(mix the selected individuals)_
        #stack(
          dir: ltr,
          image("dna_blue.svg", width: 1.5cm),
          text(24pt)[*+*],
          image("dna_green.svg", width: 1.5cm),
          text(24pt)[*$->$*],
          image("dna_mixed.svg", width: 1.5cm),
        )
      ],
      inset: 15pt,
      corner-radius: 5pt,
      fill: purple.lighten(80%),
      stroke: purple.darken(20%),
      shape: rect,
      name: <crossover>,
    ),
    node(
      (2, 1.3),
      [
        *Mutation*
        #linebreak()
        _(some random #linebreak() genome variation)_
        #image("dna_mut.svg", width: 1.5cm)
      ],
      inset: 15pt,
      corner-radius: 5pt,
      fill: purple.lighten(80%),
      stroke: purple.darken(20%),
      shape: rect,
      name: <mutation>,
    ),

    edge(<init>, <build_mlp>, "-|>", mark-scale: 200%),
    edge(<build_mlp>, <simu_training>, "-|>", mark-scale: 200%),
    edge(<simu_training>, <reinforce>, "-|>", mark-scale: 200%),
    edge(<reinforce>, <simu_eval>, "-|>", mark-scale: 200%),
    edge(<simu_eval>, <score>, "-|>", mark-scale: 200%),
    edge(<score>, <continue>, "-|>", mark-scale: 200%),
    edge(<continue>, <select>, "-|>", mark-scale: 200%, bend: 30deg, label: [No]),
    edge(<select>, <crossover>, "-|>", mark-scale: 200%),
    edge(<crossover>, <mutation>, "-|>", mark-scale: 200%),
    edge(<mutation>, <build_mlp>, "-|>", mark-scale: 200%, bend: 40deg),
    edge(<reinforce.130deg>, <simu_training.55deg>, "-|>", mark-scale: 200%, bend: -90deg, label: [100 Epochs]),

    node(
      (0.5, -1),
      [*Evaluation*],
      name: <helper>,
      corner-radius: 5pt,
      fill: purple.lighten(80%),
      stroke: purple.darken(20%),
    ),

    node(
      enclose: (
        <build_mlp>,
        <simu_training>,
        <reinforce>,
        <simu_eval>,
        <score>,
        <helper>,
      ),
      inset: 20pt,
      fill: purple.lighten(95%),
      stroke: (paint: purple.darken(20%), thickness: 1pt, dash: "dashed"),
      corner-radius: 5pt,
    ),
  )
]
