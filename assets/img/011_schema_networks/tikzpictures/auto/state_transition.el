(TeX-add-style-hook
 "state_transition"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("standalone" "tikz")))
   (TeX-run-style-hooks
    "latex2e"
    "standalone"
    "standalone10"
    "amsmath"
    "bm"
    "makecell"
    "color"
    "xcolor")
   (LaTeX-add-xcolor-definecolors
    "mygray"))
 :latex)

