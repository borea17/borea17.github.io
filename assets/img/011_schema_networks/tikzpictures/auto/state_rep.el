(TeX-add-style-hook
 "state_rep"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("standalone" "varwidth" "border=10pt")))
   (TeX-run-style-hooks
    "latex2e"
    "standalone"
    "standalone10"
    "tikz"
    "amsmath"
    "bm"
    "makecell"
    "color"
    "xcolor"))
 :latex)

