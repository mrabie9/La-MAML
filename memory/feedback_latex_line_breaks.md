---
name: LaTeX prose line breaks
description: Never insert hard line breaks within paragraphs or prose in .tex files
type: feedback
---

Never insert hard line breaks within paragraphs or prose text in LaTeX (.tex) files. Each paragraph should be a single unbroken line, regardless of length.

**Why:** User preference for clean, unwrapped LaTeX source.

**How to apply:** Applies to all prose, section text, captions, and running text. Does NOT apply to LaTeX structural elements where line breaks are semantically meaningful (e.g. `\\` in tables or align environments, or blank lines separating paragraphs).
