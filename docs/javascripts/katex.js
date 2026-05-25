// KaTeX auto-render bootstrap (pymdownx.arithmatex with generic: true)
// Loaded after katex.min.js + auto-render.min.js by mkdocs.yml.
document$.subscribe(() => {
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$',  right: '$$',  display: true  },
      { left: '$',   right: '$',   display: false },
      { left: '\\(', right: '\\)', display: false },
      { left: '\\[', right: '\\]', display: true  },
    ],
    throwOnError: false,
  });
});
