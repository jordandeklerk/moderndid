/* Keep sphinx-copybutton working with sphinx-immaterial instant navigation. */
window.DOCUMENTATION_OPTIONS = window.DOCUMENTATION_OPTIONS || { URL_ROOT: "" };

(function () {
  if (window.__moderndidCopybuttonLoaded) {
    return;
  }
  window.__moderndidCopybuttonLoaded = true;

  var selector = "div.highlight pre";

  function icon() {
    var existing = document.querySelector("button.copybtn");
    if (existing) {
      return existing.innerHTML;
    }
    return (
      '<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" ' +
      'width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" ' +
      'stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">' +
      "<title>Copy to clipboard</title>" +
      '<path stroke="none" d="M0 0h24v24H0z" fill="none"/>' +
      '<rect x="8" y="8" width="12" height="12" rx="2" />' +
      '<path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2" />' +
      "</svg>"
    );
  }

  function attach() {
    var cells = document.querySelectorAll(selector);
    var template = null;
    for (var i = 0; i < cells.length; i++) {
      var cell = cells[i];
      var next = cell.nextElementSibling;
      if (next && next.classList && next.classList.contains("copybtn")) {
        continue;
      }
      if (!cell.id) {
        cell.setAttribute("id", "shimcodecell" + i);
      }
      if (template === null) {
        template = icon();
      }
      var button = document.createElement("button");
      button.className = "copybtn o-tooltip--left";
      button.setAttribute("data-tooltip", "Copy");
      button.setAttribute("data-clipboard-target", "#" + cell.id);
      button.innerHTML = template;
      cell.parentNode.insertBefore(button, cell.nextSibling);
    }
  }

  function schedule() {
    window.setTimeout(function () {
      if (!document.querySelector("button.copybtn")) {
        attach();
      }
    }, 0);
  }

  function boot() {
    schedule();
    if (typeof document$ !== "undefined" && document$ && document$.subscribe) {
      document$.subscribe(schedule);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
