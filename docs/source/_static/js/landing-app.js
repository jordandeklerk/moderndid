/*
Wire the landing page's prompt and start menu. Focusing the prompt hands the reader to the theme's
search field. Pressing a key from the menu opens its page. The j and k keys or the arrow keys move the
menu's cursor row for Enter to open.
*/
(function () {
  if (window.__moderndidLandingAppLoaded) {
    return;
  }
  window.__moderndidLandingAppLoaded = true;

  // Since instant navigation swaps pages without a full load, each page's timer starts at the click or history step behind it.
  var started = 0;
  window.addEventListener("click", function () {
    started = performance.now();
  }, true);
  window.addEventListener("popstate", function () {
    started = performance.now();
  }, true);

  function menuLinks() {
    return [].slice.call(document.querySelectorAll(".mdid-menu a[data-key]"));
  }

  function select(index) {
    var links = menuLinks();
    var chosen = (index + links.length) % links.length;
    links.forEach(function (link, i) {
      link.classList.toggle("is-selected", i === chosen);
    });
  }

  function setupPrompt() {
    var prompt = document.querySelector(".mdid-prompt__input");
    var toggle = document.querySelector('input[data-md-toggle="search"]');
    var field = document.querySelector(".md-search__input");
    if (!prompt || !toggle || !field || prompt.dataset.wired) {
      return;
    }
    prompt.dataset.wired = "true";
    prompt.addEventListener("focus", function () {
      toggle.checked = true;
      toggle.dispatchEvent(new Event("change"));
      field.focus();
    });
  }

  function setupMenu() {
    var menu = document.querySelector(".mdid-menu");
    if (!menu || menu.dataset.wired) {
      return;
    }
    menu.dataset.wired = "true";
    menuLinks().forEach(function (link, i) {
      link.addEventListener("mousemove", function () {
        select(i);
      });
      link.addEventListener("focus", function () {
        select(i);
      });
    });
    select(0);
    var footer = menu.querySelector(".mdid-menu__footer");
    if (footer) {
      footer.querySelector(".mdid-menu__time").textContent = (performance.now() - started).toFixed(2) + "ms";
      footer.hidden = false;
    }
  }

  // Since instant navigation keeps this listener across pages, it looks the menu up on each key press.
  document.addEventListener("keydown", function (event) {
    var target = event.target;
    var search = document.querySelector('input[data-md-toggle="search"]');
    if (event.metaKey || event.ctrlKey || event.altKey || (search && search.checked)) {
      return;
    }
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
      return;
    }
    var links = menuLinks();
    if (!links.length) {
      return;
    }
    var current = 0;
    links.forEach(function (link, i) {
      if (link.classList.contains("is-selected")) {
        current = i;
      }
    });
    // The arrow keys move the cursor row only while the page fits the window. A page that scrolls keeps them for scrolling.
    var fits = document.documentElement.scrollHeight <= window.innerHeight;
    if (event.key === "j" || (fits && event.key === "ArrowDown")) {
      event.preventDefault();
      select(current + 1);
      return;
    }
    if (event.key === "k" || (fits && event.key === "ArrowUp")) {
      event.preventDefault();
      select(current - 1);
      return;
    }
    if (event.repeat) {
      return;
    }
    var link = event.key === "Enter" && target === document.body ? links[current] : links.filter(function (item) {
      return item.dataset.key === event.key;
    })[0];
    if (link) {
      event.preventDefault();
      select(links.indexOf(link));
      // A click lets the theme's instant navigation load the page.
      link.click();
    }
  });

  function boot() {
    setupPrompt();
    setupMenu();
    // Since instant navigation swaps pages without a reload, set the prompt and menu up again on each new page.
    if (typeof document$ !== "undefined" && document$ && document$.subscribe) {
      document$.subscribe(function () {
        setupPrompt();
        setupMenu();
      });
    }
  }

  if (document.readyState !== "loading") {
    boot();
  } else {
    document.addEventListener("DOMContentLoaded", boot);
  }
})();
