/*
Trace the "On this page" outline with a hairline that curves in and out with its indentation, and slide a
solid thumb along it over the sections on screen.
*/
(function () {
  if (window.__moderndidTocRailLoaded) {
    return;
  }
  window.__moderndidTocRailLoaded = true;

  var NS = "http://www.w3.org/2000/svg";
  var teardown = null;

  function make(name, attributes) {
    var element = document.createElementNS(NS, name);
    Object.keys(attributes || {}).forEach(function (key) {
      element.setAttribute(key, attributes[key]);
    });
    return element;
  }

  function clamp(value, low, high) {
    return Math.min(Math.max(value, low), high);
  }

  function setup() {
    if (teardown) {
      teardown();
      teardown = null;
    }
    var nav = document.querySelector(".md-sidebar--secondary .md-nav--secondary");
    var list = nav && nav.querySelector('[data-md-component="toc"]');
    if (!list) {
      return;
    }
    // Instant navigation rewrites the outline's links to full URLs, so each target comes from the link's hash.
    var entries = Array.prototype.slice
      .call(list.querySelectorAll("a.md-nav__link"))
      .map(function (link) {
        return {
          link: link,
          text: link.querySelector(".md-ellipsis") || link,
          target: link.hash ? document.getElementById(decodeURIComponent(link.hash.slice(1))) : null,
        };
      })
      .filter(function (entry) {
        return entry.target;
      });
    if (entries.length < 2) {
      return;
    }
    var still = window.matchMedia("(prefers-reduced-motion: reduce)");

    nav.classList.add("mdid-rail-on");
    var svg = make("svg", { class: "mdid-rail", "aria-hidden": "true" });
    var track = make("path", { class: "mdid-rail__track" });
    var thumb = make("path", { class: "mdid-rail__thumb" });
    svg.appendChild(track);
    svg.appendChild(thumb);
    nav.insertBefore(svg, nav.firstChild);

    // A small ring beside the outline's title fills as you read the whole page.
    var title = nav.querySelector(".md-nav__title");
    var ring = make("svg", { class: "mdid-ring", viewBox: "0 0 16 16", "aria-hidden": "true" });
    var arc = make("circle", {
      class: "mdid-ring__arc",
      cx: 8,
      cy: 8,
      r: 6.5,
      pathLength: 100,
      "stroke-dasharray": 100,
      "stroke-dashoffset": 100,
      transform: "rotate(-90 8 8)",
    });
    ring.appendChild(make("circle", { class: "mdid-ring__track", cx: 8, cy: 8, r: 6.5 }));
    ring.appendChild(arc);
    if (title) {
      title.appendChild(ring);
    }

    var total = 0;
    var heights = [];
    var shown = { from: 0, to: 0, share: 0 };
    var goal = { from: 0, to: 0, share: 0 };
    var state = "";
    var frame = 0;
    var last = 0;

    // The line runs just left of each entry's text and curves between indentation levels.
    function layout() {
      var box = nav.getBoundingClientRect();
      var path = "";
      var previous = null;
      entries.forEach(function (entry, index) {
        var row = entry.link.getBoundingClientRect();
        var x = entry.text.getBoundingClientRect().left - box.left - 10;
        var inset = Math.min(7, (row.height - 2) / 2);
        var top = row.top - box.top + inset;
        var bottom = row.bottom - box.top - inset;
        entry.top = top;
        entry.bottom = bottom;
        if (!previous) {
          path = "M" + x + " " + top;
        } else if (previous.x !== x) {
          var middle = (previous.bottom + top) / 2;
          path += " C" + previous.x + " " + middle + " " + x + " " + middle + " " + x + " " + top;
        }
        path += " L" + x + " " + bottom;
        previous = { x: x, bottom: bottom };
      });
      track.setAttribute("d", path);
      thumb.setAttribute("d", path);
      svg.setAttribute("height", nav.scrollHeight);
      svg.setAttribute("width", box.width);

      // Sample the path once so a height on the outline converts to a distance along the line.
      total = thumb.getTotalLength();
      heights = [];
      for (var length = 0; length <= total; length += 1) {
        heights.push(thumb.getPointAtLength(length).y);
      }
      thumb.style.strokeDasharray = "0 " + (total + 10);
      measure();
      shown.from = goal.from;
      shown.to = goal.to;
      shown.share = goal.share;
      draw();
    }

    function along(y) {
      var low = 0;
      var high = heights.length - 1;
      if (y <= heights[0]) {
        return 0;
      }
      if (y >= heights[high]) {
        return total;
      }
      while (high - low > 1) {
        var middle = (low + high) >> 1;
        if (heights[middle] < y) {
          low = middle;
        } else {
          high = middle;
        }
      }
      return low + (y - heights[low]) / Math.max(heights[high] - heights[low], 1e-6);
    }

    function measure() {
      var page = document.documentElement;
      var header = document.querySelector(".md-header");
      var ceiling = header ? Math.max(header.getBoundingClientRect().bottom, 0) : 0;
      var tops = entries.map(function (entry) {
        return entry.target.getBoundingClientRect().top;
      });
      var foot = page.scrollHeight - window.scrollY;
      var reading = ceiling + (window.innerHeight - ceiling) * 0.3;
      var bottom = window.scrollY + window.innerHeight >= page.scrollHeight - 2;
      var current = -1;
      tops.forEach(function (top, index) {
        if (top <= reading) {
          current = index;
        }
      });
      if (bottom) {
        current = entries.length - 1;
      }
      var visible = entries.map(function (entry, index) {
        var next = index + 1 < entries.length ? tops[index + 1] : foot;
        return next > ceiling && tops[index] < window.innerHeight;
      });
      var first = visible.indexOf(true);
      var final = visible.lastIndexOf(true);

      // The thumb spans the rows of every section on screen and stops where their straight runs end.
      goal.from = first >= 0 ? along(entries[first].top) : 0;
      goal.to = final >= 0 ? along(entries[final].bottom) : goal.from;
      goal.share = clamp(window.scrollY / Math.max(page.scrollHeight - window.innerHeight, 1), 0, 1);

      var labels = entries
        .map(function (entry, index) {
          return index === current ? "c" : visible[index] ? "v" : "-";
        })
        .join("");
      if (labels !== state) {
        entries.forEach(function (entry, index) {
          entry.link.classList.toggle("mdid-current", labels[index] === "c");
          entry.link.classList.toggle("mdid-visible", labels[index] !== "-");
        });
        state = labels;
      }
    }

    function draw() {
      var span = Math.max(shown.to - shown.from, 0);
      thumb.style.strokeDasharray = span.toFixed(2) + " " + (total + 10);
      thumb.style.strokeDashoffset = (-shown.from).toFixed(2);
      thumb.style.opacity = span > 1 ? "" : "0";
      arc.setAttribute("stroke-dashoffset", (100 - 100 * shown.share).toFixed(2));
    }

    // Ease toward each goal so the thumb settles into place instead of jumping.
    function tick(time) {
      frame = 0;
      var step = last ? Math.min(time - last, 64) : 16;
      last = time;
      var blend = still.matches ? 1 : 1 - Math.exp(-step / 110);
      var moving = false;
      Object.keys(shown).forEach(function (key) {
        var gap = goal[key] - shown[key];
        shown[key] = Math.abs(gap) < 0.05 ? goal[key] : shown[key] + gap * blend;
        moving = moving || shown[key] !== goal[key];
      });
      draw();
      if (moving) {
        frame = window.requestAnimationFrame(tick);
      } else {
        last = 0;
      }
    }

    function schedule() {
      measure();
      if (!frame) {
        frame = window.requestAnimationFrame(tick);
      }
    }

    var resize = new ResizeObserver(layout);
    resize.observe(nav);
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", layout);
    layout();

    teardown = function () {
      resize.disconnect();
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", layout);
      if (frame) {
        window.cancelAnimationFrame(frame);
      }
      svg.remove();
      ring.remove();
      nav.classList.remove("mdid-rail-on");
      entries.forEach(function (entry) {
        entry.link.classList.remove("mdid-current", "mdid-visible");
      });
    };
  }

  function boot() {
    setup();
    // Instant navigation swaps the page, so rebuild the outline for each new page.
    if (typeof document$ !== "undefined" && document$ && document$.subscribe) {
      document$.subscribe(setup);
    }
  }

  if (document.readyState !== "loading") {
    boot();
  } else {
    document.addEventListener("DOMContentLoaded", boot);
  }
})();
