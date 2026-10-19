/*
Draw the ModernDiD mark on the landing page in Fira Mono characters. A band of light sweeps across
it as the two cats blink now and then.
*/
(function () {
  if (window.__moderndidLandingMarkLoaded) {
    return;
  }
  window.__moderndidLandingMarkLoaded = true;

  // logo-mark.svg traced at 112 columns. Cells b and d fill the mark, digits 0 to 7 its edges, and h its outline.
  // Cells z and A to H are the cats' eyes. They fill in while the cats blink.
  var MAP = [
    "",
    "                           hhhhhhhhhhhhh                               hhhhhhhhhhhhh",
    "                     hhhhhh             hhhhhh                hhhhhhhhh             hhhhhh",
    "                  hhh    12345566666554321    hhh hhhh     hhh  1    12345566666554321    hhhh",
    "               hhh   246bbbbbbbbbbbbbbbbbbb642   h 1  hh hh   165356bbbbbbbbbbbbbbbbbbb6531   hh",
    "             hh   35bbbbbbbbbbbbbbbbbbbbbbbbbbb53  bb2  h  253bbbbbbbbbbbbbbbbbbbbbbbbbbbbb641  hh",
    "           hh  14bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbddd5bbb5   4bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb63  hh",
    "          h  15bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbdbdddbbbbbb36bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb3  hh",
    "         h  4bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbddbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb62  h",
    "        h 16bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbdddbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb4  h",
    "       h 1bbbbbbbbbbbbbbbbbbb5432222345bbbddbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb6  h",
    "      h 1bbbbbbbbbbbbbbbb631    hhhh    15bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb653221122356bbbbbbbbbbbbbbbbbb6  h",
    "     h  bbbbbbbbbbbbbbb62  hhhhh    hhh 1bbEDGbbbbbbbbbbbbbbbbbbbbbbb63    hhhhhh    25bbbbbbbbbbbbbbbb6  h",
    "     h 5bbbbbbbbbbbbbb3  hh         hh  4bbCFbbb655566bbbbbbbbbbbbb61  hhhh      hhhh  15bbbbbbbbbbbbbbb4 h",
    "    h 1bbbbbbbbbbbbbb2 hh          h  25bbbbbb66bbbbbbbbbbbbbbbbbb3  hh              hh  2bbbbbbbbbbbbbbb1 h",
    "    h 4bbbbbbbbbbbbb4 h            h 4bbbbbbbbb6655556bbbbbbbbbbb2 hh                  hh 1bbbbbbbbbbbbbb4 h",
    "    h bbbbbbbbbbbbbb  h            h  3bbbbbbbbbbbbbbb66bbbbbbbb4 h                      h 3bbbbbbbbbbbbbb h",
    "    h bbbbbbbbbbbbb6 h              hh 16bbbbbbbbbbbbbbbbbbbbbbb  hhhhhhh                 h bbbbbbbbbbbbbb h",
    "    h bbbbbbbbbbbbb6 h                h  23212333446bbbbbbbb64433321232  hh               h 5bbbbbbbbbbbbb1h",
    "    h bbbbbbbbbbbbbb  h                hhhhhhh  bbbbbbbbbbbbbbbbbbbbbbb61  h              h 5bbbbbbbbbbbbb1h",
    "    h 6bbbbbbbbbbbbb4 h                      h 4bbbbbbbb66bbbbbbbbbbbbbbb4  h             h bbbbbbbbbbbbbb h",
    "    h 3bbbbbbbbbbbbbb2 hh                  hh 2bbbbbbbbbbb6555566bbbbbbbbb4 h            h 3bbbbbbbbbbbbb5 h",
    "     h bbbbbbbbbbbbbbb3  hh              hh  3bbbbbbbbbbbbbbbbbb66bbbbbb52  h          hh 2bbbbbbbbbbbbbb2 h",
    "     h 3bbbbbbbbbbbbbbb51  hhhh      hhhh  15bbbbbbbbbbbbb665556bbbFCbb4  hh         hh  3bbbbbbbbbbbbbb5 h",
    "      h 5bbbbbbbbbbbbbbbb52    hhhhhh    35bbbbbbbbbbbbbbbbbbbbbbbGDEbb1 hhh    hhhhh  16bbbbbbbbbbbbbbb1 h",
    "      h  6bbbbbbbbbbbbbbbbbb643221122346bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb51    hhhh    136bbbbbbbbbbbbbbbb2 h",
    "       h  5bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbddbbb5432222345bbbbbbbbbbbbbbbbbbb2 h",
    "        h  4bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbdddbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb61 h",
    "         h  16bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbddbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb4  h",
    "          hh  36bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb63bbbbbbdddbdbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb51  h",
    "            hh  26bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb5   5bbb6dddbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb51  hh",
    "              hh  146bbbbbbbbbbbbbbbbbbbbbbbbbbbbb342  h  26b  25bbbbbbbbbbbbbbbbbbbbbbbbbbb63   hh",
    "                hh    356bbbbbbbbbbbbbbbbbbb653562   hh hh  2 h   246bbbbbbbbbbbbbbbbbbb642   hhh",
    "                  hhhh    12345566666554321       hhh     hhhh hhh    12345566666554321    hhh",
    "                      hhhhhh             hhhhhhhhh                hhhhhh             hhhhhh",
    "                            hhhhhhhhhhhhh                               hhhhhhhhhhhhh",
    "",
  ];
  var COLS = 112;
  var BODY = "*x%$@";
  var EDGE = "·~-=+*x%";
  var HALO = "+=x·+=";
  var teardown = null;

  function decode() {
    var cells = [];
    MAP.forEach(function (line) {
      for (var c = 0; c < COLS; c++) {
        var code = line.charAt(c) || " ";
        var cell = { kind: 0, edge: 0, dark: false, eye: false, phase: Math.random() };
        if (code === "b" || code === "d") {
          cell.kind = 1;
          cell.dark = code === "d";
        } else if (code >= "0" && code <= "7") {
          cell.kind = 2;
          cell.edge = +code;
        } else if (code === "h") {
          cell.kind = 3;
        } else if (code === "z") {
          cell.eye = true;
        } else if (code >= "A" && code <= "H") {
          cell.kind = 2;
          cell.edge = code.charCodeAt(0) - 65;
          cell.eye = true;
        }
        cells.push(cell);
      }
    });
    return cells;
  }

  // Rows span the mark plus an equal margin on both sides that fits its outline. Trimmed rows kept
  // only the left margin and pushed the mark right of center.
  function columns(cells) {
    var body = [COLS, 0];
    var all = [COLS, 0];
    cells.forEach(function (cell, i) {
      var c = i % COLS;
      if (cell.kind || cell.eye) {
        all = [Math.min(all[0], c), Math.max(all[1], c)];
        if (cell.kind !== 3) {
          body = [Math.min(body[0], c), Math.max(body[1], c)];
        }
      }
    });
    var margin = Math.max(body[0] - all[0], all[1] - body[1]);
    return [body[0] - margin, body[1] + 1 + margin];
  }

  function escape(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  }

  function setup() {
    if (teardown) {
      teardown();
      teardown = null;
    }
    var art = document.getElementById("mdid-mark");
    if (!art) {
      return;
    }
    var cells = decode();
    var rows = MAP.length;
    var span = columns(cells);
    var still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var timer = 0;
    var blinkUntil = 0;
    var nextBlink = performance.now() + 2400;

    function frame(now) {
      var seconds = now / 1000;
      if (now > nextBlink) {
        blinkUntil = now + 150;
        nextBlink = now + 5200 + Math.random() * 3600;
      }
      var blinking = !still && now < blinkUntil;
      var sweep = ((seconds % 9) / 9) * (COLS + rows * 2.2) - rows * 1.1;
      var html = "";
      for (var r = 0; r < rows; r++) {
        var row = "";
        var run = "";
        var current = "";
        for (var c = span[0]; c < span[1]; c++) {
          var cell = cells[r * COLS + c];
          var kind = blinking && cell.eye ? 1 : cell.kind;
          var glyph = " ";
          var cls = "";
          if (kind === 1) {
            var level = cell.dark ? 4 : 3;
            if (!still) {
              var distance = Math.abs(c + r * 2.2 - sweep);
              if (distance < 7) {
                level -= Math.round((1 - distance / 7) * 2.4);
              }
              if (Math.random() < 0.012) {
                level -= 1;
              }
            }
            glyph = BODY.charAt(Math.max(0, level));
            cls = "b";
          } else if (kind === 2) {
            glyph = EDGE.charAt(cell.edge);
            cls = "e";
          } else if (kind === 3) {
            var step = Math.floor(seconds * 1.6 + cell.phase * 6);
            glyph = still ? ((r + c) % 3 ? "+" : "=") : HALO.charAt(step % HALO.length);
            cls = "h";
          }
          if (cls !== current) {
            if (run) {
              row += current ? '<span class="' + current + '">' + escape(run) + "</span>" : run;
            }
            run = "";
            current = cls;
          }
          run += glyph;
        }
        if (run) {
          row += current ? '<span class="' + current + '">' + escape(run) + "</span>" : run;
        }
        html += (r ? "\n" : "") + row;
      }
      art.innerHTML = html;
    }

    function loop() {
      timer = 0;
      if (document.hidden) {
        return;
      }
      frame(performance.now());
      timer = window.setTimeout(loop, 90);
    }

    function resume() {
      if (!document.hidden && !timer) {
        loop();
      }
    }

    if (still) {
      frame(performance.now());
    } else {
      loop();
      document.addEventListener("visibilitychange", resume);
    }

    teardown = function () {
      if (timer) {
        window.clearTimeout(timer);
      }
      document.removeEventListener("visibilitychange", resume);
    };
  }

  function boot() {
    setup();
    // Since instant navigation swaps pages without a reload, set the mark up again on each new page.
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
