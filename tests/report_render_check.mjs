// Offline rendering check for web/app.js report functions.
// Usage: node tests/report_render_check.mjs <report.json> <repo-root>
// Stubs the DOM, loads the real app.js, renders a populated report, and
// exercises the source-excerpt fetch path. Prints a JSON verdict.
// No network, no paid calls; fetch is stubbed in-process.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import vm from "node:vm";

const [reportPath, repoRoot] = process.argv.slice(2, 4);
const report = JSON.parse(readFileSync(reportPath, "utf-8"));
const appSource = readFileSync(join(repoRoot, "web", "app.js"), "utf-8");

function makeElement() {
  return {
    value: "",
    innerHTML: "",
    textContent: "",
    disabled: false,
    title: "",
    dataset: {},
    addEventListener() {},
    appendChild() {},
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    querySelector: () => null,
    querySelectorAll: () => [],
    closest: () => null,
    setAttribute() {},
  };
}

const documentListeners = {};
const excerptFetches = [];
const fakeDocument = {
  getElementById: () => makeElement(),
  addEventListener: (type, listener) => {
    (documentListeners[type] = documentListeners[type] || []).push(listener);
  },
  createElement: () => makeElement(),
};
const fakeWindow = { addEventListener() {}, setInterval: () => 0, clearInterval() {}, location: { search: "" } };

async function fakeFetch(url) {
  excerptFetches.push(String(url));
  return {
    ok: true,
    async json() {
      return {
        available: true,
        commit: "a".repeat(40),
        file: "src/modules/navigator/rtl.cpp",
        start_line: 242,
        end_line: 243,
        lines: ["if (cone) {", "  climb();"],
        total_lines: 900,
      };
    },
  };
}

const context = vm.createContext({
  console,
  document: fakeDocument,
  window: fakeWindow,
  fetch: fakeFetch,
  setInterval: () => 0,
  clearInterval() {},
  URLSearchParams,
});
vm.runInContext(`${appSource}\n;globalThis.__check = { analysisReportHtml, renderHypothesis };`, context);
const { analysisReportHtml, renderHypothesis } = vm.runInContext("globalThis.__check", context);

const failures = [];
function check(name, condition) {
  if (!condition) failures.push(name);
}

const html = analysisReportHtml(report);
const [hypothesis] = report.ranked_hypotheses;
check("hypothesis-title", html.includes("Cone floor"));
check("mechanism-text", html.includes("safety floor"));
check("confidence-badge", html.includes("high"));
check("snippet-text", html.includes("_rtl_alt = calculate_return_alt"));
check("expression", html.includes("max(83.94 + 10.0, 83.94 + 2*10.0)"));
check("expected-expression", html.includes("103.94"));
check("variable", html.includes("destination_alt") && html.includes("setpoint triplet"));
check("unresolved-evidence", html.includes("Mission file absent"));
check("source-commit", html.includes("a".repeat(12)));
const excerptUrl = [...html.matchAll(/data-excerpt-url="([^"]+)"/g)].map((m) => m[1]);
check("excerpt-url-present", excerptUrl.length === 1);
check(
  "excerpt-url-params",
  excerptUrl.length === 1 &&
    excerptUrl[0].includes("commit=" + "a".repeat(40)) &&
    excerptUrl[0].includes("file=src%2Fmodules%2Fnavigator%2Frtl.cpp") &&
    excerptUrl[0].includes("start=242") &&
    excerptUrl[0].includes("end=250"),
);
check("plot-img", html.includes("/artifacts?path=%2Fplots%2Falt.png"));
check("plot-warning", html.includes("EAS vs TAS"));

// Exercise the excerpt click handler with a stub target element.
const clickListeners = (documentListeners["click"] || []).filter((listener) =>
  String(listener).includes("excerptUrl")
);
check("click-listener-registered", clickListeners.length === 1);
const spyTarget = { dataset: { excerptUrl: excerptUrl[0] }, textContent: "" };
const clickEvent = { target: { closest: () => spyTarget } };
await clickListeners[0](clickEvent);
check("excerpt-fetch-called", excerptFetches.length === 1);
check("excerpt-body-rendered", spyTarget.textContent.includes("climb();"));

// Second click must not refetch (loaded flag).
spyTarget.dataset.excerptLoaded = "1";
await clickListeners[0](clickEvent);
check("excerpt-fetch-once", excerptFetches.length === 1);

// Legacy report without source_commit degrades gracefully.
const legacy = JSON.parse(JSON.stringify(report));
delete legacy.source_commit;
const legacyHtml = renderHypothesis(legacy.ranked_hypotheses[0], 0, "");
check("legacy-no-commit-note", legacyHtml.includes("source snapshot unavailable"));
check("legacy-no-excerpt", !legacyHtml.includes("data-excerpt-url"));

// Empty plots render an explicit note, not silence.
check("empty-plots-note", renderHypothesis({ ...hypothesis, plots: [] }, 0, "a".repeat(40)).includes("No plots were generated"));

console.log(JSON.stringify({ failures, excerptFetches }));
if (failures.length > 0) process.exitCode = 1;
