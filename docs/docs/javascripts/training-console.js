/* The console remains selectable and readable when JavaScript is unavailable. */
function prepareTrainingConsoles() {
  document.querySelectorAll("[data-console-copy]").forEach((button) => {
    if (button.dataset.ready) return;
    button.dataset.ready = "true";
    if (!navigator.clipboard || !navigator.clipboard.writeText) return;
    button.hidden = false;
    button.addEventListener("click", async () => {
      const panel = button.closest(".training-console");
      const status = panel.querySelector(".training-console__status");
      const label = button.querySelector("[data-console-copy-label]");
      try {
        await navigator.clipboard.writeText(panel.querySelector("pre code").textContent);
        label.textContent = "Copied";
        status.textContent = "Training output copied.";
      } catch {
        label.textContent = "Copy failed";
        status.textContent = "Copy was unavailable. Select the output and copy it manually.";
      }
      window.setTimeout(() => { label.textContent = "Copy"; }, 2500);
    });
  });
}
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", prepareTrainingConsoles);
} else {
  prepareTrainingConsoles();
}
// Also support Material's instant navigation if it is enabled later.
if (typeof document$ !== "undefined") document$.subscribe(prepareTrainingConsoles);
