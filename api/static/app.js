const form = document.getElementById("search-form");
const input = document.getElementById("image-input");
const preview = document.getElementById("preview");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const template = document.getElementById("result-template");
const seeMoreButton = document.getElementById("see-more")
const timer = document.getElementById("time")
const modeInputs = document.querySelectorAll('input[name="search-mode"]')
const knnSlider = document.getElementById("knn-k")
const knnValue = document.getElementById("knn-k-value")
const knnControl = document.querySelector(".k-control")


let state = {}

seeMoreButton.addEventListener("click", () => {
  seeMoreButton.hidden = true
  renderResults(state, state.scores.length)
})

function setStatus(message) {
  statusEl.hidden = false
  statusEl.textContent = message;
}

function setTimer(time) {
  timer.hidden = false
  timer.textContent = time
}

function setPreview(file) {
  if (!file) {
    preview.hidden = true
    preview.removeAttribute("src");
    return;
  }
  preview.hidden = false
  preview.src = URL.createObjectURL(file);
}

function attachImageWithFallback(imgEl, src) {
  function imageErr() {
    imgEl.alt = `Unable to load ${src}`;
    imgEl.removeAttribute("src");
  }

  imgEl.src = `${src}`;
  imgEl.onerror = imageErr;
}

function renderResults(data, maxResults) {
  resultsEl.innerHTML = "";

  if (data.scores.length < 1) {
    setTimer(`Query Time: ${(data.time*1000).toFixed(2)} ms`)
    setStatus("No visually similar matches found above threshold.");
    return;
  }

  data.scores.slice(0, maxResults).forEach((result, index) => {
    const node = template.content.firstElementChild.cloneNode(true);
    const image = node.querySelector(".result-image");
    const score = node.querySelector(".result-score");
    const openButton = node.querySelector(".result-open");

    score.textContent = `Score: ${result.score.toFixed(2)}`;
    node.style.animationDelay = `${index * 0.03}s`;
    attachImageWithFallback(image, result.src);

    openButton?.addEventListener("click", (event) => {
      event.preventDefault();
      openButton.blur();
      window.open(result.src, "_blank", "noopener,noreferrer");
    });

    resultsEl.appendChild(node);
  });

  setStatus(`Showing ${Math.min(data.scores.length, maxResults)} of ${data.scores.length} matches.`);
  setTimer(`Query Time: ${(data.time*1000).toFixed(2)} ms`)
}

input.addEventListener("change", () => {
  const [file] = input.files || [];
  setPreview(file);
});

function syncKnnValue() {
  if (!knnSlider || !knnValue) return;
  knnValue.textContent = knnSlider.value;
}

syncKnnValue();
knnSlider?.addEventListener("input", syncKnnValue);

function syncKnnVisibility() {
  if (!knnControl) return;
  const selectedMode = Array.from(modeInputs).find((input) => input.checked)?.value;
  knnControl.hidden = selectedMode !== "hnsw";
}

syncKnnVisibility();
modeInputs.forEach((input) => input.addEventListener("change", syncKnnVisibility));

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const [file] = input.files || [];
  if (!file) {
    setStatus("Choose an image first.");
    return;
  }

  setStatus("Searching...");
  resultsEl.innerHTML = "";

  const formData = new FormData();
  formData.append("file", file);
  const mode = Array.from(modeInputs).find((input) => input.checked)?.value || "cosine";
  const kValue = knnSlider?.value ? Number.parseInt(knnSlider.value, 10) : 10;

  try {
    const response = await fetch(`/search?mode=${encodeURIComponent(mode)}&k=${encodeURIComponent(kValue)}`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Search failed (${response.status})`);
    }

    console.log(response)

    const data = await response.json();
    state = data
    console.log(data)
    renderResults(data || [], 5);
    seeMoreButton.hidden = false
  } catch (error) {
    setStatus(error.message || "Something went wrong while searching.");
  }
});
