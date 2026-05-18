const form = document.getElementById("search-form");
const input = document.getElementById("image-input");
const preview = document.getElementById("preview");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const template = document.getElementById("result-template");

const IMAGE_EXTENSIONS = ["", ".jpg", ".jpeg", ".png", ".webp"];

function setStatus(message) {
  statusEl.textContent = message;
}

function setPreview(file) {
  if (!file) {
    preview.removeAttribute("src");
    return;
  }
  preview.src = URL.createObjectURL(file);
}

function scoreLabel(score) {
  return `${(score * 100).toFixed(2)}% match`;
}

function imageCandidates(src) {
  return IMAGE_EXTENSIONS.map((ext) => `/images/${src}${ext}`);
}

function attachImageWithFallback(imgEl, src) {
  const candidates = imageCandidates(src);
  let i = 0;

  function loadNext() {
    if (i >= candidates.length) {
      imgEl.alt = `Unable to load ${src}`;
      imgEl.removeAttribute("src");
      return;
    }
    imgEl.src = candidates[i];
    i += 1;
  }

  imgEl.onerror = loadNext;
  loadNext();
}

function renderResults(scores) {
  resultsEl.innerHTML = "";

  if (scores.length < 1) {
    setStatus("No visually similar matches found above threshold.");
    return;
  }

  const maxResults = 24;
  scores.slice(0, maxResults).forEach((result, index) => {
    const node = template.content.firstElementChild.cloneNode(true);
    const image = node.querySelector(".result-image");
    const score = node.querySelector(".result-score");
    const src = node.querySelector(".result-src");

    score.textContent = `Score: ${result.score.toFixed(2)}`;
    src.textContent = result.src;
    node.style.animationDelay = `${index * 0.03}s`;
    attachImageWithFallback(image, result.src);

    resultsEl.appendChild(node);
  });

  setStatus(`Showing ${Math.min(scores.length, maxResults)} of ${scores.length} matches.`);
}

input.addEventListener("change", () => {
  const [file] = input.files || [];
  setPreview(file);
});

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

  try {
    const response = await fetch("/search", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Search failed (${response.status})`);
    }

    const data = await response.json();
    renderResults(data.scores || []);
  } catch (error) {
    setStatus(error.message || "Something went wrong while searching.");
  }
});
