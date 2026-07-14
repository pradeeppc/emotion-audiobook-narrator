// Empty string = "same origin as this page". Works both when the backend
// serves this file directly (deployed, and `uvicorn main:app` locally at
// http://127.0.0.1:8000/) and is only wrong if you open this file straight
// from disk (file://) without running the backend -- see README.md.
const API_BASE = "";

const textInput = document.getElementById("text-input");
const statusEl = document.getElementById("status");
const analyzeBtn = document.getElementById("analyze-btn");
const narrateBtn = document.getElementById("narrate-btn");
const table = document.getElementById("emotion-table");
const tableBody = table.querySelector("tbody");
const player = document.getElementById("player");

function setStatus(msg) {
  statusEl.textContent = msg;
}

analyzeBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) return setStatus("Paste some text first.");

  setStatus("Analyzing emotions...");
  table.hidden = true;

  try {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    tableBody.innerHTML = "";
    for (const s of data.sentences) {
      const row = document.createElement("tr");
      row.innerHTML = `<td>${s.text}</td><td>${s.emotion} <span class="family">(${s.family})</span></td><td>${s.confidence}</td>`;
      tableBody.appendChild(row);
    }
    table.hidden = false;
    setStatus(`Analyzed ${data.sentences.length} sentence(s).`);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
});

narrateBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) return setStatus("Paste some text first.");

  setStatus("Generating narration... this can take a while on CPU.");
  player.hidden = true;

  try {
    const res = await fetch(`${API_BASE}/narrate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(await res.text());

    const blob = await res.blob();
    player.src = URL.createObjectURL(blob);
    player.hidden = false;
    setStatus("Done. Press play below.");
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
});
