"use strict";
/* Click-to-zoom on the figure plates and a copy button for the BibTeX entry.
   Both degrade to plain content if this file never loads. */

const zoom = document.getElementById('zoom');
const zoomImg = document.getElementById('zoomImg');

document.querySelectorAll('.figbox img').forEach(img => {
  img.addEventListener('click', () => {
    zoomImg.src = img.currentSrc || img.src;
    zoomImg.alt = img.alt;
    zoom.showModal();
  });
});
if (zoom){
  zoom.addEventListener('click', () => zoom.close());
  zoom.addEventListener('close', () => { zoomImg.src = ''; });
}

const copyBtn = document.getElementById('copyBib');
if (copyBtn){
  copyBtn.addEventListener('click', async () => {
    const text = document.getElementById('bibtex').textContent;
    try { await navigator.clipboard.writeText(text); }
    catch { return; }
    const was = copyBtn.textContent;
    copyBtn.textContent = 'Copied';
    setTimeout(() => { copyBtn.textContent = was; }, 1400);
  });
}
