document.querySelectorAll('form').forEach(form => {
  form.addEventListener('submit', () => {
    const button = form.querySelector('button[type="submit"]');
    if (button) { button.dataset.original = button.textContent; button.disabled = true; button.textContent = 'Working…'; }
  });
});
window.addEventListener('pageshow', () => {
  document.querySelectorAll('button[data-original]').forEach(button => { button.disabled = false; button.textContent = button.dataset.original; });
});
const audioForm = document.querySelector('#audio-form');
if (audioForm) {
  const start = document.querySelector('#record-start');
  const stop = document.querySelector('#record-stop');
  const submit = document.querySelector('#audio-submit');
  const status = document.querySelector('#record-status');
  const preview = document.querySelector('#record-preview');
  let recorder, stream, timer, previewUrl;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    start.disabled = true;
    status.textContent = 'Recording is unavailable in this browser. Upload an audio file instead.';
  }
  start.addEventListener('click', async () => {
    start.disabled = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio: true});
      const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'].find(t => MediaRecorder.isTypeSupported(t));
      recorder = mime ? new MediaRecorder(stream, {mimeType: mime}) : new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        clearTimeout(timer);
        stream.getTracks().forEach(track => track.stop());
        stop.disabled = true; start.disabled = false; submit.disabled = false;
        if (!chunks.length) { status.textContent = 'The recording was empty. Please try again.'; return; }
        const type = recorder.mimeType || chunks[0].type;
        const extension = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : 'webm';
        const blob = new Blob(chunks, {type});
        const transfer = new DataTransfer();
        transfer.items.add(new File([blob], 'microphone.' + extension, {type}));
        document.querySelector('#audio-file').files = transfer.files;
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = URL.createObjectURL(blob);
        preview.src = previewUrl; preview.hidden = false;
        status.textContent = 'Recording ready. Listen to it, then click Transcribe and find.';
      };
      recorder.start();
      stop.disabled = false; submit.disabled = true;
      status.textContent = 'Recording… Click Stop when finished.';
      timer = setTimeout(() => { if (recorder.state === 'recording') recorder.stop(); }, Number(audioForm.dataset.maxSeconds) * 1000);
    } catch (error) {
      stream?.getTracks().forEach(track => track.stop());
      start.disabled = false; submit.disabled = false;
      status.textContent = 'Microphone could not start: ' + error.message + '. You can upload a recording.';
    }
  });
  stop.addEventListener('click', () => { if (recorder?.state === 'recording') recorder.stop(); });
  window.addEventListener('pagehide', () => { stream?.getTracks().forEach(track => track.stop()); clearTimeout(timer); });
}
