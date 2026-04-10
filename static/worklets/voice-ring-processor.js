/**
 * 2초 링 버퍼 + 녹음 시 청크 전달 + RMS(볼륨 미터용)
 * sampleRate는 AudioContext에 따름(보통 48000)
 */
class VoiceRingProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._ringSize = 0;
    this._ring = null;
    this._write = 0;
    this._recording = false;
    this.port.onmessage = (ev) => {
      const d = ev.data || {};
      if (d.type === 'init') {
        const sec = typeof d.seconds === 'number' ? d.seconds : 2;
        this._ringSize = Math.max(128, Math.floor(sampleRate * sec));
        this._ring = new Float32Array(this._ringSize);
        this._write = 0;
      }
      if (d.type === 'start-record') {
        this._recording = true;
        const pre = this._exportPreroll();
        const preCopy = new Float32Array(pre);
        this.port.postMessage({ type: 'preroll', f32: preCopy }, [preCopy.buffer]);
      }
      if (d.type === 'stop-record') {
        this._recording = false;
      }
    };
  }

  _exportPreroll() {
    if (!this._ring || !this._ringSize) return new Float32Array(0);
    const out = new Float32Array(this._ringSize);
    if (this._write < this._ringSize) {
      for (let i = 0; i < this._write; i++) out[i] = this._ring[i];
      return out.subarray(0, this._write);
    }
    const start = this._write % this._ringSize;
    for (let i = 0; i < this._ringSize; i++) {
      out[i] = this._ring[(start + i) % this._ringSize];
    }
    return out;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || !ch.length) return true;

    if (this._ring && this._ringSize) {
      for (let i = 0; i < ch.length; i++) {
        this._ring[this._write % this._ringSize] = ch[i];
        this._write++;
      }
    }

    let sum = 0;
    for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i];
    const rms = Math.sqrt(sum / ch.length);
    this.port.postMessage({ type: 'rms', r: rms });

    if (this._recording) {
      const copy = new Float32Array(ch);
      this.port.postMessage({ type: 'chunk', f32: copy }, [copy.buffer]);
    }

    return true;
  }
}

registerProcessor('voice-ring-processor', VoiceRingProcessor);
