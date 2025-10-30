import React, { useState, useEffect } from "react";

export default function VoiceEmotion({ userId }) {
  const [emotion, setEmotion] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);

  useEffect(() => {
    if (!navigator.mediaDevices) {
      alert("Your browser does not support audio recording");
      return;
    }
  }, []);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    const chunks = [];

    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: recorder.mimeType });
      setAudioBlob(blob);
      console.log('Recording complete, MIME type:', recorder.mimeType);
    };

    recorder.start();
    setMediaRecorder(recorder);
    setRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorder) mediaRecorder.stop();
    setRecording(false);
  };

  const toBase64 = blob => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });

  // Convert audio blob to WAV format
  const convertToWav = async (audioBlob) => {
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    // Get audio data
    const numberOfChannels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length * numberOfChannels * 2;
    const sampleRate = audioBuffer.sampleRate;
    const buffer = new ArrayBuffer(44 + length);
    const view = new DataView(buffer);
    
    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numberOfChannels * 2, true); // byte rate
    view.setUint16(32, numberOfChannels * 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    writeString(36, 'data');
    view.setUint32(40, length, true);
    
    // Write audio data
    const channels = [];
    for (let i = 0; i < numberOfChannels; i++) {
      channels.push(audioBuffer.getChannelData(i));
    }
    
    let offset = 44;
    for (let i = 0; i < audioBuffer.length; i++) {
      for (let channel = 0; channel < numberOfChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, channels[channel][i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
      }
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
  };

  const predictEmotion = async () => {
    if (!audioBlob) return alert("Record your voice first!");
    setLoading(true);
    try {
      // Convert to WAV format
      console.log('Converting audio to WAV format...');
      const wavBlob = await convertToWav(audioBlob);
      console.log('WAV conversion complete');
      
      const audioBase64 = await toBase64(wavBlob);

      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId, audioData: audioBase64 }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Server error: ${response.status}`);
      }
      
      const data = await response.json();
      setEmotion(data.emotion);
      setConfidence(data.confidence);
    } catch (error) {
      alert(`Error: ${error.message}`);
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    // Check if it's an audio file
    if (!file.type.startsWith('audio/')) {
      alert('Please upload an audio file');
      return;
    }
    
    console.log('Uploaded file:', file.name, file.type);
    setAudioBlob(file);
  };

  const resetAll = () => {
    setAudioBlob(null);
    setEmotion("");
    setConfidence(0);
    setRecording(false);
    if (mediaRecorder) {
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
      setMediaRecorder(null);
    }
  };

  return (
    <div style={{ maxWidth: "600px", margin: "auto", padding: "30px", textAlign: "center", fontFamily: "Inter, sans-serif", background: "white", borderRadius: "16px", boxShadow: "0 4px 15px rgba(0,0,0,0.1)" }}>
      <h1 style={{ fontSize: "24px", fontWeight: "600", marginBottom: "20px" }}>🎙️ Voice Emotion Recognition</h1>

      {/* Recording Section */}
      <div style={{ marginBottom: "20px", padding: "15px", background: "#f8f9fa", borderRadius: "8px" }}>
        <h3 style={{ fontSize: "16px", marginBottom: "10px" }}>Record Audio</h3>
        <div>
          {!recording && <button onClick={startRecording} style={{ marginRight: "10px", padding: "10px 20px" }}>🎙️ Start Recording</button>}
          {recording && <button onClick={stopRecording} style={{ padding: "10px 20px" }}>⏹️ Stop Recording</button>}
        </div>
      </div>

      {/* File Upload Section */}
      <div style={{ marginBottom: "20px", padding: "15px", background: "#f8f9fa", borderRadius: "8px" }}>
        <h3 style={{ fontSize: "16px", marginBottom: "10px" }}>Or Upload Audio File</h3>
        <input 
          type="file" 
          accept="audio/*" 
          onChange={handleFileUpload}
          style={{ padding: "10px" }}
        />
      </div>

      {/* Audio Status */}
      {audioBlob && (
        <p style={{ color: "#28a745", marginBottom: "15px" }}>✓ Audio ready for analysis</p>
      )}

      {/* Results */}
      <div style={{ marginBottom: "20px", padding: "15px", background: "#f0f8ff", borderRadius: "8px" }}>
        {loading ? (
          <p style={{ fontSize: "18px" }}>Analyzing...</p>
        ) : (
          <>
            <p style={{ fontSize: "18px", marginBottom: "5px" }}>Detected Emotion: <strong style={{ color: "#007bff" }}>{emotion || "None"}</strong></p>
            <p style={{ fontSize: "16px" }}>Confidence: <strong>{(confidence * 100).toFixed(2)}%</strong></p>
          </>
        )}
      </div>

      {/* Action Buttons */}
      <div style={{ display: "flex", gap: "10px", justifyContent: "center" }}>
        <button 
          onClick={predictEmotion} 
          disabled={!audioBlob || loading}
          style={{ padding: "10px 25px", fontSize: "16px", background: "#007bff", color: "white", border: "none", borderRadius: "5px", cursor: audioBlob ? "pointer" : "not-allowed" }}>
          🎧 Analyze
        </button>
        <button 
          onClick={resetAll}
          style={{ padding: "10px 25px", fontSize: "16px", background: "#6c757d", color: "white", border: "none", borderRadius: "5px", cursor: "pointer" }}>
          🔄 Reset
        </button>
      </div>
    </div>
  );
}
