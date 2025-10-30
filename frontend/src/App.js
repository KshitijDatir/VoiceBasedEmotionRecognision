import React from "react";
import VoiceEmotion from "./components/VoiceEmotion";

function App() {
  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "linear-gradient(135deg, #dbeafe, #f3e8ff)",
      }}
    >
      <VoiceEmotion userId="user123" />
    </div>
  );
}

export default App;
