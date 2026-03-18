import { useState, useCallback, useRef } from 'react';

const SpeechRecognitionAPI =
  typeof window !== 'undefined'
    ? (window.SpeechRecognition || (window as unknown as { webkitSpeechRecognition?: new () => SpeechRecognition }).webkitSpeechRecognition)
    : null;

/**
 * Web Speech API 기반 STT/TTS
 * 미지원 시 텍스트 입력 폴백 사용
 */
export function useVoice() {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  const isSTTSupported = !!SpeechRecognitionAPI;
  const isTTSSupported =
    typeof window !== 'undefined' && 'speechSynthesis' in window;

  const startListening = useCallback((onResult: (text: string) => void) => {
    if (!SpeechRecognitionAPI) return;
    const rec = new SpeechRecognitionAPI();
    rec.continuous = false;
    rec.lang = 'ko-KR';
    rec.onresult = (e: SpeechRecognitionEvent) => {
      const text = e.results[0][0].transcript;
      onResult(text);
    };
    rec.onend = () => setIsListening(false);
    recognitionRef.current = rec;
    rec.start();
    setIsListening(true);
  }, []);

  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
    }
  }, [isListening]);

  const speak = useCallback(
    (text: string) => {
      if (!isTTSSupported) return;
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'ko-KR';
      utterance.onend = () => setIsSpeaking(false);
      speechSynthesis.speak(utterance);
      setIsSpeaking(true);
    },
    [isTTSSupported]
  );

  return {
    isSTTSupported,
    isTTSSupported,
    isListening,
    isSpeaking,
    startListening,
    stopListening,
    speak,
  };
}
