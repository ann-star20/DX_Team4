import { useState, useRef, useEffect } from 'react';
import { useAppStore } from '../store/useAppStore';
import { useServiceLoop } from '../hooks/useServiceLoop';
import { useVoice } from '../hooks/useVoice';
import { useKeyboardNav } from '../hooks/useKeyboardNav';
import './ConversationPanel.css';

interface ConversationPanelProps {
  onClose: () => void;
  onShowRecommend: () => void;
}

export function ConversationPanel({ onClose, onShowRecommend }: ConversationPanelProps) {
  const { chatHistory, addChatMessage } = useAppStore();
  const { sendMessage, analyzeEmotion, requestRecommend } = useServiceLoop();
  const { isSTTSupported, startListening, speak } = useVoice();
  const [input, setInput] = useState('');
  useKeyboardNav({ onBack: onClose });
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setInput('');
    setLoading(true);
    try {
      await analyzeEmotion(text);
      const reply = await sendMessage(text);
      speak(reply);

      if (
        reply.includes('추천') ||
        reply.includes('어떠세요') ||
        reply.includes('어때요')
      ) {
        await requestRecommend();
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRecommendClick = () => {
    addChatMessage({ role: 'user', content: '볼거 추천해줘' });
    requestRecommend();
    onShowRecommend();
  };

  return (
    <div className="conversation-panel">
      <div className="conversation-avatar">
        <div className="avatar-circle">마음</div>
      </div>
      <div className="conversation-content">
        <div className="conversation-header">
          <h3>마음이음과 대화하기</h3>
          <button type="button" className="btn-close" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="conversation-messages">
          {chatHistory.length === 0 && (
            <div className="message assistant">
              <p>안녕하세요! 오늘 하루 어땠어요? 편하게 이야기해 주세요.</p>
            </div>
          )}
          {chatHistory.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <p>{msg.content}</p>
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <p className="typing">...</p>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <div className="conversation-actions">
          <button
            type="button"
            className="btn-quick"
            onClick={handleRecommendClick}
          >
            볼거 추천해줘
          </button>
        </div>
        <form onSubmit={handleSubmit} className="conversation-form">
          {isSTTSupported && (
            <button
              type="button"
              className="btn-mic"
              onClick={() =>
                startListening((t) => {
                  setInput((prev) => (prev ? `${prev} ${t}` : t));
                })
              }
              disabled={loading}
              title="음성 입력"
            >
              🎤
            </button>
          )}
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={isSTTSupported ? '말하거나 입력하세요...' : '메시지를 입력하세요...'}
            disabled={loading}
            autoFocus
          />
          <button type="submit" disabled={loading || !input.trim()}>
            전송
          </button>
        </form>
      </div>
    </div>
  );
}
