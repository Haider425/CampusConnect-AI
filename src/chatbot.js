import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Loader, MinusCircle } from 'lucide-react';
import './App.css'; 


const Chatbot = () => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([{ 
    sender: "bot", 
    text: "Hi! I'm your Test Centre assistant. How can I help you today?" 
  }]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [minimized, setMinimized] = useState(false);
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
  
    const userMessage = { sender: "user", text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);
  
    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });
  
      const data = await response.json();
      setMessages(prev => [...prev, { sender: "bot", text: data.reply }]);
    } catch (error) {
      setMessages(prev => [...prev, { 
        sender: "bot", 
        text: "Sorry, I'm having trouble connecting. Please try again." 
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="chatbot-container">
      {!open ? (
        <button 
          onClick={() => setOpen(true)} 
          className="chatbot-toggle-button"
        >
          <MessageCircle size={24} />
        </button>
      ) : (
        <div className="chatbot-window">
          {/* Header */}
          <div className="chatbot-header">
            <div className="header-content">
              <MessageCircle size={20} />
              <span className="header-title">Test Centre Assistant</span>
            </div>
            <div className="header-actions">
              <button 
                onClick={() => setMinimized(!minimized)} 
                className="header-button"
              >
                <MinusCircle size={20} />
              </button>
              <button 
                onClick={() => setOpen(false)} 
                className="header-button"
              >
                <X size={20} />
              </button>
            </div>
          </div>

          {/* Chat Container */}
          {!minimized && (
            <>
              <div className="chatbot-messages">
                <div className="messages-container">
                  {messages.map((msg, index) => (
                    <div
                      key={index}
                      className={`message ${msg.sender === "user" ? "user-message" : "bot-message"}`}
                    >
                      <div className="message-content">
                        {msg.text}
                      </div>
                    </div>
                  ))}
                  {isTyping && (
                    <div className="typing-indicator">
                      <div className="typing-content">
                        <Loader className="animate-spin" size={16} />
                        <span>Typing...</span>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              </div>
                  
              {/* Input Area */}
              <div className="chatbot-input-area">
                <div className="input-container">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask about the Test Centre..."
                    className="chatbot-input"
                  />
                  <button
                    onClick={sendMessage}
                    disabled={!input.trim()}
                    className={`send-button ${input.trim() ? "active" : "disabled"}`}
                  >
                    <Send size={20} />
                  </button>
                </div>
                <div className="input-hint">
                  Press Enter to send
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default Chatbot;