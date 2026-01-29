import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Bot, User } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const mockResponses = [
  "Based on your location preferences, I'd recommend **Bella Italia** in downtown Philadelphia - it has a 4.5 star rating and is known for authentic pasta dishes!",
  "For a quick bite, try **Joe's Diner** on Market Street. It's popular among locals for breakfast and has great reviews for its pancakes!",
  "If you're looking for something upscale, **The Capital Grille** in Center City has excellent steaks and a sophisticated atmosphere. Perfect for special occasions!",
  "Want something casual? **Shake Shack** at Rittenhouse Square is always a hit - great burgers and the outdoor seating is lovely in good weather.",
  "For authentic Mexican, check out **El Vez** - their tacos and margaritas are top-rated in the city!",
  "I noticed you prefer Italian cuisine! **Osteria** by Marc Vetri is a James Beard Award winner - definitely worth the visit.",
  "Looking for vegetarian options? **HipCityVeg** has amazing plant-based burgers and smoothies. Very popular with the health-conscious crowd!",
];

const ChatbotWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hi! 👋 I'm your Restaurant Recommender. Tell me what kind of food you're in the mood for, and I'll suggest some great places!",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

//   const handleSend = () => {
//     if (!inputValue.trim()) return;

//     const userMessage: Message = {
//       id: Date.now().toString(),
//       role: 'user',
//       content: inputValue,
//       timestamp: new Date(),
//     };

//     setMessages((prev) => [...prev, userMessage]);
//     setInputValue('');
//     setIsTyping(true);

//     // Simulate AI response
//     setTimeout(() => {
//       const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];
//       const assistantMessage: Message = {
//         id: (Date.now() + 1).toString(),
//         role: 'assistant',
//         content: randomResponse,
//         timestamp: new Date(),
//       };
//       setMessages((prev) => [...prev, assistantMessage]);
//       setIsTyping(false);
//     }, 1000 + Math.random() * 1000);
//   };

  const handleSend = async () => {
    if (!inputValue.trim()) return;

    // 1. Add User Message to UI immediately
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    try {
      // 2. Call your Python Backend
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            message: userMessage.content,
            user_id: "---zemaUC8WeJeWKqS6p9Q" // <--- Replace with dynamic ID or one from your CSV
        }),
      });

      const data = await response.json();

      // 3. Add AI Response to UI
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.reply, // The text from Gemini
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error chatting with backend:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "Sorry, I'm having trouble connecting to the restaurant database right now.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      {/* Floating Button */}
      <button
        onClick={() => setIsOpen(true)}
        className={`fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full flex items-center justify-center transition-all duration-300 hover:scale-110 ${
          isOpen ? 'opacity-0 pointer-events-none scale-90' : 'opacity-100'
        }`}
        style={{
          background: 'linear-gradient(135deg, hsl(270 70% 55%), hsl(290 65% 60%))',
          boxShadow: '0 4px 20px hsl(270 70% 55% / 0.4)',
        }}
        aria-label="Open Restaurant Recommender"
      >
        <MessageCircle className="w-6 h-6 text-white" />
      </button>

      {/* Chat Window */}
      <div
        className={`fixed bottom-6 right-6 z-50 w-[380px] h-[520px] rounded-2xl overflow-hidden flex flex-col transition-all duration-300 ${
          isOpen
            ? 'opacity-100 scale-100 translate-y-0'
            : 'opacity-0 scale-95 translate-y-4 pointer-events-none'
        }`}
        style={{
          boxShadow: '0 16px 48px hsl(270 30% 30% / 0.25), 0 8px 24px hsl(270 30% 30% / 0.15)',
        }}
      >
        {/* Header */}
        <div
          className="px-4 py-3 flex items-center justify-between"
          style={{
            background: 'linear-gradient(135deg, hsl(270 70% 55%), hsl(290 65% 60%))',
          }}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-white font-bold text-sm">Restaurant Recommender</h3>
              <p className="text-white/80 text-xs">Powered by AI</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsOpen(false)}
            className="text-white hover:bg-white/20 rounded-full"
          >
            <X className="w-5 h-5" />
          </Button>
        </div>

        {/* Messages */}
        <ScrollArea className="flex-1 bg-card" ref={scrollRef}>
          <div className="p-4 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-2 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                    message.role === 'user'
                      ? 'bg-secondary'
                      : ''
                  }`}
                  style={message.role === 'assistant' ? {
                    background: 'linear-gradient(135deg, hsl(270 70% 55%), hsl(290 65% 60%))',
                  } : {}}
                >
                  {message.role === 'user' ? (
                    <User className="w-4 h-4 text-muted-foreground" />
                  ) : (
                    <Bot className="w-4 h-4 text-white" />
                  )}
                </div>
                <div
                  className={`max-w-[80%] px-4 py-2.5 rounded-2xl text-sm ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground rounded-br-md'
                      : 'bg-secondary text-foreground rounded-bl-md'
                  }`}
                >
                  {message.content.split('**').map((part, i) =>
                    i % 2 === 1 ? (
                      <strong key={i}>{part}</strong>
                    ) : (
                      <span key={i}>{part}</span>
                    )
                  )}
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="flex gap-2">
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center"
                  style={{
                    background: 'linear-gradient(135deg, hsl(270 70% 55%), hsl(290 65% 60%))',
                  }}
                >
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="bg-secondary px-4 py-3 rounded-2xl rounded-bl-md">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Input */}
        <div className="p-3 bg-card border-t border-border">
          <div className="flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Ask for a recommendation..."
              className="flex-1 rounded-full bg-secondary border-0 focus-visible:ring-primary"
            />
            <Button
              onClick={handleSend}
              disabled={!inputValue.trim() || isTyping}
              size="icon"
              className="rounded-full shrink-0"
              style={{
                background: inputValue.trim() && !isTyping
                  ? 'linear-gradient(135deg, hsl(270 70% 55%), hsl(290 65% 60%))'
                  : undefined,
              }}
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </>
  );
};

export default ChatbotWidget;
