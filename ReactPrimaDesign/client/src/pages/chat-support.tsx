import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2, Send, Mic, AlertTriangle, MicOff, User, Bot } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import MainLayout from '@/layouts/main-layout';
import { BottomNav } from '@/components/navigation/bottom-nav';

// Import the Voice Analysis for scam detection
import { analyzeTranscriptForScams, startVoiceRecording, getAudioFromRecorder } from '@/lib/voice-analysis';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: Date;
  isLoading?: boolean;
}

interface QuickReply {
  id: string;
  text: string;
}

export default function ChatSupport() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  
  // Messages state
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      content: 'Hello! I\'m your UPI safety assistant. How can I help you today?',
      role: 'assistant',
      timestamp: new Date()
    }
  ]);
  
  // Input state
  const [input, setInput] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  // Quick replies
  const [quickReplies, setQuickReplies] = useState<QuickReply[]>([
    { id: '1', text: 'How do I verify if a UPI ID is safe?' },
    { id: '2', text: 'I think I sent money to a scammer' },
    { id: '3', text: 'How do I report a UPI scam?' },
    { id: '4', text: 'What are common UPI scams I should know about?' }
  ]);
  
  // Voice recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingTimerRef = useRef<number | null>(null);
  
  // Auto-scroll chat to bottom
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  // Clean up recording timer on unmount
  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
      }
    };
  }, [isRecording]);
  
  // Handle message submission
  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    
    if (!input.trim() && !isRecording) return;
    
    // Create user message
    const userMessageContent = input.trim();
    const userMessageId = Date.now().toString();
    
    // Clear input
    setInput('');
    
    // Add user message to state
    setMessages(prev => [
      ...prev,
      {
        id: userMessageId,
        content: userMessageContent,
        role: 'user',
        timestamp: new Date()
      }
    ]);
    
    // Add loading message
    const loadingMessageId = `loading-${Date.now()}`;
    setMessages(prev => [
      ...prev,
      {
        id: loadingMessageId,
        content: '',
        role: 'assistant',
        timestamp: new Date(),
        isLoading: true
      }
    ]);
    
    setIsSubmitting(true);
    
    try {
      // Get mock user ID (in a real app, this would come from auth)
      const userId = 1;
      
      // Call the server to get AI response
      const response = await fetch('/api/chat/1/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: userMessageContent
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to get response');
      }
      
      const data = await response.json();
      
      // Replace loading message with actual response
      setMessages(prev => 
        prev.filter(msg => msg.id !== loadingMessageId).concat({
          id: Date.now().toString(),
          content: data.assistantMessage.content,
          role: 'assistant',
          timestamp: new Date(data.assistantMessage.timestamp)
        })
      );
      
      // Update quick replies
      if (data.quickReplies && data.quickReplies.length > 0) {
        setQuickReplies(
          data.quickReplies.map((text: string, index: number) => ({
            id: `qr-${Date.now()}-${index}`,
            text
          }))
        );
      }
    } catch (error) {
      console.error('Error getting chat response:', error);
      
      // Remove loading message
      setMessages(prev => prev.filter(msg => msg.id !== loadingMessageId));
      
      // Add error message
      setMessages(prev => [
        ...prev,
        {
          id: Date.now().toString(),
          content: "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.",
          role: 'assistant',
          timestamp: new Date()
        }
      ]);
      
      toast({
        title: 'Error',
        description: 'Failed to get response from AI assistant',
        variant: 'destructive'
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  
  // Handle selecting a quick reply
  const handleQuickReply = (text: string) => {
    setInput(text);
    setTimeout(() => handleSubmit(), 100);
  };
  
  // Start voice recording
  const startRecording = async () => {
    try {
      const recorder = await startVoiceRecording();
      mediaRecorderRef.current = recorder;
      
      // Start recording
      recorder.start();
      setIsRecording(true);
      
      // Set up timer to show recording duration
      let duration = 0;
      recordingTimerRef.current = window.setInterval(() => {
        duration += 1;
        setRecordingDuration(duration);
        
        // Auto-stop after 30 seconds
        if (duration >= 30) {
          stopRecording();
        }
      }, 1000);
      
      toast({
        title: 'Recording started',
        description: 'Speak clearly into your microphone'
      });
    } catch (error) {
      console.error('Failed to start recording:', error);
      toast({
        title: 'Microphone access denied',
        description: 'Please allow microphone access to use voice input',
        variant: 'destructive'
      });
    }
  };
  
  // Stop voice recording and process audio
  const stopRecording = async () => {
    if (!mediaRecorderRef.current || !isRecording) return;
    
    // Stop the timer
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    
    // Get the recorder
    const recorder = mediaRecorderRef.current;
    
    // Stop recording and get audio data
    recorder.stop();
    setIsRecording(false);
    setRecordingDuration(0);
    
    try {
      // Get audio blob
      const audioBlob = await getAudioFromRecorder(recorder);
      
      // Create a file for upload
      const audioFile = new File([audioBlob], 'voice-message.mp3', { type: 'audio/mp3' });
      
      // Show processing message
      toast({
        title: 'Processing voice',
        description: 'Analyzing your voice message...'
      });
      
      // Add user message placeholder
      const userMessageId = Date.now().toString();
      setMessages(prev => [
        ...prev,
        {
          id: userMessageId,
          content: '🎤 Voice message (transcribing...)',
          role: 'user',
          timestamp: new Date()
        }
      ]);
      
      // Add loading message
      const loadingMessageId = `loading-${Date.now()}`;
      setMessages(prev => [
        ...prev,
        {
          id: loadingMessageId,
          content: '',
          role: 'assistant',
          timestamp: new Date(),
          isLoading: true
        }
      ]);
      
      // For demo purposes, we'll simulate the transcription and analysis
      // In a real app, you would upload the audio to the server
      
      // Simulate a response delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Simulated transcription (in production this would come from the server)
      const transcription = "I received a call saying my UPI account needs verification and they asked me to transfer Rs. 1 to verify my account. Is this legitimate?";
      
      // Update the user message with the transcription
      setMessages(prev => 
        prev.map(msg => 
          msg.id === userMessageId 
            ? { ...msg, content: `🎤 ${transcription}` } 
            : msg
        )
      );
      
      // Analyze the transcript for scams
      const analysis = await analyzeTranscriptForScams(transcription);
      
      // Generate AI response based on scam analysis
      let aiResponse = "";
      if (analysis.isScam) {
        aiResponse = `⚠️ This appears to be a scam! 
        
What you described is a classic verification scam. Legitimate banks and UPI services NEVER ask you to transfer money to verify your account.

I detected these red flags:
${analysis.scamIndicators.map(indicator => `- ${indicator}`).join('\n')}

What to do:
1. Do not transfer any money
2. Block and report the number that called you
3. Report this to cybercrime.gov.in or call 1930
        
Stay safe!`;
      } else {
        // Call the chat API for a response
        const response = await fetch('/api/chat/1/message', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content: transcription
          })
        });
        
        if (!response.ok) {
          throw new Error('Failed to get response');
        }
        
        const data = await response.json();
        aiResponse = data.assistantMessage.content;
        
        // Update quick replies
        if (data.quickReplies && data.quickReplies.length > 0) {
          setQuickReplies(
            data.quickReplies.map((text: string, index: number) => ({
              id: `qr-${Date.now()}-${index}`,
              text
            }))
          );
        }
      }
      
      // Replace loading message with actual response
      setMessages(prev => 
        prev.filter(msg => msg.id !== loadingMessageId).concat({
          id: Date.now().toString(),
          content: aiResponse,
          role: 'assistant',
          timestamp: new Date()
        })
      );
      
    } catch (error) {
      console.error('Error processing voice recording:', error);
      
      // Update user message to indicate error
      setMessages(prev => 
        prev.map(msg => 
          msg.id === 'voice-processing' 
            ? { ...msg, content: '🎤 Voice message (processing failed)' } 
            : msg
        )
      );
      
      toast({
        title: 'Voice processing failed',
        description: 'Unable to process your voice message. Please try again or use text input.',
        variant: 'destructive'
      });
    }
  };
  
  return (
    // Use the fullHeight prop to make the layout work for chat but keep the bottom nav
    <MainLayout className="p-0" fullHeight>
      {/* Mobile-optimized chat layout structure */}
      <div className="flex flex-col h-full w-full">
        {/* Fixed Header - Always visible at the top */}
        <header className="border-b bg-card px-4 py-3 flex-shrink-0 z-10 bg-white dark:bg-gray-800 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                <Bot size={18} className="text-white" />
              </div>
              <h1 className="text-lg font-semibold">AI Safety Assistant</h1>
            </div>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setLocation('/home')}
              className="h-8 px-3"
            >
              Close
            </Button>
          </div>
        </header>
        
        {/* Scrollable Messages Area - The only part that should scroll */}
        <div className="flex-1 overflow-hidden relative">
          {/* Use a native scrollable div instead of ScrollArea for better mobile performance */}
          <div 
            className="overflow-y-auto overscroll-contain px-4 pt-2 pb-4 h-full"
            style={{
              maxHeight: 'calc(100vh - 180px)', // Account for header and input area
              // Use these CSS variables for better mobile compatibility
              "--safe-area-inset-bottom": "env(safe-area-inset-bottom, 0px)",
            } as React.CSSProperties}
          >
            <div className="flex flex-col gap-4 pb-4">
              {messages.map((message) => (
                <div 
                  key={message.id} 
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div 
                    className={`flex gap-2 max-w-[85%] ${
                      message.role === 'user' 
                        ? 'flex-row-reverse' 
                        : 'flex-row'
                    }`}
                  >
                    <div 
                      className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                        message.role === 'user' 
                          ? 'bg-primary text-primary-foreground' 
                          : 'bg-muted'
                      }`}
                    >
                      {message.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                    </div>
                    <div 
                      className={`rounded-lg p-3 shadow-sm max-w-full ${
                        message.role === 'user' 
                          ? 'bg-primary text-primary-foreground' 
                          : 'bg-muted dark:bg-gray-800'
                      }`}
                    >
                      {message.isLoading ? (
                        <div className="flex items-center gap-2 h-6">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          <span className="text-sm">Thinking...</span>
                        </div>
                      ) : (
                        <div className="whitespace-pre-wrap break-words">
                          {message.content}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>
          
          {/* Quick Replies - Absolutely positioned at the bottom of the chat area */}
          {quickReplies.length > 0 && (
            <div className="absolute bottom-0 left-0 right-0 bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm border-t z-10">
              <div className="flex overflow-x-auto py-2 px-4 gap-2 no-scrollbar hide-scrollbar">
                {quickReplies.map((reply) => (
                  <Button
                    key={reply.id}
                    variant="outline"
                    size="sm"
                    className="whitespace-nowrap flex-shrink-0 shadow-sm text-xs rounded-full border-primary/30 hover:bg-primary/10 hover:text-primary"
                    onClick={() => handleQuickReply(reply.text)}
                    disabled={isSubmitting}
                  >
                    {reply.text}
                  </Button>
                ))}
              </div>
            </div>
          )}
        </div>
        
        {/* Fixed Input Area - Stays at the bottom above the navigation */}
        <div className="flex-shrink-0 border-t bg-white dark:bg-gray-800 px-3 py-2 z-20 shadow-[0_-1px_3px_rgba(0,0,0,0.1)] pb-20">
          {isRecording ? (
            <div className="w-full flex items-center gap-2">
              <div className="flex-1 bg-muted rounded-full p-2 flex items-center overflow-hidden">
                <div className="flex-1 flex items-center gap-2 whitespace-nowrap overflow-hidden">
                  <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse flex-shrink-0" />
                  <span className="text-sm truncate">Recording... {recordingDuration}s</span>
                </div>
              </div>
              <Button 
                variant="destructive"
                size="icon"
                className="h-10 w-10 rounded-full flex-shrink-0"
                onClick={stopRecording}
                aria-label="Stop recording"
              >
                <MicOff size={16} />
              </Button>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="w-full flex items-center gap-2">
              <div className="relative flex-1">
                <Input 
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type your message..."
                  disabled={isSubmitting}
                  className="w-full min-w-0 py-2 px-4 rounded-full border-gray-300 dark:border-gray-600 focus:border-primary focus:ring-2 focus:ring-primary/20"
                  autoComplete="off"
                  autoCapitalize="sentences"
                  spellCheck="true"
                  id="chat-input"
                  aria-label="Message input"
                  // These attributes help with mobile keyboard behavior
                  enterKeyHint="send"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit();
                    }
                  }}
                  // Important for handling mobile virtual keyboards
                  onFocus={() => {
                    // Add a small timeout to ensure scrolling happens after keyboard appears
                    setTimeout(() => {
                      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
                    }, 300);
                  }}
                />
                {isSubmitting && (
                  <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                    <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
                  </div>
                )}
              </div>
              <Button 
                variant="default"
                size="icon"
                type="submit"
                className="h-10 w-10 rounded-full flex-shrink-0 bg-primary hover:bg-primary/90"
                disabled={!input.trim() || isSubmitting}
                aria-label="Send message"
              >
                <Send size={16} />
              </Button>
              <Button 
                variant="outline"
                size="icon"
                type="button"
                className="h-10 w-10 rounded-full flex-shrink-0 border-gray-300 dark:border-gray-600"
                onClick={startRecording}
                disabled={isSubmitting}
                aria-label="Start voice recording"
              >
                <Mic size={16} />
              </Button>
            </form>
          )}
        </div>
      </div>
    </MainLayout>
  );
}