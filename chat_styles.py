def get_chat_styling():
    return """
        <style>
        .user-message-container {
            display: flex;
            justify-content: flex-end;
            align-items: start;
            margin: 10px 0;
            gap: 10px;
            animation: fadeIn 0.3s ease-in;
        }
        
        .assistant-message-container {
            display: flex;
            justify-content: flex-start;
            align-items: start;
            margin: 10px 0;
            gap: 10px;
            animation: fadeIn 0.3s ease-in;
        }
        
        .user-message {
            padding: 10px 15px;
            border-radius: 15px 15px 0 15px;
            background-color: #e6f3ff;
            max-width: 80%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            color: #2c5282;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        
        .assistant-message {
            padding: 10px 15px;
            border-radius: 15px 15px 15px 0;
            background-color: #f0f0f0;
            max-width: 80%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            color: #4a5568;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        
        .message-avatar {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .user-avatar {
            background-color: #e6f3ff;
            color: #2c5282;
        }
        
        .assistant-avatar {
            background-color: #f0f0f0;
            color: #4a5568;
        }
        
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
            min-height: 200px;
            max-height: 600px;
            overflow-y: auto;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            scroll-behavior: smooth;
        }
        
        .chat-container::after {
            content: "";
            clear: both;
            display: table;
        }
        
        /* Thinking animation */
        .thinking-dots::after {
            content: '...';
            display: inline-block;
            animation: thinking 1.5s infinite;
            width: 20px;
        }
        
        @keyframes thinking {
            0% { content: '.'; }
            33% { content: '..'; }
            66% { content: '...'; }
            100% { content: '.'; }
        }
        
        /* Message fade-in animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Improve scrollbar styling */
        .chat-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 3px;
        }
        
        .chat-container::-webkit-scrollbar-thumb:hover {
            background: #666;
        }
        
        /* Support for older browsers */
        @supports not (animation-name: fadeIn) {
            .user-message-container,
            .assistant-message-container {
                opacity: 1;
                transform: none;
            }
        }
        </style>
    """
