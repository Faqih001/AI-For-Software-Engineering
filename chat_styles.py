def get_chat_styling():
    return """
        <style>
        .user-message-container {
            display: flex;
            justify-content: flex-end;
            align-items: start;
            margin: 10px 0;
            gap: 10px;
        }
        .assistant-message-container {
            display: flex;
            justify-content: flex-start;
            align-items: start;
            margin: 10px 0;
            gap: 10px;
        }
        .user-message {
            padding: 10px;
            border-radius: 15px;
            background-color: #e6f3ff;
            max-width: 80%;
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
        }
        .user-avatar {
            background-color: #e6f3ff;
            color: #2c5282;
        }
        .assistant-avatar {
            background-color: #f0f0f0;
            color: #4a5568;
        }
        .assistant-message {
            padding: 10px;
            margin: 5px auto 5px 10px;
            border-radius: 15px;
            background-color: #f0f0f0;
            max-width: 80%;
            float: left;
            clear: both;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
            min-height: 200px;
            max-height: 500px;
            overflow-y: auto;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chat-container::after {
            content: "";
            clear: both;
            display: table;
        }
        
        /* Thinking animation */
        .thinking-dots::after {
            content: '';
            animation: thinking 1.5s infinite;
        }
        @keyframes thinking {
            0% { content: ''; }
            25% { content: '.'; }
            50% { content: '..'; }
            75% { content: '...'; }
            100% { content: ''; }
        }
        .thinking-message {
            display: flex;
            align-items: center;
            color: #666;
        }
        </style>
    """
