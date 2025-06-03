def get_chat_styling():
    return """
        <style>
        .user-message {
            padding: 10px;
            margin: 5px 10px 5px auto;
            border-radius: 15px;
            background-color: #e6f3ff;
            max-width: 80%;
            float: right;
            clear: both;
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
        }
        .chat-container::after {
            content: "";
            clear: both;
            display: table;
        }
        </style>
    """
