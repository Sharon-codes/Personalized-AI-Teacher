import os
import cv2
import face_recognition
import numpy as np
import speech_recognition as sr
import customtkinter as ctk
from threading import Thread
import queue
from datetime import datetime
import pyttsx3
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Use os.getenv to safely load the variable
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please ensure the .env file is correctly set up.")

# Set the GOOGLE_API_KEY environment variable for google-generativeai
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Configure the Gemini API
genai.configure()  # No need to pass api_key since it's now in GOOGLE_API_KEY

# Initialize the Gemini model
MODEL_NAME = "gemini-1.5-flash"  # Updated to a supported model
model = genai.GenerativeModel(MODEL_NAME)

# Face database
known_face_encodings = []
known_face_names = []
database_path = "dataset/"

# Conversation memory and learning level
conversation_history = []
student_learning_level = "beginner"
interaction_mode = None  # Will be set to "text" or "voice"

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)  # Speed of speech
tts_engine.setProperty("volume", 0.9)  # Volume (0.0 to 1.0)

# Fallback responses for common topics
FALLBACK_RESPONSES = {
    "laws of motion": (
        "The laws of motion were discovered by Sir Isaac Newton and they help us understand how things move! There are three laws: "
        "First, an object will stay still or keep moving in a straight line unless something pushes or pulls it—this is called inertia. "
        "For example, a ball won’t roll unless you kick it. "
        "Second, when you push something, it moves faster depending on how hard you push and how heavy it is—this is written as Force equals Mass times Acceleration. "
        "So, pushing a light toy car is easier than pushing a heavy wagon. "
        "Third, for every action, there’s an equal and opposite reaction. When you jump off a boat, you push the boat backward while you move forward! "
        "These laws are everywhere—like when you ride a bike or throw a ball."
    )
}

def load_face_database():
    print("Loading face database...")
    for filename in os.listdir(database_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(database_path, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)
    print("Face database loaded.")

def identify_person():
    print("Starting face identification...")
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video frame.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            video_capture.release()
            cv2.destroyAllWindows()
            print(f"Identified person: {name}")
            return name
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    print("Face identification stopped. Returning Unknown.")
    return "Unknown"

def adapt_response(response, learning_level, mode):
    if mode == "voice":
        if learning_level == "beginner":
            return f"I’ll explain in simple words: {response}"
        elif learning_level == "intermediate":
            return f"Here’s a clear explanation for you: {response}"
        else:
            return f"Let me give you a detailed answer: {response}"
    else:  # Text mode
        if learning_level == "beginner":
            return f"Let me explain simply: {response}"
        elif learning_level == "intermediate":
            return f"Here’s a clear explanation: {response}"
        else:
            return f"Here’s a detailed answer: {response}"

def speak_text(text):
    """Read the text aloud using pyttsx3."""
    print(f"Speaking: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def generate_response(query, user_name, context=""):
    global conversation_history, student_learning_level, interaction_mode
    # Check for fallback response
    query_lower = query.lower()
    if query_lower in FALLBACK_RESPONSES:
        response_text = FALLBACK_RESPONSES[query_lower]
        adapted_response = adapt_response(response_text, student_learning_level, interaction_mode)
        conversation_history.append({"role": "Human", "text": query, "time": datetime.now().strftime("%H:%M:%S")})
        conversation_history.append({"role": "Assistant", "text": adapted_response, "time": datetime.now().strftime("%H:%M:%S")})
        print(f"Using fallback response: {adapted_response}")
        return adapted_response

    history_context = "\n".join([f"{entry['role']}: {entry['text']}" for entry in conversation_history[-5:]])
    prompt = f"Previous conversation:\n{history_context}\n\n{context}Human: Hello, {user_name}! You asked: {query}\nAssistant: "
    print(f"Generating response for prompt: {prompt}")
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        adapted_response = adapt_response(response_text, student_learning_level, interaction_mode)
        conversation_history.append({"role": "Human", "text": query, "time": datetime.now().strftime("%H:%M:%S")})
        conversation_history.append({"role": "Assistant", "text": adapted_response, "time": datetime.now().strftime("%H:%M:%S")})
        print(f"Extracted response: {adapted_response}")
        return adapted_response
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        error_msg = f"Sorry, I couldn't connect to the Gemini API. Please ensure your API key is correct and you have internet access."
        return error_msg

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice input...")
        speak_text("Please speak your question now.")
        recognizer.pause_threshold = 1.0  # Wait longer for user to finish speaking
        recognizer.energy_threshold = 300  # Adjust sensitivity to background noise
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            query = recognizer.recognize_google(audio)
            print(f"You said: {query}")
            return query
        except sr.WaitTimeoutError:
            print("Speech timeout.")
            return "Sorry, I didn't hear anything. Please try again."
        except sr.UnknownValueError:
            print("Speech not understood.")
            return "Sorry, I didn't understand that. Could you repeat?"
        except sr.RequestError as e:
            print(f"Speech service error: {e}")
            return "Sorry, there was an error with the speech service."

def handle_quick_action(action, user, chat_frame, response_queue):
    query = action
    user_bubble = ctk.CTkLabel(chat_frame, text=f"You: {query}", wraplength=500, justify="right", fg_color="#BBDEFB", corner_radius=10, padx=10, pady=5)
    user_bubble.pack(anchor="e", pady=5)
    robot_bubble = ctk.CTkLabel(chat_frame, text="Typing...", wraplength=500, justify="left", fg_color="#D1E8D5", corner_radius=10, padx=10, pady=5)
    robot_bubble.pack(anchor="w", pady=5)
    Thread(target=process_query, args=(query, user, response_queue, chat_frame, robot_bubble)).start()
    chat_frame._parent_canvas.yview_moveto(1.0)

def handle_text_input(entry, user, chat_frame, response_queue):
    query = entry.get()
    if query:
        print(f"Text input received: {query}")
        user_bubble = ctk.CTkLabel(chat_frame, text=f"You: {query}", wraplength=500, justify="right", fg_color="#BBDEFB", corner_radius=10, padx=10, pady=5)
        user_bubble.pack(anchor="e", pady=5)
        robot_bubble = ctk.CTkLabel(chat_frame, text="Typing...", wraplength=500, justify="left", fg_color="#D1E8D5", corner_radius=10, padx=10, pady=5)
        robot_bubble.pack(anchor="w", pady=5)
        entry.delete(0, "end")
        Thread(target=process_query, args=(query, user, response_queue, chat_frame, robot_bubble)).start()
        chat_frame._parent_canvas.yview_moveto(1.0)

def handle_voice_input(user, chat_frame, response_queue):
    query = get_voice_input()
    print(f"Voice input received: {query}")
    user_bubble = ctk.CTkLabel(chat_frame, text=f"You: {query}", wraplength=500, justify="right", fg_color="#BBDEFB", corner_radius=10, padx=10, pady=5)
    user_bubble.pack(anchor="e", pady=5)
    robot_bubble = ctk.CTkLabel(chat_frame, text="Typing...", wraplength=500, justify="left", fg_color="#D1E8D5", corner_radius=10, padx=10, pady=5)
    robot_bubble.pack(anchor="w", pady=5)
    Thread(target=process_query, args=(query, user, response_queue, chat_frame, robot_bubble)).start()
    chat_frame._parent_canvas.yview_moveto(1.0)

def process_query(query, user, q, chat_frame, robot_bubble):
    print(f"Processing query: {query} for user: {user}")
    context = "You are a friendly robot teacher. Respond in a way that encourages learning and engagement. "
    try:
        if query.lower() == "start a quiz":
            response = "Let’s start a quiz! I’ll ask you a question based on your learning level. Here’s your first question: What is 2 + 2?"
        else:
            response = generate_response(query, user, context)
        print(f"Putting response in queue: {response}")
        q.put((response, chat_frame, robot_bubble))
    except Exception as e:
        print(f"Error processing query: {e}")
        error_response = "I’m sorry, I encountered an error while processing your request."
        q.put((error_response, chat_frame, robot_bubble))

def create_main_gui():
    global interaction_mode

    window = ctk.CTk()
    window.title("Robot Teacher Assistant")
    window.geometry("700x800")

    response_queue = queue.Queue()

    header_frame = ctk.CTkFrame(window, fg_color="#E0F7FA")
    header_frame.pack(fill="x", pady=10)
    avatar_label = ctk.CTkLabel(header_frame, text="🤖", font=("Arial", 40))
    avatar_label.pack(side="left", padx=20)
    current_hour = datetime.now().hour
    greeting = "Good morning" if current_hour < 12 else "Good afternoon" if current_hour < 17 else "Good evening"
    user_name = identify_person()
    title_label = ctk.CTkLabel(header_frame, text=f"{greeting}, {user_name}! I’m Your Robot Teacher", font=("Arial", 24, "bold"))
    title_label.pack(side="left", padx=10)

    level_frame = ctk.CTkFrame(window, fg_color="#E0F7FA")
    level_frame.pack(fill="x", padx=20, pady=5)
    level_label = ctk.CTkLabel(level_frame, text=f"Learning Level: {student_learning_level}", font=("Arial", 14))
    level_label.pack(side="left", padx=10)
    level_var = ctk.StringVar(value=student_learning_level)
    level_menu = ctk.CTkOptionMenu(level_frame, values=["beginner", "intermediate", "advanced"], variable=level_var,
                                   command=lambda lvl: set_learning_level(lvl, level_label))
    level_menu.pack(side="right", padx=10)

    chat_frame = ctk.CTkScrollableFrame(window, width=650, height=400, fg_color="#F5F5F5")
    chat_frame.pack(pady=10, padx=20)
    initial_greeting = f"{greeting}, {user_name}! How’s your day going? Ready to learn something new?"
    robot_bubble = ctk.CTkLabel(chat_frame, text=initial_greeting, 
                                wraplength=500, justify="left", fg_color="#D1E8D5", corner_radius=10, padx=10, pady=5)
    robot_bubble.pack(anchor="w", pady=5)
    suggestion_bubble = ctk.CTkLabel(chat_frame, text="Try asking a question, or I can quiz you on a topic!", 
                                     wraplength=500, justify="left", fg_color="#D1E8D5", corner_radius=10, padx=10, pady=5)
    suggestion_bubble.pack(anchor="w", pady=5)

    if interaction_mode == "voice":
        # Speak the initial greeting in voice mode
        speak_text(initial_greeting)
        speak_text("Try asking a question, or I can quiz you on a topic!")

    action_frame = ctk.CTkFrame(window, fg_color="#E0F7FA")
    action_frame.pack(fill="x", padx=20, pady=5)
    ask_button = ctk.CTkButton(action_frame, text="Ask a Question", command=lambda: handle_quick_action("Ask a question", user_name, chat_frame, response_queue))
    ask_button.pack(side="left", padx=5)
    fact_button = ctk.CTkButton(action_frame, text="Tell a Fact", command=lambda: handle_quick_action("Tell me an interesting fact", user_name, chat_frame, response_queue))
    fact_button.pack(side="left", padx=5)
    quiz_button = ctk.CTkButton(action_frame, text="Quiz Me", command=lambda: handle_quick_action("Start a quiz", user_name, chat_frame, response_queue))
    quiz_button.pack(side="left", padx=5)

    input_frame = ctk.CTkFrame(window, fg_color="#E0F7FA")
    input_frame.pack(fill="x", padx=20, pady=10)

    # UI elements based on mode
    if interaction_mode == "text":
        query_entry = ctk.CTkEntry(input_frame, width=500, placeholder_text="Type your question here...")
        query_entry.pack(side="left", padx=5)
        submit_button = ctk.CTkButton(input_frame, text="Send", command=lambda: handle_text_input(query_entry, user_name, chat_frame, response_queue))
        submit_button.pack(side="left", padx=5)
        voice_button = ctk.CTkButton(input_frame, text="🎤 Speak", command=lambda: handle_voice_input(user_name, chat_frame, response_queue))
        voice_button.pack(side="left", padx=5)
    else:  # Voice mode
        voice_button = ctk.CTkButton(input_frame, text="🎤 Speak", command=lambda: handle_voice_input(user_name, chat_frame, response_queue))
        voice_button.pack(side="left", padx=5)
        # Automatically start listening for voice input
        Thread(target=handle_voice_input, args=(user_name, chat_frame, response_queue)).start()

    def set_learning_level(level, label):
        global student_learning_level
        student_learning_level = level
        label.configure(text=f"Learning Level: {level}")
        print(f"Learning level set to: {level}")

    def check_queue():
        print(f"Checking queue... Queue size: {response_queue.qsize()}")
        try:
            response, chat_frame, robot_bubble = response_queue.get_nowait()
            print(f"Updating GUI with response: {response}")
            robot_bubble.configure(text=f"Robot: {response}")
            chat_frame._parent_canvas.yview_moveto(1.0)
            # In voice mode, speak the response and listen for the next input
            if interaction_mode == "voice":
                speak_text(response)
                Thread(target=handle_voice_input, args=(user_name, chat_frame, response_queue)).start()
        except queue.Empty:
            print("Queue is empty, waiting for response...")
        window.after(100, check_queue)

    window.after(100, check_queue)
    print("GUI initialized.")
    window.mainloop()

def mode_selection_screen():
    global interaction_mode

    window = ctk.CTk()
    window.title("Robot Teacher Assistant - Choose Mode")
    window.geometry("400x300")

    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("green")

    welcome_label = ctk.CTkLabel(window, text="Welcome! How would you like to interact?", font=("Arial", 18, "bold"))
    welcome_label.pack(pady=20)

    def set_mode(mode):
        global interaction_mode
        interaction_mode = mode
        print(f"Interaction mode set to: {mode}")
        window.destroy()
        create_main_gui()

    text_button = ctk.CTkButton(window, text="Text-Based", command=lambda: set_mode("text"))
    text_button.pack(pady=10)

    voice_button = ctk.CTkButton(window, text="Voice-Based", command=lambda: set_mode("voice"))
    voice_button.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    load_face_database()
    mode_selection_screen()