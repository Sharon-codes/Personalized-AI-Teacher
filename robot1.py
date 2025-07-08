import os
import cv2
import face_recognition
import numpy as np
import speech_recognition as sr
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import time
import re
import pyttsx3  # For English speech
from gtts import gTTS  # For Hindi speech
import winsound  # For Windows audio playback

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please ensure the .env file is correctly set up.")

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Configure the Gemini API
genai.configure()
MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# Paths
dataset_path = "dataset/"
student_data_path = "student_data.xlsx"

# Learning styles
learning_styles = {
    "practical": "Explain concepts with real-life examples and practical scenarios that the student can relate to.",
    "application": "Focus on how concepts can be applied to solve problems and create solutions.",
    "theory": "Provide detailed theoretical explanations with foundational principles and academic context."
}

# Global variables
conversation_history = []
student_data = {}
interaction_mode = "voice"
preferred_language = None

# Initialize pyttsx3 engine for English
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 180)  # Fast, natural speed for English
tts_engine.setProperty('volume', 0.9)

# Fallback responses (English only for simplicity)
FALLBACK_RESPONSES = {
    "laws of motion": {
        "practical": (
            "Think about riding a bicycle. When you stop pedaling, you don’t instantly stop—that’s inertia, the first law. "
            "Pedal harder, you speed up—that’s force equals mass times acceleration, the second law. "
            "Push the pedals, the bike pushes you forward—action-reaction, the third law."
        ),
        "application": (
            "Newton’s first law explains why seatbelts save lives. The second law, force equals mass times acceleration, "
            "calculates force needed to move stuff. The third law powers rockets by pushing gas out to lift off."
        ),
        "theory": (
            "Newton’s first law: objects stay at rest or in motion unless acted upon. Second law: force equals mass times acceleration. "
            "Third law: every action has an equal, opposite reaction."
        )
    }
}

def load_student_data():
    """Load student data from Excel sheet."""
    print("Loading student data...")
    try:
        if not os.path.exists(student_data_path):
            create_student_data_file()
        else:
            df = pd.read_excel(student_data_path)
            for _, row in df.iterrows():
                student_data[row['Name']] = {'learning_style': row['Learning Style']}
            print(f"Loaded data for {len(student_data)} students.")
    except Exception as e:
        print(f"Error loading student data: {e}")
        create_student_data_file()

def create_student_data_file():
    """Create a sample student data file."""
    print("Creating student data file...")
    student_names = [os.path.splitext(f)[0] for f in os.listdir(dataset_path) if f.endswith((".jpg", ".png"))]
    if not student_names:
        student_names = ['John', 'Emily', 'Unknown']
    
    import random
    sample_data = {'Name': student_names, 'Learning Style': [random.choice(['practical', 'application', 'theory']) for _ in student_names]}
    df = pd.DataFrame(sample_data)
    df.to_excel(student_data_path, index=False)
    
    for _, row in df.iterrows():
        student_data[row['Name']] = {'learning_style': row['Learning Style']}
    print(f"Sample student data created with {len(student_data)} entries.")

def load_face_encodings():
    """Load face encodings from dataset directory."""
    print("Loading face encodings...")
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Created dataset directory: {dataset_path}")
        return
    
    for filename in os.listdir(dataset_path):
        if filename.endswith((".jpg", ".png")):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(dataset_path, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"Loaded encoding for: {name}")
                else:
                    print(f"No face found in: {filename}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
    print(f"Loaded {len(known_face_names)} face encodings.")

def identify_person():
    """Identify a person using facial recognition."""
    print("Starting face identification...")
    video_capture = cv2.VideoCapture(0)
    start_time = time.time()
    timeout = 15
    
    print("Looking for your face...")
    while time.time() - start_time < timeout:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding in face_encodings:
            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    video_capture.release()
                    print(f"Welcome, {name}!")
                    return name
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    print("Identification timed out. Using 'Unknown'.")
    return "Unknown"

def get_learning_style(name):
    """Get learning style for a student."""
    if name in student_data:
        return student_data[name].get('learning_style', 'practical')
    update_student_data(name, 'practical')
    return 'practical'

def update_student_data(name, learning_style):
    """Update student data in Excel."""
    print(f"Adding/updating {name} with style: {learning_style}")
    student_data[name] = {'learning_style': learning_style}
    
    try:
        if os.path.exists(student_data_path):
            df = pd.read_excel(student_data_path)
            if name not in df['Name'].values:
                new_row = pd.DataFrame({'Name': [name], 'Learning Style': [learning_style]})
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                df.loc[df['Name'] == name, 'Learning Style'] = learning_style
            df.to_excel(student_data_path, index=False)
        else:
            df = pd.DataFrame({'Name': [name], 'Learning Style': [learning_style]})
            df.to_excel(student_data_path, index=False)
        print(f"Updated student data for {name}.")
    except Exception as e:
        print(f"Error updating student data: {e}")

def clean_response_text(response):
    """Clean response text by removing markdown and converting bullet points to numbered points."""
    response = re.sub(r'\*+', '', response)
    response = re.sub(r'_+', '', response)
    
    lines = response.split('\n')
    point_counter = 1
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('-') or line.startswith('*'):
            line = line.lstrip('-* ').strip()
            if line:
                cleaned_lines.append(f"Point {point_counter}: {line}")
                point_counter += 1
        else:
            cleaned_lines.append(line)
    
    return ' '.join(cleaned_lines)

def is_hindi(text):
    """Simple check for Hindi (Devanagari script)."""
    return any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in text)

def speak_text(text):
    """Speak text in English (pyttsx3) or Hindi (gTTS) based on content or preference."""
    print(f"Speaking: {text}")
    
    # Split text into phrases for natural flow
    phrases = re.split(r'(?<=[.,!?])\s+', text)
    if len(phrases) == 1:
        words = text.split()
        phrases = [' '.join(words[i:i+5]) for i in range(0, len(words), 5)]
    
    for i, phrase in enumerate(phrases):
        if not phrase.strip():
            continue
        
        if preferred_language == "hindi" or is_hindi(phrase):  # Use gTTS for Hindi
            try:
                print(f"Generating Hindi audio for: {phrase}")
                tts = gTTS(text=phrase, lang='hi', slow=False)
                temp_file = f"temp_audio_{i}.mp3"
                tts.save(temp_file)
                print(f"Saved Hindi audio to {temp_file}")
                if os.path.exists(temp_file):
                    winsound.PlaySound(temp_file, winsound.SND_FILENAME | winsound.SND_ASYNC)  # Async to avoid blocking
                    time.sleep(0.5)  # Brief wait to ensure playback starts
                    os.remove(temp_file)
                    print(f"Played and removed {temp_file}")
                else:
                    print(f"Error: {temp_file} not found after saving")
                pause = 0.15 if i < len(phrases) - 1 else 0
                if "Point" in phrase or phrase.endswith(('!', '?')):
                    pause = 0.3
                time.sleep(pause)
            except Exception as e:
                print(f"Hindi playback error: {e}")
                speak_text("क्षमा करें, हिंदी में बोलने में समस्या हो रही है।")  # Fallback message
        else:  # Use pyttsx3 for English
            if "Point" in phrase or phrase.endswith(('!', '?')):
                tts_engine.setProperty('rate', 160)
            else:
                tts_engine.setProperty('rate', 180)
            tts_engine.say(phrase)
            tts_engine.runAndWait()
            if i < len(phrases) - 1:
                time.sleep(0.15 if "Point" not in phrase else 0.3)

def adapt_response(response, learning_style, mode="voice"):
    """Adapt response to learning style with structured speech."""
    prefix_en = {
        "practical": "Let me give you a practical explanation. ",
        "application": "Let me show you how to apply this. ",
        "theory": "Let me explain the theory behind this. "
    }
    prefix_hi = {
        "practical": "मैं आपको एक व्यावहारिक व्याख्या देता हूँ। ",
        "application": "मैं आपको दिखाता हूँ कि इसे कैसे लागू करें। ",
        "theory": "मैं आपको इसके पीछे की सिद्धांत बताता हूँ। "
    }
    prefix = prefix_hi if preferred_language == "hindi" or is_hindi(response) else prefix_en
    cleaned_response = clean_response_text(response)
    structured_response = f"{prefix.get(learning_style, prefix['practical'])} {cleaned_response} ... Anything else you’d like to know?" if preferred_language != "hindi" and not is_hindi(response) else f"{prefix.get(learning_style, prefix['practical'])} {cleaned_response} ... और कुछ जानना चाहेंगे?"
    return structured_response

def generate_response(query, user_name, learning_style):
    """Generate response using Gemini API, respecting language preference."""
    global conversation_history
    
    query_lower = query.lower()
    if query_lower in FALLBACK_RESPONSES:
        response_text = FALLBACK_RESPONSES[query_lower].get(learning_style, FALLBACK_RESPONSES[query_lower]["practical"])
        conversation_history.append({"role": "Human", "text": query, "time": datetime.now().strftime("%H:%M:%S")})
        conversation_history.append({"role": "Assistant", "text": response_text, "time": datetime.now().strftime("%H:%M:%S")})
        return response_text

    style_context = learning_styles.get(learning_style, learning_styles["practical"])
    history_context = "\n".join([f"{entry['role']}: {entry['text']}" for entry in conversation_history[-5:]])
    
    language_instruction = f"Respond in {preferred_language} if specified; otherwise, use English unless the query is in Hindi."
    
    prompt = f"""Previous conversation:
{history_context}

You are an AI teacher assistant. {style_context}
Keep responses medium-length and focused on study-related topics.
{language_instruction}

Student: {user_name}
Learning Style: {learning_style}
Question: {query}

Respond matching their learning style."""
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        conversation_history.append({"role": "Human", "text": query, "time": datetime.now().strftime("%H:%M:%S")})
        conversation_history.append({"role": "Assistant", "text": response_text, "time": datetime.now().strftime("%H:%M:%S")})
        return response_text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Sorry, I couldn’t process that. Check your API key or internet connection." if preferred_language != "hindi" else "क्षमा करें, मैं इसे प्रोसेस नहीं कर सका। कृपया अपनी API कुंजी या इंटरनेट कनेक्शन जांचें।"

def get_language_preference():
    """Ask user for language preference and set it globally."""
    global preferred_language
    speak_text("Hello! Which language do you prefer: English or Hindi? Say 'English' or 'Hindi'.")
    print("Asking: Which language do you prefer: English or Hindi?")
    
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        try:
            audio = recognizer.listen(source, timeout=10)
            choice = recognizer.recognize_google(audio).lower()
            print(f"You said: {choice}")
            if "hindi" in choice:
                preferred_language = "hindi"
                speak_text("You chose Hindi. Let’s proceed!")
            else:
                preferred_language = "english"
                speak_text("You chose English. Let’s proceed!")
        except (sr.UnknownValueError, sr.WaitTimeoutError, sr.RequestError):
            preferred_language = "english"
            speak_text("I didn’t catch that. Defaulting to English. Let’s proceed!")

def get_voice_input():
    """Record and recognize speech with retry mechanism and 4-second silence detection."""
    recognizer = sr.Recognizer()
    max_attempts = 3
    
    for attempt in range(max_attempts):
        with sr.Microphone() as source:
            print("Listening...")
            if attempt == 0:
                speak_text("Please ask your question now. Take as long as you need." if preferred_language != "hindi" else "कृपया अपना प्रश्न अब पूछें। जितना समय चाहिए लें।")
            else:
                speak_text("Sorry, I didn’t catch that. Please say it again." if preferred_language != "hindi" else "क्षमा करें, मैंने वह नहीं सुना। कृपया फिर से कहें।")
            
            recognizer.adjust_for_ambient_noise(source, duration=2)
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=None)
                recognizer.pause_threshold = 4.0
                query = recognizer.recognize_google(audio, language='hi-IN' if preferred_language == "hindi" else 'en-US')
                print(f"You said: {query}")
                return query
            except sr.WaitTimeoutError:
                print("No speech detected within 10 seconds.")
                if attempt == max_attempts - 1:
                    return "Sorry, I didn’t hear anything after waiting." if preferred_language != "hindi" else "क्षमा करें, मुझे कुछ भी सुनाई नहीं दिया।"
            except sr.UnknownValueError:
                print("Could not understand the speech.")
                if attempt == max_attempts - 1:
                    return "Sorry, I couldn’t understand what you said after trying." if preferred_language != "hindi" else "क्षमा करें, मैं समझ नहीं सका कि आपने क्या कहा।"
            except sr.RequestError as e:
                print(f"Speech error: {e}")
                return "Speech service error. Please check your connection." if preferred_language != "hindi" else "वाक् सेवा में त्रुटि। कृपया अपना कनेक्शन जांचें।"
        
        time.sleep(1)

def process_query(query, user_name, learning_style):
    """Process query and generate response."""
    print(f"Processing: {query}")
    try:
        if query.lower() == "start a quiz":
            if learning_style == "practical":
                response = "You’re building a fence. If one side is 6 meters and the other is 8 meters, what’s the diagonal length?" if preferred_language != "hindi" else "आप एक बाड़ बना रहे हैं। यदि एक तरफ 6 मीटर और दूसरी 8 मीटर है, तो विकर्ण की लंबाई क्या होगी?"
            elif learning_style == "application":
                response = "Design a bridge to hold 100 kilograms. How would you use tension and compression?" if preferred_language != "hindi" else "100 किलोग्राम सहने वाला पुल डिज़ाइन करें। आप तनाव और संपीड़न का उपयोग कैसे करेंगे?"
            else:
                response = "What’s the equation for kinetic energy, and how is it derived?" if preferred_language != "hindi" else "गतिज ऊर्जा का समीकरण क्या है, और यह कैसे निकाला जाता है?"
        elif any(error in query.lower() for error in ["didn’t hear", "couldn’t understand", "speech service error"]):
            return query
        else:
            response = generate_response(query, user_name, learning_style)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, something went wrong." if preferred_language != "hindi" else "क्षमा करें, कुछ गलत हो गया।"

def voice_interaction_loop(user_name, learning_style):
    """Main loop for voice interaction."""
    greeting = "Good morning" if datetime.now().hour < 12 else "Good afternoon" if datetime.now().hour < 17 else "Good evening"
    hindi_greeting = "सुप्रभात" if datetime.now().hour < 12 else "नमस्ते" if datetime.now().hour < 17 else "शुभ संध्या"
    initial_greeting = f"{greeting}, {user_name}! I’m here to help with your studies, tailored to your {learning_style} style. Ask me anything!" if preferred_language != "hindi" else f"{hindi_greeting}, {user_name}! मैं आपके अध्ययन में मदद के लिए हूँ, आपकी {learning_style} शैली के अनुसार। कुछ भी पूछें!"
    print(initial_greeting)
    speak_text(initial_greeting)

    while True:
        query = get_voice_input()
        print(f"Query: {query}")
        
        if "thank you" in query.lower() or "धन्यवाद" in query.lower():
            speak_text("Goodbye. I hope I was helpful!" if preferred_language != "hindi" else "अलविदा। मुझे आशा है कि मैं मददगार रहा!")
            print("Session ended.")
            break
        
        response = process_query(query, user_name, learning_style)
        if "sorry" in response.lower() or "क्षमा करें" in response.lower():
            print(f"Response: {response}")
            speak_text(response)
        else:
            adapted_response = adapt_response(response, learning_style)
            print(f"Response: {adapted_response}")
            speak_text(adapted_response)
        time.sleep(1)

def main():
    """Start the application."""
    load_face_encodings()
    load_student_data()
    
    user_name = identify_person()
    get_language_preference()
    learning_style = get_learning_style(user_name)
    print(f"User: {user_name}, Learning Style: {learning_style}, Language: {preferred_language}")
    
    voice_interaction_loop(user_name, learning_style)

if __name__ == "__main__":
    main()