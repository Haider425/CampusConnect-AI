from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
import requests
import time
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)

# Enhanced context with more detailed information and variations
TEST_CENTRE_CONTEXT = {
    "full": """
You are the Test Centre Assistant at Ontario Tech University. You help students with test accommodations and bookings.

Locations:
1. North campus: Shawenjigewining Hall (formerly UA), Room 343A
   - Located on the 3rd floor
   - Accessible via elevator and stairs
   - Near the Student Life office
   
2. Downtown campus: Charles Hall, Room 236
   - Located on the 2nd floor
   - Accessible via elevator and stairs
   - Near the Student Services desk

Key Policies:
1. Booking Requirements:
   - All tests must be booked 7 days in advance minimum
   - Students must be registered with Student Accessibility Services (SAS)
   - Each test requires a separate booking through the SAS Portal
   
2. Important Deadlines:
   - Fall 2024 finals registration: November 13, 2024
   - Winter 2025 finals registration: March 17, 2025
   
3. Contact Information:
   - Email: testcentre@ontariotechu.ca
   - Response time: Usually within 1-2 business days
   
4. Accommodations:
   - Must be approved by SAS before booking
   - Need to be renewed each semester
   - Time extensions are automatically applied to online tests

Always maintain a professional tone while being helpful and understanding. For specific cases 
not covered by standard policies, direct students to email testcentre@ontariotechu.ca.
""",

    "location": """
The Test Centre has two convenient locations at Ontario Tech University:

1. North Campus Location:
   - Building: Shawenjigewining Hall (formerly UA Building)
   - Room: 343A (3rd floor)
   - Landmarks: Near Student Life office, accessible via elevator
   
2. Downtown Campus Location:
   - Building: Charles Hall
   - Room: 236 (2nd floor)
   - Landmarks: Near Student Services desk, accessible via elevator

Both locations are fully accessible and equipped with assistive technologies.
""",

    "booking": """
Test Centre Booking Information:

1. Advance Notice Required:
   - All tests/quizzes/midterms: 7 days minimum
   - Fall 2024 finals deadline: November 13, 2024
   - Winter 2025 finals deadline: March 17, 2025

2. Booking Process:
   - Log into SAS Portal
   - Select "Book Assessment"
   - Choose course and test date
   - Verify accommodations
   - Submit request

3. Requirements:
   - Must be registered with SAS
   - Need current accommodation letter
   - One booking per assessment
""",

    "contact": """
Test Centre Contact Information:

Primary Contact:
- Email: testcentre@ontariotechu.ca
- Response time: 1-2 business days

When contacting, please include:
- Your student number
- Course code
- Assessment details
- Specific questions/concerns

For urgent matters during business hours, visit either Test Centre location in person.
""",

    "accommodation": """
Test Centre Accommodation Information:

1. Types of Accommodations:
   - Extended time
   - Quiet space
   - Assistive technology
   - Break time
   - Reader/scribe services

2. Requirements:
   - Current SAS registration
   - Valid medical documentation
   - Approved accommodation letter
   - Semester-by-semester renewal

3. Online vs In-Person:
   - Online: Extra time added automatically in Canvas
   - In-Person: Accommodations provided at Test Centre
""",

    "default": """
As the Test Centre Assistant at Ontario Tech University, provide clear, accurate information 
about test accommodations, bookings, and policies. Be professional yet approachable. 
If unsure, recommend contacting testcentre@ontariotechu.ca.
""",
    
    "conversation": """
You are the friendly Test Centre Assistant at Ontario Tech University. Your role is to:
- Welcome students warmly while maintaining professionalism
- Provide clear, accurate information about Test Centre services
- Show understanding of accommodation needs
- Guide conversations toward practical solutions
- Be patient with questions and concerns
- Offer specific, actionable next steps
""",

    "greeting": [
        "Welcome to Ontario Tech's Test Centre! How can I assist you with accommodations or test bookings today?",
        "Hello! I'm here to help with Test Centre services. What information do you need?",
        "Hi there! I can help you with test bookings, accommodations, and other Test Centre questions. What brings you here today?",
        "Welcome! I'm your Test Centre Assistant. How may I help you with your testing needs?"
    ],

    "farewell": [
        "Thank you for contacting the Test Centre. Don't hesitate to reach out if you need anything else!",
        "I hope I've helped answer your questions. Feel free to email testcentre@ontariotechu.ca for any additional support.",
        "Thanks for your questions! Remember to book your tests at least 7 days in advance. Have a great day!",
        "Glad I could help! Don't forget to check the SAS Portal for your latest accommodation details."
    ],
}

# Expanded FAQ list with more variations and details
faq = [
    {
        "question": "How do I contact the Test Centre?",
        "variations": [
            "what's the test centre email",
            "how can I reach the test centre",
            "test centre contact info",
            "who do I contact about accommodations",
            "test centre phone number",
            "how to get in touch with test centre",
            "contact information for testing"
        ],
        "answer": "The best way to contact the Test Centre is by email at testcentre@ontariotechu.ca. They typically respond within 1-2 business days. For urgent matters, you can visit either Test Centre location during business hours."
    },
    {
        "question": "Where is the Test Centre located?",
        "variations": [
            "test centre building",
            "where can I find the test centre",
            "test centre room number",
            "which building is the test centre in",
            "test centre campus location",
            "downtown test centre",
            "north campus test centre",
            "UA building test centre",
            "Charles Hall test centre",
            "how do I get to the test centre",
            "directions to test centre",
            "where is testing services",
            "test centre floor",
            "building locations"
        ],
        "answer": "The Test Centre has two locations:\n1. North Campus: Shawenjigewining Hall (formerly UA Building), Room 343A (3rd floor, near Student Life)\n2. Downtown Campus: Charles Hall, Room 236 (2nd floor, near Student Services)\n\nBoth locations are accessible via elevator and stairs."
    },
    {
        "question": "What are the booking deadlines for assessments?",
        "variations": [
            "when do I need to book by",
            "test booking deadline",
            "final exam registration deadline",
            "when should I book my test",
            "last day to book",
            "registration cutoff",
            "booking timeline",
            "how far in advance to book",
            "assessment booking deadline",
            "deadline for test booking"
        ],
        "answer": "All tests, quizzes and mid-terms must be booked a minimum of 7 days in advance of your test date. For Fall 2024 finals, the registration deadline is November 13, 2024. For Winter 2025 finals, the deadline is March 17, 2025. Late bookings require special approval and may not be guaranteed."
    },
    {
        "question": "What happens if I miss the booking deadline?",
        "variations": [
            "late test booking",
            "missed the deadline",
            "forgot to book test",
            "past the booking deadline",
            "too late to book",
            "can I still book after deadline",
            "missed registration deadline",
            "deadline passed",
            "overdue booking"
        ],
        "answer": "If you miss the booking deadline, you must contact testcentre@ontariotechu.ca immediately. While late booking requests are reviewed case-by-case, there's no guarantee they will be approved, and some accommodations may not be available. Always try to book at least 7 days in advance to ensure your accommodations can be provided."
    },
    {
        "question": "How do I book accommodated assessments?",
        "variations": [
            "how to book a test",
            "booking process",
            "schedule an exam",
            "make a test booking",
            "book accommodation",
            "schedule assessment",
            "reserve test time",
            "how to register for test",
            "exam booking steps",
            "assessment registration"
        ],
        "answer": "To book an accommodated assessment:\n1. Log into the Student Accessibility Services (SAS) Portal\n2. Click on 'Book Assessment'\n3. Select your course and test date\n4. Verify your accommodations\n5. Submit your booking request\n\nRemember: Each assessment requires a separate booking and must be made at least 7 days in advance."
    },
    {
        "question": "What are the requirements for eligibility?",
        "variations": [
            "who can use test centre",
            "test centre eligibility",
            "accommodation requirements",
            "can I use test centre",
            "who is eligible",
            "qualification for test centre",
            "documentation needed",
            "required for accommodations",
            "registration requirements"
        ],
        "answer": "To be eligible for Test Centre services, you must:\n1. Be registered with Student Accessibility Services (SAS)\n2. Have current medical documentation supporting your accommodations\n3. Be authorized by your Accessibility Specialist\n4. Have approved testing accommodations\n5. Renew your accommodations each semester"
    },
    {
        "question": "How do online assessments work?",
        "variations": [
            "online test accommodations",
            "virtual assessment process",
            "remote testing",
            "canvas accommodations",
            "online exam process",
            "digital assessment",
            "virtual test taking",
            "remote exam accommodations"
        ],
        "answer": "For online assessments:\n1. TCIS coordinates with your instructors\n2. Time-based accommodations (extra time, breaks) are applied automatically in Canvas\n3. You must still book through the SAS Portal\n4. Technical issues should be reported immediately to testcentre@ontariotechu.ca"
    },
    {
        "question": "Will my accommodations be automatically coordinated?",
        "variations": [
            "automatic accommodations",
            "do I need to request accommodations",
            "accommodation renewal",
            "setup accommodations",
            "accommodation coordination",
            "accommodation application",
            "semester accommodation",
            "course accommodations"
        ],
        "answer": "No, accommodations are not automatic. You must:\n1. Opt into accommodations each semester\n2. Review and accept them through the SAS Portal\n3. Send accommodation letters to professors\n4. Book each test separately\n\nIt's your responsibility to ensure accommodations are renewed and properly set up each term."
    }
]

def detect_query_type(user_message):
    """Enhanced query type detection with more context awareness"""
    msg = user_message.lower()
    
    # Location-related terms
    location_terms = [
        'where', 'location', 'building', 'room', 'floor', 'campus', 
        'downtown', 'north', 'ua', 'charles', 'shawenjigewining',
        'find', 'get to', 'directions', 'address', 'situated',
        'located', 'place', 'building', 'which floor', 'what room'
    ]
    
    # Booking-related terms
    booking_terms = [
        'book', 'register', 'sign up', 'schedule', 'appointment',
        'deadline', 'date', 'time', 'slot', 'reservation',
        'when', 'available', 'upcoming', 'test date', 'timing',
        'registration', 'sign-up', 'reserve', 'booking process'
    ]
    
    # Accommodation-related terms
    accommodation_terms = [
        'accommodation', 'extra time', 'extension', 'quiet',
        'modify', 'change', 'update', 'renew', 'letter',
        'sas', 'accessibility', 'service', 'help', 'support',
        'extended time', 'special needs', 'assistance', 'aids'
    ]
    
    # Contact-related terms
    contact_terms = [
        'contact', 'email', 'reach', 'phone', 'call',
        'speak', 'talk', 'ask', 'question', 'inquire',
        'get in touch', 'connect', 'message', 'communicate'
    ]
    
    # Greeting detection
    greetings = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon',
        'good evening', 'greetings', 'howdy', 'hello there',
        'hola', 'morning', 'afternoon', 'evening'
    ]
    
    # Farewell detection
    farewells = [
        'bye', 'goodbye', 'thanks', 'thank you', 'appreciate',
        'helped', 'clear', 'understood', 'got it', 'see you',
        'take care', 'have a good day', 'bye bye'
    ]

    # First, check for greetings and farewells
    if any(greeting in msg for greeting in greetings):
        return 'greeting'
    if any(farewell in msg for farewell in farewells):
        return 'farewell'
        
    # Then check for specific query types
    if any(term in msg for term in location_terms):
        return 'location'
    if any(term in msg for term in booking_terms):
        return 'booking'
    if any(term in msg for term in accommodation_terms):
        return 'accommodation'
    if any(term in msg for term in contact_terms):
        return 'contact'
        
    # If no specific type is detected, use conversation
    return 'conversation'

# Initialize model and cache
print("Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = {q["question"]: model.encode(q["question"]) for q in faq}
for faq_item in faq:
    for variation in faq_item["variations"]:
        question_embeddings[variation] = model.encode(variation)
print("Model loaded and embeddings cached successfully!")

def find_best_match(user_input, threshold=0.5):
    """Find best matching FAQ entry using semantic search"""
    try:
        user_embedding = model.encode(user_input)
        
        # Calculate similarities with all questions and variations
        similarities = {}
        for faq_item in faq:
            # Check main question
            main_similarity = 1 - cosine(user_embedding, question_embeddings[faq_item["question"]])
            similarities[faq_item["answer"]] = max(similarities.get(faq_item["answer"], 0), main_similarity)
            
            # Check variations
            for variation in faq_item["variations"]:
                var_similarity = 1 - cosine(user_embedding, question_embeddings[variation])
                similarities[faq_item["answer"]] = max(similarities.get(faq_item["answer"], 0), var_similarity)
        
        # Find best match
        best_answer = max(similarities.items(), key=lambda x: x[1])
        
        if best_answer[1] < threshold:
            return None
        return best_answer[0]
        
    except Exception as e:
        print(f"Error in find_best_match: {str(e)}")
        return None

def get_llm_response(prompt, max_retries=3):
    """Get response from LLM with enhanced context awareness and retry logic"""
    query_type = detect_query_type(prompt)
    context = TEST_CENTRE_CONTEXT.get(query_type, TEST_CENTRE_CONTEXT["default"])
    
    # Handle greeting and farewell specially
    if query_type == 'greeting':
        import random
        return random.choice(TEST_CENTRE_CONTEXT["greeting"])
    
    if query_type == 'farewell':
        import random
        return random.choice(TEST_CENTRE_CONTEXT["farewell"])

    # Construct prompt based on query type
    if query_type == 'location':
        full_prompt = f"{TEST_CENTRE_CONTEXT['location']}\nQuestion: {prompt}\nProvide specific location details in a clear, helpful way:"
    elif query_type == 'booking':
        full_prompt = f"{TEST_CENTRE_CONTEXT['booking']}\nQuestion: {prompt}\nProvide specific booking information and next steps:"
    elif query_type == 'accommodation':
        full_prompt = f"{TEST_CENTRE_CONTEXT['accommodation']}\nQuestion: {prompt}\nExplain accommodation details clearly:"
    elif query_type == 'conversation':
        full_prompt = f"{context}\nBe friendly but professional. Question: {prompt}\nResponse:"
    else:
        full_prompt = f"{context}\nQ: {prompt}\nA:"

    for attempt in range(max_retries):
        try:
            response = requests.post('http://localhost:11434/api/generate',
                                   json={
                                       "model": "qwen:0.5b",
                                       "prompt": full_prompt,
                                       "stream": False,
                                       "temperature": 0.7,
                                       "max_tokens": 150
                                   },
                                   timeout=60)
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                if result:
                    return result
                
        except Exception as e:
            print(f"LLM error attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)  # Brief pause before retry
            
    return None

def parallel_get_responses(user_message):
    """Get FAQ and LLM responses in parallel with enhanced error handling"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        faq_future = executor.submit(find_best_match, user_message)
       
        llm_future = executor.submit(get_llm_response, user_message)
        
        return faq_future.result(), llm_future.result()

# Add this at the top level of your file
conversation_history = {}

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", "default")

        # Initialize or get conversation history
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Add user message to history
        conversation_history[session_id].append({
            "role": "user", 
            "message": user_message
        })
        
        if not user_message:
            return jsonify({"reply": "Please enter your question about the Test Centre."})
        
        query_type = detect_query_type(user_message)
        is_follow_up = len(conversation_history[session_id]) > 1
        
        # For greetings and general conversation
        if query_type == 'greeting' and not is_follow_up:
            llm_response = get_llm_response(user_message)
            conversation_history[session_id].append({
                "role": "assistant",
                "message": llm_response
            })
            return jsonify({
                "reply": llm_response or "Hello! How can I help you with the Test Centre today?",
                "source": "llm"
            })
        
        # For booking follow-ups
        if is_follow_up and "book" in user_message.lower():
            booking_response = ("To book a test, follow these steps:\n"
                              "1. Log into the Student Accessibility Services (SAS) Portal\n"
                              "2. Select 'Book Assessment'\n"
                              "3. Choose your course and test date\n"
                              "4. Submit your booking at least 7 days in advance\n\n"
                              "Need help with any of these steps?")
            conversation_history[session_id].append({
                "role": "assistant",
                "message": booking_response
            })
            return jsonify({
                "reply": booking_response,
                "source": "system"
            })
        
        # For other queries, use parallel processing
        faq_response, llm_response = parallel_get_responses(user_message)
        
        if faq_response:
            final_response = faq_response
        elif llm_response:
            final_response = llm_response
        else:
            final_response = "I apologize, but I'm not sure about that specific query. How else can I help you with the Test Centre today?"
        
        conversation_history[session_id].append({
            "role": "assistant",
            "message": final_response
        })
        
        return jsonify({
            "reply": final_response,
            "source": "faq" if faq_response else "llm"
        })
        
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return jsonify({
            "reply": "I'm having trouble processing your request. Please try asking your question again.",
            "error": str(e)
        })
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True);