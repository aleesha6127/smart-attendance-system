import os
import pickle
from typing import List, Dict, Any
from datetime import datetime
import PyPDF2
import docx

class SimpleAIAssistant:
    def __init__(self):
        """
        Initialize a simple rule-based AI assistant for MCA topics
        """
        # Simple knowledge base for MCA topics
        self.knowledge_base = {
            "programming": {
                "definition": "Programming is the process of creating a set of instructions that tell a computer how to perform a task.",
                "examples": ["Python", "Java", "C++", "JavaScript"],
                "importance": "Programming is fundamental to computer science and software development."
            },
            "python": {
                "definition": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
                "features": ["Easy to learn", "Interpreted", "Object-oriented", "Dynamically typed", "Extensive libraries"],
                "applications": ["Web development", "Data science", "Machine learning", "Automation", "Scripting"],
                "libraries": ["NumPy", "Pandas", "Matplotlib", "TensorFlow", "Django", "Flask"]
            },
            "cloud computing": {
                "definition": "Cloud computing is the delivery of computing services over the internet ('the cloud') to offer faster innovation, flexible resources, and economies of scale.",
                "models": ["IaaS (Infrastructure as a Service)", "PaaS (Platform as a Service)", "SaaS (Software as a Service)"],
                "deployment_types": ["Public cloud", "Private cloud", "Hybrid cloud", "Community cloud"],
                "benefits": ["Cost effectiveness", "Scalability", "Accessibility", "Reliability", "Maintenance"],
                "providers": ["Amazon AWS", "Microsoft Azure", "Google Cloud Platform", "IBM Cloud"]
            },
            "algorithm": {
                "definition": "An algorithm is a step-by-step procedure for solving a problem or completing a task.",
                "characteristics": ["Input", "Output", "Definiteness", "Finiteness", "Effectiveness"],
                "types": ["Sorting", "Searching", "Graph algorithms", "Dynamic programming"]
            },
            "data structure": {
                "definition": "A data structure is a way of organizing and storing data in a computer so that it can be accessed and modified efficiently.",
                "types": ["Arrays", "Linked Lists", "Stacks", "Queues", "Trees", "Graphs", "Hash Tables"],
                "applications": ["Database management", "Operating systems", "Compiler design"]
            },
            "database": {
                "definition": "A database is an organized collection of structured information, or data, typically stored electronically in a computer system.",
                "types": ["Relational", "NoSQL", "Object-oriented", "Graph databases"],
                "sql": "Structured Query Language (SQL) is used to communicate with databases."
            },
            "software engineering": {
                "definition": "Software engineering is the systematic application of engineering approaches to software development.",
                "processes": ["Requirements", "Design", "Implementation", "Testing", "Maintenance"],
                "methodologies": ["Agile", "Waterfall", "DevOps", "Scrum"]
            },
            "computer network": {
                "definition": "A computer network is a set of computers sharing resources located on or provided by network nodes.",
                "types": ["LAN", "WAN", "MAN", "PAN"],
                "protocols": ["TCP/IP", "HTTP/HTTPS", "FTP", "SMTP"]
            },
            "operating system": {
                "definition": "An operating system is system software that manages computer hardware, software resources, and provides common services for computer programs.",
                "functions": ["Process management", "Memory management", "File system management", "Security"],
                "examples": ["Windows", "Linux", "macOS", "Unix"]
            }
        }
        
        # Load MCA syllabus documents if available
        self.load_mca_documents()
    
    def load_mca_documents(self, data_path=None):
        """
        Load MCA syllabus documents from the specified path
        """
        if data_path is None:
            # Use path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, "rag", "mca_syllabus")

        if not os.path.exists(data_path):
            # Create the directory if it doesn't exist
            os.makedirs(data_path, exist_ok=True)
            print(f"Created directory: {data_path}")
            print("Please add MCA syllabus documents to this directory")
            return
        
        # Load documents from the directory
        for filename in os.listdir(data_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.read()
                        # Add document content to knowledge base based on filename
                        if 'python' in filename.lower():
                            self.knowledge_base['python']['document_content'] = content
                        elif 'cloud' in filename.lower():
                            self.knowledge_base['cloud computing']['document_content'] = content
                except Exception as e:
                    print(f"Error loading document {filename}: {e}")
    
    def find_relevant_topic(self, question: str) -> str:
        """
        Find the most relevant topic based on keywords in the question
        """
        question_lower = question.lower()
        
        # Keyword mapping to topics
        topic_keywords = {
            "programming": ["program", "programming", "code", "coding", "language"],
            "python": ["python", "django", "flask", "pandas", "numpy", "machine learning", "data science", "automation"],
            "cloud computing": ["cloud", "aws", "azure", "gcp", "scalability", "iaas", "paas", "saas", "cloud computing", "amazon web services"],
            "algorithm": ["algorithm", "algorithms", "sort", "search", "efficiency", "complexity"],
            "data structure": ["data structure", "structure", "array", "list", "tree", "graph", "hash"],
            "database": ["database", "databases", "sql", "dbms", "relational", "query"],
            "software engineering": ["software engineering", "engineering", "design", "development", "methodology", "process"],
            "computer network": ["network", "networks", "tcp", "ip", "protocol", "internet"],
            "operating system": ["operating system", "os", "process", "memory", "management"]
        }
        
        # Score each topic based on keyword matches
        scores = {}
        for topic, keywords in topic_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in question_lower:
                    score += 1
            scores[topic] = score
        
        # Find the topic with the highest score
        best_topic = max(scores, key=scores.get)
        if scores[best_topic] > 0:  # Only return matched topic if we found at least one keyword
            return best_topic
        
        # If no specific topic found, return general help
        return "general"
    
    def _clean_markdown(self, text: str) -> str:
        """
        Remove markdown formatting from text
        """
        import re
        # Remove markdown headers (# ## ###)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Remove bold/italic markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        return text
    
    def get_answer(self, topic: str) -> str:
        """
        Get answer based on the identified topic
        """
        if topic == "general":
            return ("I'm your AI tutor for MCA subjects. I can help explain concepts related to programming, "
                   "algorithms, data structures, databases, software engineering, computer networks, "
                   "and operating systems. Please ask a specific question about any of these topics!")

        if topic in self.knowledge_base:
            info = self.knowledge_base[topic]
            
            # Check if we have document content for this topic
            if "document_content" in info:
                document_content = info["document_content"]
                # Clean markdown headers
                cleaned_content = self._clean_markdown(document_content)
                answer = cleaned_content[:1000]
                if len(document_content) > 1000:
                    answer += "... For more details, please refer to the complete course materials."
                return answer.strip()
            
            # Build a natural conversational response
            parts = []
            
            if "definition" in info:
                parts.append(info['definition'])
            
            if "examples" in info:
                parts.append(f"Some examples include {', '.join(info['examples'])}.")
            
            if "types" in info:
                parts.append(f"The main types are {', '.join(info['types'])}.")
            
            if "characteristics" in info:
                parts.append(f"Key characteristics are {', '.join(info['characteristics'])}.")
            
            if "features" in info:
                parts.append(f"Important features include {', '.join(info['features'])}.")
            
            if "applications" in info:
                parts.append(f"It's commonly used in {', '.join(info['applications'])}.")
            
            if "importance" in info:
                parts.append(info['importance'])
            
            if "processes" in info:
                parts.append(f"The key processes involved are {', '.join(info['processes'])}.")
            
            if "methodologies" in info:
                parts.append(f"Common methodologies include {', '.join(info['methodologies'])}.")
            
            if "protocols" in info:
                parts.append(f"The protocols used are {', '.join(info['protocols'])}.")
            
            if "functions" in info:
                parts.append(f"Main functions include {', '.join(info['functions'])}.")
            
            if "sql" in info:
                parts.append(info['sql'])
            
            if "libraries" in info:
                parts.append(f"Popular libraries are {', '.join(info['libraries'])}.")
            
            if "providers" in info:
                parts.append(f"Major providers include {', '.join(info['providers'])}.")
            
            if "benefits" in info:
                parts.append(f"Key benefits are {', '.join(info['benefits'])}.")
            
            if "models" in info:
                parts.append(f"Service models include {', '.join(info['models'])}.")
            
            if "deployment_types" in info:
                parts.append(f"Deployment types are {', '.join(info['deployment_types'])}.")
            
            return " ".join(parts) if parts else "I have some information on this topic, but let me know what specific aspect you'd like to learn about."
        
        return "I don't have specific information on that topic. Please ask about programming, algorithms, data structures, databases, software engineering, computer networks, or operating systems."
    
    def is_out_of_syllabus(self, question: str) -> bool:
        """
        Check if a question is out of MCA syllabus
        """
        # Keywords that indicate out-of-syllabus topics
        out_of_syllabus_keywords = [
            'cooking', 'sports', 'entertainment', 'celebrities', 'politics',
            'gossip', 'non-academic', 'personal', 'unrelated', 'random', 'movie',
            'music', 'game', 'vacation', 'holiday', 'food', 'travel'
        ]
        
        question_lower = question.lower()
        for keyword in out_of_syllabus_keywords:
            if keyword in question_lower:
                return True
        
        return False

class AITutor:
    def __init__(self):
        """
        Initialize the AI Tutor system with simple rule-based assistant
        """
        self.simple_assistant = SimpleAIAssistant()
        self.conversation_history = []
        
    def chat(self, user_input: str, user_role: str = "student") -> Dict[str, Any]:
        """
        Main chat function for the AI tutor
        """
        # Validate user role
        if user_role not in ['admin', 'teacher', 'student']:
            user_role = 'student'
        
        # Check if question is out of syllabus
        if self.simple_assistant.is_out_of_syllabus(user_input):
            return {
                'answer': "This question appears to be outside the MCA syllabus. I can only help with MCA-related topics like programming, algorithms, data structures, databases, software engineering, computer networks, and operating systems.",
                'confidence': 0.0,
                'sources': [],
                'relevant_context': "",
                'is_out_of_syllabus': True
            }
        
        # Find relevant topic and get answer
        topic = self.simple_assistant.find_relevant_topic(user_input)
        answer = self.simple_assistant.get_answer(topic)
        
        # Create response
        response = {
            'answer': answer,
            'confidence': 0.8,  # Fixed confidence for rule-based system
            'sources': [{'filename': 'mca_knowledge_base', 'content': topic}],
            'relevant_context': answer[:500] + "..." if len(answer) > 500 else answer,
            'is_out_of_syllabus': False
        }
        
        # Add to conversation history
        self.conversation_history.append({
            'user_input': user_input,
            'response': response,
            'timestamp': str(datetime.now()),
            'user_role': user_role
        })
        
        # Limit conversation history to last 10 exchanges
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response
    
    def get_emotion_state(self, user_input: str) -> str:
        """
        Determine the emotional state based on user input for animated mascot
        """
        # Simple emotion detection based on keywords
        negative_keywords = ['confused', 'don\'t understand', 'hard', 'difficult', 'frustrated', 'stuck', 'help', 'not clear']
        positive_keywords = ['thank you', 'thanks', 'great', 'good', 'ok', 'understand', 'perfect', 'awesome', 'amazing', 'clear']
        questioning_keywords = ['what', 'how', 'why', 'when', 'explain', 'define', 'describe', 'difference', 'advantage', 'disadvantage']
        
        user_lower = user_input.lower()
        
        if any(keyword in user_lower for keyword in negative_keywords):
            return "confused"
        elif any(keyword in user_lower for keyword in positive_keywords):
            return "happy"
        elif any(keyword in user_lower for keyword in questioning_keywords):
            return "thinking"
        else:
            return "neutral"

# For testing purposes
if __name__ == "__main__":
    # Initialize the AI tutor
    ai_tutor = AITutor()
    
    print("Simple AI Tutor initialized! Type 'quit' to exit.")
    print("Try asking questions about programming, algorithms, data structures, databases, etc.\n")
    
    while True:
        user_input = input("Ask a question about MCA syllabus: ")
        if user_input.lower() == 'quit':
            break
            
        response = ai_tutor.chat(user_input, 'student')
        print(f"\nAnswer: {response['answer']}")
        print(f"Confidence: {response['confidence']:.2f}")
        if response['sources']:
            print(f"Topic: {response['sources'][0]['content']}")
        print("-" * 50)