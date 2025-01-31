from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Enable CORS for all origins (useful for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for the course data
class Course(BaseModel):
    Name: str
    Description: str
    Topics_Covered: str
    Subject: str
    Difficulty: str

# Sample data
courses_data = [
    {
        "Name": "Building Your First AI Model",
        "Description": "Step-by-step guide to building your first AI model using Python and TensorFlow.",
        "Topics_Covered": "AI Models, TensorFlow",
        "Subject": "AIML",
        "Difficulty": "Medium"
    },
    {
        "Name": "Advanced Deep Learning Architectures",
        "Description": "Explore advanced architectures like GANs, transformers, and autoencoders.",
        "Topics_Covered": "GANs, Transformers, Autoencoders",
        "Subject": "AIML",
        "Difficulty": "Hard"
    },
    {
        "Name": "Reinforcement Learning Basics",
        "Description": "Get introduced to reinforcement learning and its applications in AI systems.",
        "Topics_Covered": "Reinforcement Learning",
        "Subject": "AIML",
        "Difficulty": "Hard"
    },
    {
        "Name": "Ethics in AI",
        "Description": "Discussion on the ethical challenges and considerations in AI development.",
        "Topics_Covered": "Ethics, AI Development",
        "Subject": "AIML",
        "Difficulty": "Easy"
    },
    {
        "Name": "Neural Networks Explained",
        "Description": "Dive into the fundamentals of neural networks, their structure, and how they work.",
        "Topics_Covered": "Neural Networks, Deep Learning",
        "Subject": "AIML",
        "Difficulty": "Medium"
    },
    {
        "Name": "Natural Language Processing Basics",
        "Description": "Introduction to natural language processing (NLP), covering text preprocessing and tokenization.",
        "Topics_Covered": "NLP, Text Processing",
        "Subject": "AIML",
        "Difficulty": "Medium"
    },
    {
        "Name": "Hyperparameter Tuning in ML",
        "Description": "Master the art of hyperparameter tuning to optimize machine learning models.",
        "Topics_Covered": "Hyperparameter Tuning, Model Optimization",
        "Subject": "AIML",
        "Difficulty": "Hard"
    },
    {
        "Name": "Future Trends in Machine Learning",
        "Description": "Explore the latest trends and future directions in machine learning.",
        "Topics_Covered": "Trends, Future of AI",
        "Subject": "AIML",
        "Difficulty": "Medium"
    },
    {
        "Name": "Linear Regression in Machine Learning",
        "Description": "Understand linear regression and its applications in machine learning with examples and code walkthroughs.",
        "Topics_Covered": "Linear Regression, Supervised Learning",
        "Subject": "AIML",
        "Difficulty": "Easy"
    }
]

@app.get("/courses", response_model=List[Course], tags=["Courses"])
async def get_courses(
    subject: Optional[str] = Query(None, description="Filter courses by subject (e.g., AIML)"),
    difficulty: Optional[str] = Query(None, description="Filter courses by difficulty (e.g., Easy, Medium, Hard)")
):
    """
    Get a list of courses. Optionally filter by subject and/or difficulty.
    """
    filtered_courses = courses_data

    return filtered_courses

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
