from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins (useful for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

courses_data = [
    {
        "Name": "Building Your First AI Model",
        "Description": "Step-by-step guide to building your first AI model using Python and TensorFlow.",
        "Topics Covered": "AI Models, TensorFlow",
        "Subject": "AIML",
        "Difficulty": "Medium"
    },
    {
        "Name": "Advanced Deep Learning Architectures",
        "Description": "Explore advanced architectures like GANs, transformers, and autoencoders.",
        "Topics Covered": "GANs, Transformers, Autoencoders",
        "Subject": "AIML",
        "Difficulty": "Hard"
    },
    {
        "Name": "Reinforcement Learning Basics",
        "Description": "Get introduced to reinforcement learning and its applications in AI systems.",
        "Topics Covered": "Reinforcement Learning",
        "Subject": "AIML",
        "Difficulty": "Hard"
    },
    {
        "Name": "Ethics in AI",
        "Description": "Discussion on the ethical challenges and considerations in AI development.",
        "Topics Covered": "Ethics, AI Development",
        "Subject": "AIML",
        "Difficulty": "Easy"
    },
    {
        "Name": "Neural Networks Explained",
        "Description": "Dive into the fundamentals of neural networks, their structure, and how they work.",
        "Topics Covered": "Neural Networks, Deep Learning",
        "Subject": "AIML",
        "Difficulty": "Medium"
    },
    {
        "Name": "Natural Language Processing Basics",
        "Description": "Introduction to natural language processing (NLP), covering text preprocessing and tokenization.",
        "Topics Covered": "NLP, Text Processing",
        "Subject": "AIML",
        "Difficulty": "Medium"
    },
    {
        "Name": "Hyperparameter Tuning in ML",
        "Description": "Master the art of hyperparameter tuning to optimize machine learning models.",
        "Topics Covered": "Hyperparameter Tuning, Model Optimization",
        "Subject": "AIML",
        "Difficulty": "Hard"
    },
    {
        "Name": "Future Trends in Machine Learning",
        "Description": "Explore the latest trends and future directions in machine learning.",
        "Topics Covered": "Trends, Future of AI",
        "Subject": "AIML",
        "Difficulty": "Medium"
    },
    {
        "Name": "Linear Regression in Machine Learning",
        "Description": "Understand linear regression and its applications in machine learning with examples and code walkthroughs.",
        "Topics Covered": "Linear Regression, Supervised Learning",
        "Subject": "AIML",
        "Difficulty": "Easy"
    }
]

@app.get("/courses", tags=["Courses"])
async def get_courses():
    return courses_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
