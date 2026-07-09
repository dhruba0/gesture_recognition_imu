from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
model_name,
torch_dtype="auto",
device_map="auto"
)



job_description = """
We are hiring a Senior Data Scientist.

Required:
- Python
- Machine Learning
- AWS
- SQL
- LLM experience

Preferred:
- MLOps
- Docker
"""
resume = """
Senior Data Scientist with 7 years of experience.

Skills:
Python, AWS SageMaker, Machine Learning,
Deep Learning, PostgreSQL, LangChain

Built several generative AI applications.
"""
prompt = f"""
You are an expert recruiter.

Job Description:
{job_description}

Resume:
{resume}

Return:

1. Match Score (0-100)
2. Strengths
3. Weaknesses
4. Hiring Recommendation

Format as JSON.
"""
{
"match_score": 91,
"strengths": [
"Strong Python background",
"AWS experience",
"LLM application development"
],
"weaknesses": [
"Docker not explicitly mentioned"
],
"recommendation": "Strong Hire"
}
``

