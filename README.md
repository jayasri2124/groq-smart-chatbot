python -m venv venv
venv\Scripts\activate      
source venv/bin/activate   
pip install -r requirements.txt
create a .env 
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
python app.py

