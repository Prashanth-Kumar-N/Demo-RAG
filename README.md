# Demo-RAG

PyMuPDF for reading documents

Spacy for tokenization

Tiktoken for counting tokens

curl -X POST "http://localhost:8000/get_response?query=To%lower%the%platform%using%the%auxiliary%lowering%switch,%what%should%be%done%first?"

curl -X POST http://localhost:8001/get_response \
  -H "Content-Type: application/json" \
  -d '{"query":"To lower the platform using the auxiliary lowering switch, what should be done first?"}'
