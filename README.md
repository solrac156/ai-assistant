# Wine AI Assistant

This project aims to create a Conversational AI Assistant 
that can help with the recommendation, description and details 
of a dataset of wine tastings.

It's built using Langchain and Azure Search AI.

For this project to work, it is necessary to deploy an instance of
a search service and an instance of a language service in Azure.

## Usage

This project uses Python 3.11.3. Here's how you set it up:

Install Python 3.11.3 from [here](https://www.python.org/downloads/)

The project uses the following Python packages:
- Azure AI Text Analytics
- Black
- Langchain
- OpenAI
- Pandas
- Python Slugify
- Python Dotenv
- Azure Search Documents
- Tiktoken
- Azure Identity

You can install the packages using pip:
```bash
pip install -r requirements.txt
```
Also, you need to deploy an instance of a Search Service
and an instance of Language Service, configure them and 
create a `.env` file like the `.env.example` provided
and set up the corresponding variables.

Once that is done, you can process the information, with 
the script at `data/prepare_data.py`:
```bash
cd data/
python prepare_data.py
```
Once that is done, you will have markdown versions of all
the rows in the csv.

Before you can use your assistant, you need to process a few
documents. You can do this by doing:

```python
from services.processor import DocumentProcessor
from dotenv import load_dotenv

load_dotenv()

processor = DocumentProcessor()
filenames = [
    # all the filenames that you want to process and 
    # store in the vector database
]
ids = [] # to save the ids of the processed documents, if you ever need them.
for filename in filenames:
    ids.extend(processor.process(filename))
```

Once that is done, you can run either 
the `conversacional_rag.py` or the `rag.py`
and you can chat with
the assistant or retrieve information
respectively.