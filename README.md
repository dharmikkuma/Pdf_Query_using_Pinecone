# Question-Answering with PDF Documents using Pinecone Vector Database

## Overview

This project is to chat with the PDF document using the pinecone vector store. you can ask anything to the model and it will provide the answer if the answer is available in the PDF document.


## Getting Started

### Prerequisites

Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/dharmikkuma/Pdf_Query_using_Pinecone.git
    ```

2. Navigate to the project directory:

    ```bash
    cd your_project
    ```

3. Run the Python script to set up the project:

    ```bash
    pip install -r requirement.txt
    ```

    This will install the necessary dependencies listed in `requirements.txt`. make sure you are using correct python environment or create new one

4. Set the Pinecone API keys and index host URL as environment variables:

    ```bash
    export API_KEY=your_api_key
    export HOST_URL = yout_index_host_url
    ```

    Replace `your_api_key` with your actual API key and host url.

### Usage
1. First you need to load the pdf text and split it into chunks.
2. Then convert these chunks into embedding vectors using embedding model.
3. now upsert the vectors to pinecone database.
4. run store_index.py file to complete step 1, 2 and 3. 
5. Run the main application file to start the project:

```bash
python app.py
