# SynthRAG

SynthRAG is an advanced question-answering system that leverages the power of large language models and retrieval-augmented generation to provide high-quality, detailed responses to user queries.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Components](#components)
6. [Configuration](#configuration)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

SynthRAG combines the strengths of retrieval-based and generative approaches to create a powerful question-answering system. It uses semantic search to find relevant information from a knowledge base and then synthesizes this information with the capabilities of large language models to generate comprehensive, context-aware answers.

## Features

- Multiple answer generation modes:
  - Direct answering
  - Zhihu-style instructed answering
  - Chain-of-Thought (CoT) answering
  - Similar single-example based answering
  - Similar multi-example based answering
  - ARI framework generation (long-form)
  - ARI framework generation (short-form)
- Semantic search for finding relevant information
- Outline generation for structured responses
- Multi-threaded processing for improved performance
- Streaming output for real-time response generation
- Markdown formatting for easy readability
- Customizable prompts and instructions

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/SynthRAG.git
   cd SynthRAG
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - Create a file named `.env` in the project root
   - Add your API keys:
     ```
     DASHSCOPE_API_KEY=your_dashscope_api_key
     ```

## Usage

To run the Streamlit app:

```
streamlit run main.py
```

This will start the web interface where you can interact with SynthRAG.

## Components

### SynthRAG Class (model.py)

The core of the system, responsible for:
- Generating outlines
- Processing queries
- Managing the question-answering pipeline

### Utility Functions (utils.py)

Contains helper functions for:
- API calls
- Embedding generation
- Semantic search
- Multi-threaded query processing

### Main Application (main.py)

The Streamlit-based user interface, allowing users to:
- Input queries
- Select answer generation modes
- View and download generated responses

## Configuration

You can customize various aspects of SynthRAG:

- Adjust the `few_shot_num` and `max_example_len` in the `SynthRAG` class
- Modify the prompts in `prompt_template.py` (not shown in the provided code)
- Change the `max_workers` for multi-threading in the `SynthRAG` class

## Contributing

Contributions to SynthRAG are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request

## License

[Specify your chosen license here]

---

For more information or support, please [contact details or link to issues page].
