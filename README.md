# Pesto Home Assignment README

## Project Overview

This project focuses on fine-tuning a language model to generate automated responses for customer queries. The notebook demonstrates the entire process, from data preparation to model deployment using the Gradio library for creating an interactive demo.

## Table of Contents

1. [Requirements](#requirements)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Fine-Tuning the Model](#fine-tuning-the-model)
6. [Deploying with Gradio](#deploying-with-gradio)

## Requirements

- Python 3.7+
- Jupyter Notebook or Google Colab
- Transformers library
- Gradio library

## Project Structure

The notebook consists of the following sections:

1. **Setup**: Importing necessary libraries and setting up the environment.
2. **Data Preparation**: Loading and preparing the dataset.
3. **Model Fine-Tuning**: Converting examples to features and fine-tuning the language model.
4. **Inference**: Generating responses using the fine-tuned model.
5. **Deployment**: Creating an interactive demo with Gradio.

## Installation

To run this project locally, follow these steps:

1. Clone the repository or download the notebook file.
2. Install the required libraries:
   ```bash
   pip install transformers gradio
   ```
3. Launch Jupyter Notebook and open the downloaded notebook file.

## Usage

1. **Data Preparation**: Prepare your dataset with columns for queries and responses.
2. **Fine-Tuning**: Run the cells in the notebook to fine-tune the model on your dataset.
3. **Inference**: Use the fine-tuned model to generate responses for new queries.
4. **Gradio Deployment**: Deploy the model using Gradio for an interactive web interface.

## Fine-Tuning the Model

The fine-tuning process involves the following steps:

1. **Load and Prepare Dataset**:
   ```python
   def convert_examples_to_features(example_batch):
       input_encodings = tokenizer(example_batch["query"], max_length=1024, truncation=True, padding=True)
       with tokenizer.as_target_tokenizer():
           target_encodings = tokenizer(example_batch["response"], max_length=128, truncation=True, padding=True)
       return {
           "input_ids": input_encodings["input_ids"],
           "attention_mask": input_encodings["attention_mask"],
           "labels": target_encodings["input_ids"]
       }

   pairs = dataset.map(convert_examples_to_features, batched=True)
   ```

2. **Fine-Tuning**: Fine-tune the model using the prepared dataset.

## Deploying with Gradio

Create an interactive demo using Gradio:

```python
import gradio as gr

def generate_response(query):
    input = tokenizer.encode(query, return_tensors="pt")
    output = model.generate(input, max_length=128)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

demo = gr.Blocks()

with demo:
    gr.Markdown("## Customer Automated Answer Generator ")
    with gr.Tabs():
        with gr.TabItem("Customer Automated Answer Generator"):
            with gr.Row():
                query_inputs = gr.Textbox()
                response_outputs = gr.Textbox()
            summary_button = gr.Button("Generate Response")

    summary_button.click(generate_response, inputs=query_inputs, outputs=response_outputs)

if __name__ == "__main__":
    demo.launch()
```

This code sets up a Gradio interface where users can input a query and receive a generated response from the fine-tuned model.

## Conclusion

This project provides a comprehensive guide to fine-tuning a language model for generating automated responses and deploying it using Gradio for an interactive user interface. Follow the steps outlined in the notebook to replicate the process and customize it for your specific use case.
