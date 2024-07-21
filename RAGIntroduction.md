
#  Retrieval Augmented Generation

## What is Retrieval Augmented Generation (RAG)?

RAG is a technique that helps in creating context specific prompts using external knowledge sources. These prompts are send to a Large Language Model (LLM) to receive context rich response.

## How is it different to Prompt Engineering?

Prompt engineering involves carefully crafting prompts which are then provided to a LLM to steer its responses in a desired direction and/or in a desired format. If the context can be set through prompts then we can get context specific response but this might not be sufficient for certain scenarios.

Imagine a scenario where we have a huge corpora of data (giga bytes of collection of documents) and we want to ask specific questions regarding the content in these documents. As the volume of data is huge, we might not be sure which section of document(s) contain the relevant content.

Naturally, the first thing we try, is to copy content from all the documents and use it to retrieve the required response from the LLM. As the amount of data could be huge and each LLM request might have a limited context length, this might restrict the amount of content that can be send in a single request to a LLM. Moreover, it could be time consuming to go through each document and retireve the contextual data. To cater to all these issues would require a bit of ingenuity to keep the number of requests to a minimum while including relevant data source in the prompt. This would require us to:

* Build a search mechanism that can retrieve relevant sections of the data for creating the prompt.

* Create multiple requests and provide relevant context to subsequent requests send to the LLM.

* An effective way of creating the prompt which might require multiple iterations.

This is what RAG does. RAG can be considered as an advanced prompt creation framework which retrieves relevant data from the external knowledge source(s) provided and creates the relevant prompt to be send to the LLM.

## How is it different to LLM fine-tuning?

Fine-tuning a LLM further trains a pre-trained model on a specific dataset to adapt its knowledge and behavior. This allows to train a LLM to specific data. Although fine-tuned LLMs might be trained for specific data, RAG can still be beneficial for retrieving context specific data.

For example; Suppose a company has a privately hosted LLM that has been fine-tuned based on company data.

* **Contextual Understanding**: Different sub-organizations or teams within the company may have specific contexts, jargon, or domain knowledge that are not covered comprehensively in the fine-tuned LLM. By incorporating retrieval-based methods through RAG, these teams can leverage their specific context to enhance the generation of responses.

* **Customization and Adaptation**: Each team can tailor the retrieval component of RAG to retrieve information relevant to their specific needs, ensuring that the generated responses are aligned with their context and requirements.

* **Privacy and Security**: Since the fine-tuned LLM is hosted privately within the company, there might be restrictions or privacy concerns about sharing sensitive data across different teams within the organization. RAG allows teams to leverage the knowledge encoded in the fine-tuned LLM while still maintaining control over their proprietary data.

* **Performance Optimization**: RAG can enhance the performance of the fine-tuned LLM by providing additional context and information during response generation.

Some more examples where RAG can be useful:

* A specific set of medical journal on which a user wants to ask queries.

* A specific set of legal documents on which a user wants to assimilate information or ask questions.

## Conclusion

RAG can be effectively utilized in conjunction with Prompt Engineering and LLM fine-tuning where we want to assimilate information based on a specific set of content(s).

## References

1. [Prompting Guide](https://www.promptingguide.ai/)

1. [Retrieval Augmented Generation — Intuitively and Exhaustively Explained](https://towardsdatascience.com/retrieval-augmented-generation-intuitively-and-exhaustively-explain-6a39d6fe6fc9)

1. [Retrieval Augmented Generation (RAG) with Llama Index and Open-Source Models](https://christophergs.com/blog/ai-engineering-retrieval-augmented-generation-rag-llama-index)
