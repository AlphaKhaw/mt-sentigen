# MT-SentiGen

## End-Product: `AI-Driven Customer Review Analysis`

![Screenshot 2023-09-17 at 2 51 03 PM](https://github.com/AlphaKhaw/mt-sentigen/assets/87654386/2834b1ae-5ad0-4150-a688-acadc90ab2db)

### **Objective**

The `AI-Driven Customer Review Analysis` system serves as a comprehensive tool designed for businesses, researchers, and analysts who are keen on understanding customer sentiments and feedback. It goes beyond traditional text analytics by leveraging state-of-the art Large Language Models (LLM) to provide not only sentiment scores but also context-rich analyses as well as automated responses generation. This includes extracting potential areas for improvement and noteworthy criticisms from raw customer reviews.

### **Features**

**1. Sentiment Analysis:**
Offers nuanced categorizations of reviews into positive, neutral, or negative sentiments, taking into account the complexity and multifaceted nature of human emotions.

**2. Context-rich Analysis:**
Identifies specific themes, trends, and business entities mentioned in reviews, allowing for a more focused and action-oriented business strategy. Highlights specific areas where a business excels or requires improvement.

**3. Automated Generated Response:**
Identifies specific themes, trends, and business entities mentioned in reviews, allowing for a more focused and action-oriented business strategy.
Efficiently manage response to reviews and maintain brand voice.

**4. User-friendly Interface:**
Provides a straightforward and accessible UI where even non-technical users can input data as plain text or upload in CSV format for bulk processing.

### Who Should Use This?

- **Business Analysts and Data Scientists:** Professionals in these roles can leverage the granular insights for quantitative and qualitative analyses, thereby aiding in strategic decision-making.

- **Market Researchers:** Academics and corporate researchers can benefit from the system's scalability to analyze large public opinions and trends.

- **Business Owners and Product Managerss:** Individuals in managerial and executive roles will find the actionable insights invaluable for shaping future products and strategies, enhancing customer satisfaction, and identifying growth opportunities.


## Repository: `MT-SentiGen`

### Objective

The `MT-SentiGen` repository serves as a comprehensive framework for executing a full-cycle, multitask AI project. Starting from data acquisition to model training and eventually to production deployment, this repository offers an industrial-grade architecture. It features the integration of two distinct machine learning paradigms: supervised Multitask Learning via the Text-to-Text Transfer Transformer (T5) and state-of-the-art Large Language Models (LLMs) like Llama2-7B-Chat. Deployed seamlessly on Streamlit and AWS, this repository also incorporates Continuous Integration/Continuous Deployment (CI/CD) processes to ensure software development best practices.

### Features

1. **End-to-End Pipeline:**
Incorporates ETL operations, data preprocessing, model training, and production deployment, providing a seamless workflow from data to insights.

2. **Multitask Model with T5 and LLMs:**
Incorporates a T5 multitask model for specialized tasks, and augments it with the generalized capabilities of Llama2-7B-Chat for a robust analytical engine.

3. **Scalability:**
Built on a microservices architecture to support both vertical and horizontal scaling, allowing for robustness and extensibility.

4. **Streamlit Integration:**
Utilizes Streamlit for front-end development, creating a responsive and intuitive user interface for interacting with the backend models.

5. **AWS Deployment:**
Optimized for cloud deployment on AWS, taking advantage of elastic scaling, data durability, and other cloud-native features.

6. **CI/CD Implementation:**
Adopts CI/CD methodologies using tools like Jenkins or GitHub Actions for automated testing and deployment, ensuring the codebase remains robust and deployable at all times.


<!-- ## Documentation -->


<!--
## Installation

```bash
# Clone the repository
git clone https://github.com/AlphaKhaw/mt-sentigen.git

# Navigate into the project directory
cd mt-sentigen

# Install dependencies
 -->

## References

### Dataset

- [Google Local Data (2021)](https://jiachengli1995.github.io/google/index.html)

### Research Papers

- [A Survey of Multi-task Learning in Natural Language Processing: Regarding Task Relatedness and Training Methods](https://aclanthology.org/2023.eacl-main.66.pdf)
- [Solving Aspect Category Sentiment Analysis as a Text Generation Task](https://aclanthology.org/2021.emnlp-main.361.pdf)
- [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/abs/1706.05098)
- [Multi-Task Learning of Generation and Classification for Emotion-Aware Dialogue Response Generation](https://aclanthology.org/2021.naacl-srw.15/)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- []()

### Repository

- [Emotion-Aware Dialogue Response Generation by Multi-Task Learning](https://github.com/nlp-waseda/mtl-eadrg)
- [Awesome Multi-Task Learning](https://github.com/Manchery/awesome-multi-task-learning)
- [Multi-Task Deep Neural Networks for Natural Language Understanding](https://github.com/namisan/mt-dnn)

### Models

- [Hugging Face Transformers Documentation - T5](https://huggingface.co/docs/transformers/model_doc/t5)
- [GPT4All](https://github.com/nomic-ai/gpt4all)
- [LlaMA-CPP](https://github.com/ggerganov/llama.cpp)
- [Python Bindings for LlaMA-CPP](https://github.com/abetlen/llama-cpp-python)
