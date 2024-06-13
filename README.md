# ABSA - MachineLearning
## **Table of content:**
- [Our Project](#item-one)
- [Machine Learning Part](#item-two)
- [About Dataset](#item-three)
- [How it works](#item-four)
- [Our chosen models](#item-five)
- [Further Improvement](#item-six)
- [Contributors](#item-seven)
  
<!-- headings -->
<a id="item-one"></a>
## Our Project
Our project is to develop a comprehensive system that combines aspect extraction and sentiment analysis using an ML approach. Using the labeled data as positive, negative, or neutral enables us to provide detailed information on how customers respond to each product or service aspect. We will present the results in an interactive dashboard in various forms of data visualization. We'll also integrate the Translate API to understand user perceptions across languages Our system will accept input data in various formats for flexibility including text, CSV, and XLSX. The mobile-friendly dashboard will allow easy access, available in both English and Bahasa for a wider audience. 

<a id="item-two"></a>
## Machine Learning Part
This section provides an overview of the machine learning pipeline used in the Aspect-Based Sentiment Analysis (ABSA) dashboard for restaurant reviews. The pipeline involves several key steps to preprocess data, extract and classify aspects within review texts, and produce structured outputs.

<a id="item-three"></a>
## About Dataset
The dataset we use consists of restaurant reviews acquired from Bizzagi. Each review has been manually labeled by our team to include the following information:
- **Review**: The restaurant review sentence or text containing the aspects.
- **Span**: A specific phrase or word within the review text relevant to the aspect category. For example, in the review “Makanannya enak tapi pelayanan buruk”, spans could be “Makanannya” and “pelayanan”.
- **Aspect**: Categorizes the span into predefined aspects such as:
  - **Rasa** (Food Quality): Taste and quality of the food.
  - **Pelayanan** (Service): Quality of the service provided.
  - **Harga** (Pricing): Cost or value for money.
  - **Tempat** (Ambience): Atmosphere and environment of the restaurant.
- **Sentiment**: Indicates the sentiment polarity associated with the span including:
  - **Positif**: Favorable opinion.
  - **Negatif**: Unfavorable opinion.
  - **Netral**: Neutral opinion.
- **Ordinal**: Represents the index of occurrences of the aspect span in the review. For example: If “rasanya” span appears once in the review, the ordinal will be 0.

<a id="item-four"></a>
## How it works
![](https://github.com/Bangkit-2024/ABSA_MachineLearning/blob/main/assets/How%20it%20works.png)

1. **Preprocess CSV File**  
  The process begins by reading a CSV file containing restaurant reviews. The data is then preprocessed to extract the relevant review texts while removing any unnecessary columns. This step prepares the data for further analysis by ensuring that only the essential information is retained.

2. **Extract Spans and Polarity from Reviews**  
  Using a pre-trained Indo Setfit model, specific spans (phrases or words) within the review texts are identified. Each span is analyzed to determine its sentiment polarity, such as positive or negative. The input for this step includes the 'review', 'span', and 'ordinal' columns from the dataset.

3. **Classify Spans into Aspect**  
  The necessary models are initialized, including an Indo BERT model for generating embeddings and a BiLSTM model for classifying aspects. The extracted spans are converted into numerical embeddings using the IndoBERT model. These embeddings, along with the 'aspect' column from the dataset, are then inputted into the BiLSTM model to predict the aspect categories for each span.

4. **Combine Predictions**  
The results from the ABSA model and the BiLSTM model are merged. This involves combining the original review texts, the predicted aspects, and their associated sentiments into a cohesive and structured format. This step ensures that all relevant information is consolidated into a single dataset.

<a id="item-five"></a>
## Our chosen models
a. **Pre-trained Indo Setfit**  
Pre-trained Indo Setfit is a model that has been pre-trained and optimized for specific tasks in Indonesian. It is derived from Hugging Face which is specifically designed for ABSA and enables sentiment analysis of specific aspects in Bahasa Indonesia. This model is used for more specific sentiment analysis where the model determines not only the overall sentiment but also the sentiment toward certain aspects of the text.

While the pre-trained model already offers strong performance, we fine-tuned it using our specific dataset to tailor its predictions to our particular use case. This involved:
  - Data Preparation: Prepare the model to receive files in the form of CSV and convert them into data frames to be processed at the next stage.
  - Training: Running several epochs of training where the model adjusts its weights based on our dataset, enhancing its ability to predict sentiments for our specific needs accurately.

Based on benchmarking and performance evaluation, the ABSA-setfit model shows better accuracy and robustness in ABSA tasks than other models that are not tuned for Indonesian. This can be seen from the accuracy of the model as follows.

After training, the accuracy for `aspect` is 0.93 while for `polarity` the accuracy is 0.90.

b. **IndoBERT**  
IndoBERT is used to obtain an embedding representation of the review text. This embedding is a vector representation of the text that captures the contextual and semantic information of the text. This is because IndoBERT, as a transformer-based model trained specifically on Indonesian texts, is very effective in understanding the nuances and context of review texts in Indonesian. This is important for generating accurate representations of the text for advanced tasks.

The IndoBERT used in our second model works as follows:
  - Tokenization: Using IndoBERT's tokenizer to convert text into tokens that can be processed by the model.
  - Embedding: The IndoBERT model generates embedding from the review text. This embedding is then used as input to the LSTM model for aspect prediction.

c. **BiLSTM**
The LSTM in the second model has uses related to aspect prediction and embedding classification. The LSTM model was trained to predict aspects of the embedding generated by IndoBERT. The predicted aspects include 'price', 'service', 'taste', and 'place'. Next, the LSTM Model takes the embedding from IndoBERT and processes it to classify the text into predefined aspects.

The BiLSTM used in our second model works as follows: 
Input Embedding: Embedding generated by IndoBERT is fed to the LSTM model.
Prediction: The LSTM processes the embedding to output aspect predictions. The result of this prediction is a label that describes the aspect of the review text.

Here are the custom layers for BiLSTM model
![](https://github.com/Bangkit-2024/ABSA_MachineLearning/blob/main/assets/BiLSTM%20stucture.png)

After training, the **accuracy** for this model is 1.00.
![](https://github.com/Bangkit-2024/ABSA_MachineLearning/blob/main/assets/BiLSTM%20confusion%20matrix.png)

<a id="item-six"></a>
### Further Improvement
From the resulting model, we plan to improve its performance and functionality by adding more data on certain aspects and sentiments to increase its accuracy. We aim to create a model that can analyze aspects of a review and predict its sentiment more accurately.

<a id="item-seven"></a>
### Contributors
M006D4KX2294 [Elfira Maya Shofwah](https://github.com/elfirams)
M006D4KX2292 [Kayla Aisya Zahra](https://github.com/kaylaisya)
