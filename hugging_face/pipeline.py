from transformers import pipeline

CPU = -1
GPU = 0

# 选择任务和模型
# https://huggingface.co/models?pipeline_tag=conversational&sort=downloads

# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


# feature-extraction 特征提取 embedding
feature_extraction = pipeline(task="feature-extraction",
                                model="distilbert-base-uncased",
                                device=CPU)
features = feature_extraction("I am happy today!")
print(features)

# fill-mask 填充 [MASK]
fill_mask = pipeline(task="fill-mask",
                        model="distilbert-base-uncased",
                        device=CPU)
preds = fill_mask("I am happy [MASK]!")
print(preds)

# ner 命名实体识别
ner = pipeline(task="ner",
                model="dslim/bert-base-NER",
                device=CPU)
preds = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(preds)

# question-answering
question_answering = pipeline(task="question-answering",
                                model="distilbert-base-uncased-distilled-squad",
                                device=CPU)
context = r"""
 Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
 question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
 a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.
"""

questions = [
    "What is extractive question answering?",
    "What is a good example of a question answering dataset?",
    "How do you fine-tune on SQuAD?"
]

for question in questions:
    answer = question_answering(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {answer['answer']}")
    print(f"Score: {answer['score']}")
    print()
    
# sentiment-analysis
# sentimental-analysis 英文
classifier = pipeline(task="sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english",
                      device=CPU)
preds = classifier("I am really happy today!")
print(preds)

# sentimental-analysis 中文
classifier = pipeline(model="uer/roberta-base-finetuned-jd-binary-chinese", 
                      task="sentiment-analysis", 
                      device=CPU)
preds = classifier("这个餐馆太难吃了。")
print(preds)

# summarization
summarization = pipeline(task="summarization",
                            model="t5-base",
                            device=CPU)
text = r"""
 America has changed dramatically during recent years. Not only has the number of
 graduates in traditional engineering disciplines such as mechanical, civil, electrical,
 chemical, and aeronautical engineering declined, but in most of the premier American
 universities engineering curricula now concentrate on and encourage largely the study
 of engineering science. As a result, there are declining offerings in engineering
 subjects dealing with infrastructure, the environment, and related issues, and greater
 concentration on high technology subjects, largely supporting increasingly complex
 scientific developments. While the latter is important, it should not be at the expense
 of more traditional engineering.
"""
summary = summarization(text)
print(summary)

# text-generation
text_generation = pipeline(task="text-generation",
                            model="gpt2",
                            device=CPU)
text = text_generation("I like to learn data science and AI.")
print(text)

# translation_zh_to_en
translation = pipeline(task="translation_zh_to_en",
                        model="Helsinki-NLP/opus-mt-zh-en",
                        device=CPU)
text = "我喜欢学习数据科学和人工智能。"
translated_text = translation(text)
print(translated_text)

# zero-shot-classification
classifier = pipeline(task="zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        device=CPU)
sequence_to_classify = "Who are you voting for in 2020?"
candidate_labels = ["politics", "public health", "elections"]
hypothesis_template = "This example is {}."
preds = classifier(sequence_to_classify,
                    candidate_labels,
                    hypothesis_template=hypothesis_template)
print(preds)


