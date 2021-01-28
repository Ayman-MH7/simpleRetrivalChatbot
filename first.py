#loading the data preprocessing libs
import pandas as pd
import numpy as np
#loading the vectorizer and similarity measures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#downloading the database
df = pd.read_csv('aws_faq.csv', delimiter="\t")
df.dropna(inplace=True)
#intializing the vectorizer 
vectorizer = TfidfVectorizer()
vectorizer.fit(np.concatenate((df.Question, df.Answer)))
#vectorize the Quesitons with no need to make a binary array
question_vector = vectorizer.transform(df.Question)
#chat with the user
print("now you can chat with the user")
while True:
    #read input from the user
    input_question = input('>> ')
    if(input_question == 'quit'.lower()):
        break
    else:
        #vectorizing the input question
        input_question_vector = vectorizer.transform([input_question])
        #make a cosine similarity for the input_question with the df.Question
        similarities = cosine_similarity(input_question_vector, question_vector)
        #finding the most similar Question
        closest = np.argmax(similarities, axis=1)
        #printing the bot response 
        print("BOT: " + df.Answer.iloc[closest].values[0])