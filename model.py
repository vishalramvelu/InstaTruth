import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

#this is base likelihood based on textual data



fake_news = pd.read_csv('Fake.csv')
true_news = pd.read_csv('True.csv')


fake_news['class'] = 0
true_news['class'] = 1

df_merge = pd.concat([fake_news, true_news], axis = 0)

df = df_merge.drop(["title", "subject","date"], axis = 1)
news=df[df['class']==1]['text'][100]

import re
import string

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# Sample text for testing
sample_text = """


An AI-generated image depicting tent camps for displaced Palestinians and a slogan that reads All Eyes on Rafah is sweeping social media.
The post has been shared more than 47 million times by Instagram users including celebrities like Dua Lipa, Lewis Hamilton and Gigi and Bella Hadid.
The image and the slogan went viral after an Israeli air strike and resulting fire at a camp for displaced Palestinians in the southern Gaza city of Rafah earlier this week.
The Hamas-run health ministry said at least 45 people were killed and hundreds more wounded in the incident. Israel said it had targeted two Hamas commanders, and that the deadly fire was possibly caused by a secondary explosion.
There has been widespread international condemnation of the Israeli strike, which Israeli Prime Minister Benjamin Netanyahu called a “tragic mishap”.


"""""
# Process the sample text
processed_text = wordopt(sample_text)
df["text"] = df["text"].apply(wordopt)
x = df["text"]
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True,stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
xv_val = vectorization.transform(x_val)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)
pred_lr=LR.predict(xv_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error

mae=mean_absolute_error(pred_lr,y_test)
mse=mean_squared_error(pred_lr,y_test)
print(mae,mse) #least the better

LR.score(xv_test, y_test) #more the better

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier()
GBC.fit(xv_train, y_train)

pred_gbc = GBC.predict(xv_test)
GBC.score(xv_test, y_test)

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=100)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)

print("LR accuracy:", LR.score(xv_test, y_test))
print("GBC accuracy:", GBC.score(xv_test, y_test))
print("RFC accuracy:", RFC.score(xv_test, y_test))

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    print(new_def_test.head())
    
    # vectorize
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    # hard predictions
    pred_LR  = LR.predict(new_xv_test)[0]
    pred_GBC = GBC.predict(new_xv_test)[0]
    pred_RFC = RFC.predict(new_xv_test)[0]
    
    # probabilities
    prob_LR  = LR.predict_proba(new_xv_test)[0]
    prob_GBC = GBC.predict_proba(new_xv_test)[0]
    prob_RFC = RFC.predict_proba(new_xv_test)[0]
    
    # print them all
    print(
        "\n\nLR   → {:<12} (fake={:.2f}, real={:.2f})\n"
        "GBC  → {:<12} (fake={:.2f}, real={:.2f})\n"
        "RFC  → {:<12} (fake={:.2f}, real={:.2f})"
        .format(
            output_lable(pred_LR), prob_LR[0],  prob_LR[1],
            output_lable(pred_GBC), prob_GBC[0], prob_GBC[1],
            output_lable(pred_RFC), prob_RFC[0], prob_RFC[1],
        )
    )
    
    # final override at P(real) ≥ 0.2
    final_label = "Not A Fake News" if prob_RFC[1] >= 0.15 else "Fake News"
    print(f"\nFinal (LR @ thresh=0.2): {final_label}")



#summarizing



from transformers import pipeline

video_text = "LeBron James’s unparalleled blend of size, athleticism, and basketball IQ has " \
"allowed him to dominate every facet of the game—scoring, playmaking, rebounding, and defense—across a remarkable 20-year career." \
" He has led three different franchises to four NBA championships, proving his ability to elevate any team to the highest level. " \
"With four league MVP awards and 19 All-Star selections, his individual honors reflect sustained excellence and consistency at an elite standard. " \
"In February 2023, he surpassed Kareem Abdul-Jabbar to become the NBA’s all-time leading scorer, underscoring both his longevity and scoring " \
"prowess. His playoff performances—highlighted by multiple 40-point elimination games and historic triple-doubles—showcase his clutch impact " \
"when it matters most. Beyond raw statistics, LeBron’s adaptability—from high-flying scorer to pass-first facilitator—emphasizes his all-around" \
"greatness. Off the court, his leadership extends into philanthropy and activism through initiatives like the I PROMISE School and vocal advocacy " \
"for social justice. Taken together—statistical dominance, team success, longevity, versatility, and cultural influence—LeBron James’s career " \
"makes a compelling case for him as the greatest basketball player of all time."

summarizer = pipeline("summarization", model = "facebook/bart-large-cnn") #load the summarizer process w pretrained tools
summary = summarizer(video_text, max_length = 150, min_length = 30)[0]['summary_text'] #run summarizer on text to get desired output of x length
print(summary)
manual_testing(summary)

#search web for verification





