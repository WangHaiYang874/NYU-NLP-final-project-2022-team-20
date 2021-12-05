import pandas as pd
import re


def return_emoticons(text):
    emoticons = "(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)"
    another_one = "¯\\\_(ツ)_/¯"
    emojis = re.findall(emoticons, text)
    emojis.extend(re.findall(another_one, text))
    return emojis


#read all data
column_list = ["author_flair_text","body", "subreddit"]
df = pd.read_csv("mbti_full_pull.csv", usecols = column_list)
texts = df["body"].to_list()
emoticons = []

for i in range(len(texts)):
    emoticons.append(return_emoticons(str(texts[i])))


print(emoticons)
df.insert(1,"emoticons", emoticons)

new_mbti = open("mbti_full_pull_new.csv", "w")
df.to_csv("mbti_full_pull_new.csv")






