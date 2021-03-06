import pandas as pd
import numpy as np
import re
from evaluation import count_sent, count_labels
from sklearn.model_selection import train_test_split


# names=['IdPost','Author','Thread','Timestamp','Content','AuthorNumPosts','AuthorReputation','LastParse','parsed','Site','CitedPost','AuthorName','Likes'])

def remove_replacements(post_content):
    # first attempt to remove things:
    content = post_content

    # pattern = '\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*.*\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*'
    # cleaned = re.sub(pattern, '', content)
    indicator = '\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*'
    without_indicator = re.sub(indicator, " ", content)
    links = '\[\S*\]'
    cleaned = re.sub(links, " ", without_indicator)
    images = "Imgur: The magic of the Internet"
    cleaned = re.sub(images, "", cleaned)
    extra_links = "http\S*"  # need to update to try and capture all links
    cleaned = re.sub(extra_links, "", cleaned)

    return cleaned


def prepare(txt):
    removedlinks = [remove_replacements(c) for c in txt['Content']]
    txt['Content Cleaned'] = removedlinks
    default = [0] * len(txt['Content'])
    txt['Label'] = default
    sentiment = [0] * len(txt['Content'])
    txt['Sentiment'] = sentiment
    txt = txt.drop(['Content'], axis=1)
    return txt


def trim(txt, i, j):
    return txt.loc[i:j]


def remove_links(content):
    links = "(http\S*)|(\S*.com\S*)|(\S*.net\S*)"
    cleaned = re.sub(links, "", content)
    return cleaned


def prepare_extreme(txt):
    removedLinks = [remove_links(c) for c in txt['content']]
    txt['Cleaned Content'] = removedLinks

    default = [0] * len(txt['content'])
    txt['Label'] = default
    txt['Sentiment'] = default
    txt = txt.drop(['content'], axis=1)

    return txt


def secondary_cleaning(content):
    # after further exploratory analysis have decided to remove certain phrases

    links = "(http\S*)|(\S*\.co\S*)|(\S*\.net\S*)"
    cleaned = re.sub(links, "", content)

    # remove usernames and emails (strings containing @)
    usrnm = "\S*@\S*"
    cleaned = re.sub(usrnm, "", cleaned)

    # remove unneeded indicator words that have been added
    ignore = "(Start contract)|(Code \[.*\])|(Spoiler \(Click to View\))"
    cleaned = re.sub(ignore, "", cleaned)

    # remove punctuation and numbers
    cleaned = re.sub('[^a-zA-Z]+', " ", cleaned)

    # make lower case
    clean_content = cleaned.lower()

    return clean_content

# extreme = prepare_extreme(contentDF)
# extreme.to_csv('./data/extremecleaned.csv')
# content = prepare(contentDF)
# content.to_csv('./data/datsetcleaned.csv')


# trim(contentDF,0,1000).to_csv('data/data0.csv', index=False)
# trim(contentDF,1000,2000).to_csv('data/data1.csv', index=False)
# trim(contentDF,2000,3000).to_csv('data/data2.csv', index=False)
# trim(contentDF,3000,4000).to_csv('data/data3.csv', index=False)
# trim(contentDF,4000,5000).to_csv('data/data4.csv', index=False)
# trim(contentDF,5000,6000).to_csv('data/data5.csv', index=False)
# trim(contentDF,6000,7000).to_csv('data/data6.csv', index=False)
# trim(contentDF,7000,8000).to_csv('data/data7.csv', index=False)
# trim(contentDF,8000,9000).to_csv('data/data8.csv', index=False)
# trim(contentDF,9000,10000).to_csv('data/data9.csv', index=False)
#

def combine_annotations(file1, file2):
    # INPUT: file1, file2 dataframes
    # OUTPUT: new dataframe with labels from file 2 in label1 sent1 columns and final columns
    combined = file1
    combined['Label 1'] = file2['Label']
    combined['Sentiment 1'] = file2['Sentiment']

    final_l = [0] * len(file1['Label'])
    final_s = [0] * len(file1['Label'])
    i = 0
    while i < len(file1['Label']):
        if combined['Label'][i] == combined['Label 1'][i]:
            final_l[i] = combined['Label'][i]
        else:
            final_l[i] = "X"

        if combined['Sentiment'][i] == combined['Sentiment 1'][i]:
            final_s[i] = combined['Sentiment'][i]
        else:
            final_s[i] = "X"
        i += 1

    combined["Final Label"] = final_l

    combined['Final Sentiment'] = final_s

    return combined


# data0 = pd.read_csv('./data/data0final.csv', header=0, sep=',')
# data1 = pd.read_csv('./data/data1final.csv', header=0, sep=',')
# data2 = pd.read_csv('./data/data2final.csv', header=0, sep=',')
# data3 = pd.read_csv('./data/data3final.csv', header=0, sep=',')
# dataext = pd.read_csv('./data/extremeAnnotated.csv', header=0, sep=',')
#
# dataext.rename(columns={'id': 'IdPost', 'thread_id': 'Thread'}, inplace=True)
# df = data0.append(data1, ignore_index=True, sort=False)
# df1 = df.append(data2, ignore_index=True, sort=False)
# df2 = df1.append(data3, ignore_index=True, sort=False)
# df_complete = df2.append(dataext, ignore_index=True, sort=False)
# df_complete.drop(['Label 1', 'Sentiment 1', 'Label 2', 'Sentiment 2','Negative posts', 'Positive posts'], axis=1, inplace=True)
#
# count_labels(df_complete)
# count_sent(df_complete)
# df_complete.to_csv('./data/FullAnnotatedData.csv', index=False)

#df_complete = pd.read_csv('./data/FullAnnotatedData.csv')
#df_no0 = df_complete[df_complete.Label != 0]
# count_labels(df_complete)
#
# count_labels(df_no0)
# count_sent(df_no0)
#
# df_no0.to_csv('./data/reducedFullAnnotations.csv', index=False)
#

# data = pd.read_csv('./data/reducedFullAnnotations.csv')
#
# X_train, X_test, y_train, y_test = train_test_split(data["Content Cleaned"], data["Label"],
#                                                     random_state=42,
#                                                     stratify=data["Label"],
#                                                     test_size=0.2)
#
# X_train.to_csv("./test_train/X_train_Imbalanced.csv", header=True)
# X_test.to_csv("./test_train/X_test_Imbalanced.csv",header=True)
#
# y_train.to_csv("./test_train/y_train_Imbalanced.csv", header=True)
# y_test.to_csv("./test_train/y_test_Imbalanced.csv", header=True)
#
# print("y_train")
# (unique, counts) = np.unique(y_train, return_counts=True)
# print(np.asarray((unique, counts)).T)
#
# print("y_test")
# (unique, counts) = np.unique(y_test, return_counts=True)
# print(np.asarray((unique,# data = pd.read_csv('./data/reducedFullAnnotations.csv')
# #
# # X_train, X_test, y_train, y_test = train_test_split(data["Content Cleaned"], data["Label"],
# #                                                     random_state=42,
# #                                                     stratify=data["Label"],
# #                                                     test_size=0.2)
# #
# # X_train.to_csv("./test_train/X_train_Imbalanced.csv", header=True)
# # X_test.to_csv("./test_train/X_test_Imbalanced.csv",header=True)
# #
# # y_train.to_csv("./test_train/y_train_Imbalanced.csv", header=True)
# # y_test.to_csv("./test_train/y_test_Imbalanced.csv", header=True)
# #
# # print("y_train")
# # (unique, counts) = np.unique(y_train, return_counts=True)
# # print(np.asarray((unique, counts)).T)
# #
# # print("y_test")
# # (unique, counts) = np.unique(y_test, return_counts=True)
# # print(np.asarray((unique, counts)).T) counts)).T)

# train = pd.read_csv("./test_train/X_train_Imbalanced.csv")
#
# test = pd.read_csv("./test_train/X_test_Imbalanced.csv")
#
# clean_test_content = [secondary_cleaning(c) for c in test["Content Cleaned"]]
# clean_train_content = [secondary_cleaning(c) for c in train["Content Cleaned"]]
#
# test["Content Cleaned"] = clean_test_content
# train["Content Cleaned"] = clean_train_content
#
# test.to_csv("./test_train/X_test_IM_clean.csv", header=True)
# train.to_csv("./test_train/X_train_IM_clean.csv", header=True)

# test_x = pd.read_csv("./test_train/X_test_IM_clean.csv")
# test_y = pd.read_csv("./test_train/y_test_Imbalanced.csv")
#
# ys = test_y["Label"]
#
# test_x["Label"] = ys
#
# test_x.to_csv("./test_train/test_IM.csv", header=True, index=False)
#
#
#
# train_x = pd.read_csv("./test_train/X_train_IM_clean.csv")
# train_y = pd.read_csv("./test_train/y_train_Imbalanced.csv")
# ys2 = train_y["Label"]
#
# train_x["Label"] = ys2
#
# train_x.to_csv("./test_train/train_IM.csv", header=True, index=False)

encode = "test"
ant = ""
file = f"./test_train/{encode}_NB_{ant}.csv"
print(file)

