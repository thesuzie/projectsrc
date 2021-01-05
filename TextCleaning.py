import pandas as pd
import numpy as np
import re

contentDF = pd.read_csv('./data/rawdataset.csv', header=0, sep=',')


# names=['IdPost','Author','Thread','Timestamp','Content','AuthorNumPosts','AuthorReputation','LastParse','parsed','Site','CitedPost','AuthorName','Likes'])


def remove_replacements(post_content):
    content = post_content
    pattern = '\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*.*\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*'
    cleaned = re.sub(pattern, '', content)
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


content = prepare(contentDF)
content.to_csv('./data/datsetcleaned.csv')


# trim(content,0,1000).to_csv('data0.csv')
# trim(content,1000,2000).to_csv('data1.csv')
# trim(content,2000,3000).to_csv('data2.csv')
# trim(content,3000,4000).to_csv('data3.csv')
# trim(content,4000,5000).to_csv('data4.csv')
# trim(content,5000,6000).to_csv('data5.csv')
# trim(content,6000,7000).to_csv('data6.csv')
# trim(content,7000,8000).to_csv('data7.csv')
# trim(content,8000,9000).to_csv('data8.csv')
# trim(content,9000,10000).to_csv('data9.csv')


# for counting labels:
def count_labels(txt, frm, to):
    section = contentDF.loc[frm:to]
    (unique, counts) = np.unique(section['Label'], return_counts=True)
    print('label, Count of labels')
    print(np.asarray((unique, counts)).T)

# count_labels(contentDF, 0, 200)
