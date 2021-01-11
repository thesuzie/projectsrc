import pandas as pd
import numpy as np
import re

contentDF = pd.read_csv('./data/extremedatasetraw.csv', header=0, sep=',')


# names=['IdPost','Author','Thread','Timestamp','Content','AuthorNumPosts','AuthorReputation','LastParse','parsed','Site','CitedPost','AuthorName','Likes'])


def remove_replacements(post_content):
    # first attempt to remove things:
    content = post_content

    #pattern = '\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*.*\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*'
    #cleaned = re.sub(pattern, '', content)
    indicator = '\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*'
    without_indicator = re.sub(indicator," ", content)
    links = '\[\S*\]'
    cleaned = re.sub(links," ",without_indicator)
    images = "Imgur: The magic of the Internet"
    cleaned = re.sub(images,"", cleaned)
    extra_links = "http\S*" #need to update to try and capture all links
    cleaned = re.sub(extra_links,"",cleaned)


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
    removedLinks =[remove_links(c) for c in txt['content']]
    txt['Cleaned Content'] = removedLinks

    default = [0] * len(txt['content'])
    txt['Label'] = default
    txt['Sentiment'] = default
    txt = txt.drop(['content'], axis=1)

    return txt


extreme = prepare_extreme(contentDF)
extreme.to_csv('./data/extremecleaned.csv')
#content = prepare(contentDF)
#content.to_csv('./data/datsetcleaned.csv')


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

# for counting labels:
def count_labels(txt, frm, to):
    section = contentDF.loc[frm:to]
    (unique, counts) = np.unique(section['Label'], return_counts=True)
    print('label, Count of labels')
    print(np.asarray((unique, counts)).T)

# count_labels(contentDF, 0, 200)
